import logging
import pandas as pd
from scipy.sparse import csr_matrix
import implicit
from fastapi import Depends, HTTPException, status
from typing import List, Dict, Set

from app.features.search.service import SearchService
from .schema import (
    BatchRecommendPayload,
    BatchRecommendResponse,
    BatchRecommendationResult,
)

logger = logging.getLogger(f"passion-tree-ai.{__name__}")

class RecommendationService:
    def __init__(self, search_service: SearchService = Depends()):
        self.search_service = search_service
        self.source_context = "app.features.recommendation.service"

        self.alpha = 0.6  # (ALS) 60%
        self.beta = 0.4  # (Content-Based) 40%

    def _normalize_scores(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        if not raw_scores:
            return {}

        vals = list(raw_scores.values())
        min_v, max_v = min(vals), max(vals)

        if max_v == min_v:
            return {k: 1.0 for k in raw_scores.keys()}

        return {k: (v - min_v) / (max_v - min_v) for k, v in raw_scores.items()}

    def _get_collaborative_scores(
        self, model, u_idx: int, user_items: csr_matrix, path_map: dict, top_k: int = 30
    ) -> Dict[str, float]:
        path_indices, scores = model.recommend(
            u_idx, user_items, N=top_k, filter_already_liked_items=True
        )

        raw_scores = {
            str(path_map[path_indices[i]]): float(scores[i])
            for i in range(len(path_indices))
        }
        return self._normalize_scores(raw_scores)

    async def _get_content_based_scores(
        self, interests: str, interacted_paths: Set[str], top_k: int = 30
    ) -> Dict[str, float]:
        try:
            search_resp = await self.search_service.search(
                query=interests,
                top_k=top_k + len(interacted_paths),
                resource_type="learning_paths",
            )

            raw_scores = {}
            for result in search_resp.results:
                path_id = str(result.id)
                if path_id not in interacted_paths:
                    raw_scores[path_id] = float(result.score)
                    if len(raw_scores) == top_k:
                        break

            return self._normalize_scores(raw_scores)
        except Exception as e:
            logger.error(f"[{self.source_context}] Content-Based search failed: {e}")
            return {}

    async def compute_batch_recommendations(
        self, payload: BatchRecommendPayload
    ) -> BatchRecommendResponse:
        logger.info(
            f"[{self.source_context}] Starting BATCH WEIGHTED HYBRID for {len(payload.users_interactions)} interactions"
        )

        try:
            user_ids = pd.Index([])
            model = None
            sparse_user_item = None
            path_map = {}
            df = pd.DataFrame()

            if payload.users_interactions:
                df = pd.DataFrame([vars(i) for i in payload.users_interactions])

                user_ids = df["user_id"].astype("category").cat.categories
                path_ids = df["path_id"].astype("category").cat.categories
                path_map = {idx: val for idx, val in enumerate(path_ids)}

                df["user_idx"] = df["user_id"].astype("category").cat.codes
                df["path_idx"] = df["path_id"].astype("category").cat.codes

                sparse_item_user = csr_matrix(
                    (df["score"].astype(float), (df["path_idx"], df["user_idx"])),
                    shape=(len(path_ids), len(user_ids)),
                )
                sparse_user_item = sparse_item_user.T.tocsr()

                logger.info(f"[{self.source_context}] Training ALS Model...")
                model = implicit.als.AlternatingLeastSquares(
                    factors=64, regularization=0.1, iterations=20, random_state=42
                )
                model.fit(sparse_item_user)

            results = []

            for profile in payload.users_profiles:
                original_user_id = profile.user_id

                cf_scores = {}
                cb_scores = {}
                interacted_paths = set()

                if not df.empty and original_user_id in user_ids.values:
                    interacted_paths = set(
                        df[df["user_id"] == original_user_id]["path_id"]
                        .astype(str)
                        .values
                    )
                    u_idx = user_ids.get_loc(original_user_id)

                    cf_scores = self._get_collaborative_scores(
                        model, u_idx, sparse_user_item[u_idx], path_map
                    )

                cb_scores = await self._get_content_based_scores(
                    profile.interests, interacted_paths
                )

                final_scores = {}
                all_candidate_paths = set(cf_scores.keys()).union(set(cb_scores.keys()))

                for path in all_candidate_paths:
                    score_cf = cf_scores.get(path, 0.0)
                    score_cb = cb_scores.get(path, 0.0)

                    if not interacted_paths:
                        final_score = score_cb
                    else:
                        final_score = (score_cf * self.alpha) + (score_cb * self.beta)

                    final_scores[path] = final_score

                sorted_paths = sorted(
                    final_scores.items(), key=lambda item: item[1], reverse=True
                )
                top_10_paths = [path for path, score in sorted_paths[:10]]

                results.append(
                    BatchRecommendationResult(
                        user_id=original_user_id, recommended_paths=top_10_paths
                    )
                )

            logger.info(
                f"[{self.source_context}] Batch computation completed. Processed {len(results)} users."
            )
            return BatchRecommendResponse(
                success=True,
                message="Weighted Hybrid batch recommendations generated successfully",
                data=results,
            )

        except Exception as e:
            logger.error(f"[{self.source_context}] Computation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch computation failed: {str(e)}",
            )
