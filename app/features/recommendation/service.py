import logging
from collections import defaultdict
from math import sqrt
from fastapi import Depends, HTTPException, status
from typing import Dict, List, Optional, Set, Tuple

from app.core.config import settings
from app.features.search.service import SearchService
from .schema import (
    BatchRecommendPayload,
    BatchRecommendResponse,
    BatchRecommendationResult,
    RecommendedPathScore,
)

logger = logging.getLogger(f"passion-tree-ai.{__name__}")

class RecommendationService:
    def __init__(self, search_service: SearchService = Depends()):
        self.search_service = search_service
        self.source_context = "app.features.recommendation.service"

        # Deterministic blend weights for user centroid construction
        self.min_behavior_weight = settings.REC_MIN_BEHAVIOR_WEIGHT
        self.max_behavior_weight = settings.REC_MAX_BEHAVIOR_WEIGHT
        self.full_behavior_density = max(settings.REC_FULL_BEHAVIOR_DENSITY, 1.0)
        self.default_top_k = 10
        self.candidate_multiplier = 4
        self.collection_name = "learning_paths"

        if self.min_behavior_weight > self.max_behavior_weight:
            self.min_behavior_weight, self.max_behavior_weight = (
                self.max_behavior_weight,
                self.min_behavior_weight,
            )

    @staticmethod
    def _normalize_vector(vector: Optional[List[float]]) -> Optional[List[float]]:
        if not vector:
            return None
        norm = sqrt(sum(v * v for v in vector))
        if norm == 0:
            return None
        return [v / norm for v in vector]

    @staticmethod
    def _extract_point_vector(point) -> Optional[List[float]]:
        vector = getattr(point, "vector", None)
        if vector is None:
            return None
        if isinstance(vector, dict):
            if not vector:
                return None
            vector = next(iter(vector.values()))
        return list(vector)

    def _fetch_path_vectors(self, path_ids: Set[str]) -> Dict[str, List[float]]:
        if not path_ids:
            return {}

        try:
            points = self.search_service.repository.client.retrieve(
                collection_name=self.collection_name,
                ids=list(path_ids),
                with_vectors=True,
                with_payload=False,
            )
        except Exception as exc:
            logger.error(f"[{self.source_context}] Failed to retrieve vectors: {exc}")
            return {}

        vectors: Dict[str, List[float]] = {}
        for point in points:
            vec = self._extract_point_vector(point)
            if vec:
                vectors[str(point.id)] = vec
        return vectors

    def _build_behavior_centroid(self, interactions) -> Optional[List[float]]:
        if not interactions:
            return None

        path_ids = {str(interaction.path_id) for interaction in interactions}
        path_vectors = self._fetch_path_vectors(path_ids)
        if not path_vectors:
            return None

        weighted_sum: Optional[List[float]] = None
        total_weight = 0.0

        for interaction in interactions:
            vec = path_vectors.get(str(interaction.path_id))
            if not vec:
                continue

            weight = max(float(interaction.score), 0.0)
            if weight == 0:
                continue

            if weighted_sum is None:
                weighted_sum = [0.0] * len(vec)

            if len(weighted_sum) != len(vec):
                logger.warning(
                    f"[{self.source_context}] Vector dimension mismatch for path_id={interaction.path_id}"
                )
                continue

            for i, value in enumerate(vec):
                weighted_sum[i] += value * weight
            total_weight += weight

        if not weighted_sum or total_weight == 0:
            return None

        centroid = [value / total_weight for value in weighted_sum]
        return self._normalize_vector(centroid)

    def _build_onboarding_vector(self, interests: str) -> Optional[List[float]]:
        if not interests or not interests.strip():
            return None

        try:
            vector = self.search_service.embedding.generate_vector(interests)
            return self._normalize_vector(list(vector))
        except Exception as exc:
            logger.error(
                f"[{self.source_context}] Failed to generate onboarding vector: {exc}"
            )
            return None

    def _blend_user_centroid(
        self,
        behavior_vector: Optional[List[float]],
        onboarding_vector: Optional[List[float]],
        interactions_count: int,
        interactions_score_sum: float,
    ) -> Optional[List[float]]:
        behavior_w, onboarding_w = self._calculate_dynamic_weights(
            interactions_count=interactions_count,
            interactions_score_sum=interactions_score_sum,
            has_behavior_vector=behavior_vector is not None,
            has_onboarding_vector=onboarding_vector is not None,
        )

        if behavior_vector and onboarding_vector:
            if len(behavior_vector) != len(onboarding_vector):
                logger.warning(
                    f"[{self.source_context}] Cannot blend vectors with different dimensions"
                )
                return behavior_vector

            combined = [
                (behavior_w * behavior_vector[i])
                + (onboarding_w * onboarding_vector[i])
                for i in range(len(behavior_vector))
            ]
            return self._normalize_vector(combined)

        if behavior_vector:
            return behavior_vector
        if onboarding_vector:
            return onboarding_vector
        return None

    def _calculate_dynamic_weights(
        self,
        interactions_count: int,
        interactions_score_sum: float,
        has_behavior_vector: bool,
        has_onboarding_vector: bool,
    ) -> Tuple[float, float]:
        if has_behavior_vector and not has_onboarding_vector:
            return 1.0, 0.0
        if has_onboarding_vector and not has_behavior_vector:
            return 0.0, 1.0
        if not has_behavior_vector and not has_onboarding_vector:
            return 0.0, 0.0

        count_density = min(max(interactions_count, 0) / self.full_behavior_density, 1.0)
        score_density = min(max(interactions_score_sum, 0.0) / self.full_behavior_density, 1.0)

        # Combine amount of behavior and strength of behavior into one density value.
        density = (count_density + score_density) / 2.0

        # Fewer interactions => rely more on onboarding vector.
        behavior_dynamic = self.min_behavior_weight + (
            (self.max_behavior_weight - self.min_behavior_weight) * density
        )

        behavior_weight = behavior_dynamic
        onboarding_weight = 1.0 - behavior_weight
        return behavior_weight, onboarding_weight

    def _get_global_popular_paths(
        self, interactions, top_k: int
    ) -> List[str]:
        if not interactions:
            return []

        popularity: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"score_sum": 0.0, "count": 0.0}
        )

        for interaction in interactions:
            path_id = str(interaction.path_id)
            popularity[path_id]["score_sum"] += max(float(interaction.score), 0.0)
            popularity[path_id]["count"] += 1.0

        ranked = sorted(
            popularity.items(),
            key=lambda item: (
                item[1]["score_sum"],
                item[1]["count"],
                item[0],
            ),
            reverse=True,
        )
        return [path_id for path_id, _ in ranked[:top_k]]

    @staticmethod
    def _filter_interacted_scored(
        candidates: List[Tuple[str, float]], interacted_paths: Set[str], top_k: int
    ) -> List[Tuple[str, float]]:
        filtered: List[Tuple[str, float]] = []
        for path_id, score in candidates:
            if path_id in interacted_paths:
                continue
            filtered.append((path_id, score))
            if len(filtered) == top_k:
                break
        return filtered

    def _query_candidates(
        self, user_centroid: List[float], interacted_paths: Set[str], top_k: int
    ) -> List[Tuple[str, float]]:
        try:
            candidate_limit = max(top_k * self.candidate_multiplier, top_k)
            search_results = self.search_service.repository.search(
                collection_name=self.collection_name,
                query_vector=user_centroid,
                top_k=candidate_limit + len(interacted_paths),
                with_payload=False,
            )

            recommended: List[Tuple[str, float]] = [
                (str(result.id), float(result.score)) for result in search_results
            ]
            return self._filter_interacted_scored(recommended, interacted_paths, top_k)
        except Exception as exc:
            logger.error(f"[{self.source_context}] Candidate query failed: {exc}")
            return []

    async def compute_batch_recommendations(
        self, payload: BatchRecommendPayload
    ) -> BatchRecommendResponse:
        logger.info(
            f"[{self.source_context}] Starting deterministic batch recommendation for {len(payload.users_profiles)} users"
        )

        try:
            interactions_by_user = defaultdict(list)
            for interaction in payload.users_interactions:
                interactions_by_user[interaction.user_id].append(interaction)

            global_popular_paths = self._get_global_popular_paths(
                payload.users_interactions,
                top_k=self.default_top_k * self.candidate_multiplier,
            )

            results = []

            for profile in payload.users_profiles:
                original_user_id = profile.user_id
                interactions = interactions_by_user.get(original_user_id, [])
                interacted_paths = {str(interaction.path_id) for interaction in interactions}
                interactions_score_sum = sum(max(float(interaction.score), 0.0) for interaction in interactions)

                behavior_vector = self._build_behavior_centroid(interactions)
                onboarding_vector = self._build_onboarding_vector(profile.interests)
                user_centroid = self._blend_user_centroid(
                    behavior_vector=behavior_vector,
                    onboarding_vector=onboarding_vector,
                    interactions_count=len(interactions),
                    interactions_score_sum=interactions_score_sum,
                )

                if user_centroid is None:
                    if not behavior_vector and not onboarding_vector:
                        logger.info(
                            f"[{self.source_context}] Using popular fallback for cold-start user_id={original_user_id}"
                        )
                        fallback_ids = self._filter_interacted_paths(
                            global_popular_paths,
                            interacted_paths,
                            self.default_top_k,
                        )
                        top_paths = [(pid, 0.0) for pid in fallback_ids]
                    else:
                        logger.warning(
                            f"[{self.source_context}] Unable to build centroid for user_id={original_user_id}"
                        )
                        top_paths = []
                else:
                    top_paths = self._query_candidates(
                        user_centroid=user_centroid,
                        interacted_paths=interacted_paths,
                        top_k=self.default_top_k,
                    )

                    if not top_paths:
                        fallback_ids = self._filter_interacted_paths(
                            global_popular_paths,
                            interacted_paths,
                            self.default_top_k,
                        )
                        top_paths = [(pid, 0.0) for pid in fallback_ids]

                results.append(
                    BatchRecommendationResult(
                        user_id=original_user_id,
                        recommended_paths=[
                            RecommendedPathScore(path_id=pid, score=score)
                            for pid, score in top_paths
                        ],
                    )
                )

            logger.info(
                f"[{self.source_context}] Batch computation completed. Processed {len(results)} users."
            )
            return BatchRecommendResponse(
                success=True,
                message="Deterministic centroid vector recommendations generated successfully",
                data=results,
            )

        except Exception as e:
            logger.error(f"[{self.source_context}] Computation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch computation failed: {str(e)}",
            )
