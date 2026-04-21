import asyncio
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "fastapi" not in sys.modules:
    fastapi_stub = types.ModuleType("fastapi")

    def _depends(*args, **kwargs):
        return None

    class _HTTPException(Exception):
        def __init__(self, status_code=None, detail=None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    fastapi_stub.Depends = _depends
    fastapi_stub.HTTPException = _HTTPException
    fastapi_stub.status = SimpleNamespace(HTTP_500_INTERNAL_SERVER_ERROR=500)
    sys.modules["fastapi"] = fastapi_stub

if "app.features.search.service" not in sys.modules:
    search_service_stub = types.ModuleType("app.features.search.service")

    class _SearchService:
        pass

    search_service_stub.SearchService = _SearchService
    sys.modules["app.features.search.service"] = search_service_stub

if "app.core.config" not in sys.modules:
    config_stub = types.ModuleType("app.core.config")
    config_stub.settings = SimpleNamespace(
        REC_MIN_BEHAVIOR_WEIGHT=0.2,
        REC_MAX_BEHAVIOR_WEIGHT=0.8,
        REC_FULL_BEHAVIOR_DENSITY=10.0,
    )
    sys.modules["app.core.config"] = config_stub

from app.features.recommendation.schema import (
    BatchRecommendPayload,
    UserInteraction,
    UserProfile,
)
from app.features.recommendation.service import RecommendationService


class FakeEmbedding:
    def generate_vector(self, interests: str):
        return [0.0, 1.0]


class FakeRepository:
    def __init__(self):
        self.client = SimpleNamespace(retrieve=lambda **kwargs: [])

    def search(self, **kwargs):
        return []


class FakeSearchService:
    def __init__(self):
        self.embedding = FakeEmbedding()
        self.repository = FakeRepository()


class RecommendationServiceUnitTests(unittest.TestCase):
    def setUp(self):
        self.service = RecommendationService(search_service=FakeSearchService())

    def test_build_behavior_centroid_weighted_and_normalized(self):
        interactions = [
            UserInteraction(user_id="u1", path_id="p1", score=3.0),
            UserInteraction(user_id="u1", path_id="p2", score=1.0),
        ]

        with patch.object(
            self.service,
            "_fetch_path_vectors",
            return_value={"p1": [1.0, 0.0], "p2": [0.0, 1.0]},
        ):
            centroid = self.service._build_behavior_centroid(interactions)

        self.assertIsNotNone(centroid)
        self.assertAlmostEqual(centroid[0], 0.9486832980, places=6)
        self.assertAlmostEqual(centroid[1], 0.3162277660, places=6)

    def test_blend_user_centroid_dynamic_weight_changes_by_density(self):
        behavior = [1.0, 0.0]
        onboarding = [0.0, 1.0]

        low_density = self.service._blend_user_centroid(
            behavior_vector=behavior,
            onboarding_vector=onboarding,
            interactions_count=1,
            interactions_score_sum=1.0,
        )
        high_density = self.service._blend_user_centroid(
            behavior_vector=behavior,
            onboarding_vector=onboarding,
            interactions_count=10,
            interactions_score_sum=10.0,
        )

        self.assertIsNotNone(low_density)
        self.assertIsNotNone(high_density)
        self.assertGreater(low_density[1], low_density[0])
        self.assertGreater(high_density[0], high_density[1])

    def test_dynamic_weights_increase_with_score_strength(self):
        low_score_behavior_w, low_score_onboarding_w = self.service._calculate_dynamic_weights(
            interactions_count=2,
            interactions_score_sum=2.0,
            has_behavior_vector=True,
            has_onboarding_vector=True,
        )

        high_score_behavior_w, high_score_onboarding_w = self.service._calculate_dynamic_weights(
            interactions_count=2,
            interactions_score_sum=8.0,
            has_behavior_vector=True,
            has_onboarding_vector=True,
        )

        self.assertGreater(high_score_behavior_w, low_score_behavior_w)
        self.assertLess(high_score_onboarding_w, low_score_onboarding_w)

    def test_filter_interacted_paths(self):
        candidates = ["p1", "p2", "p3", "p4"]
        interacted = {"p2", "p4"}

        filtered = self.service._filter_interacted_paths(candidates, interacted, top_k=2)
        self.assertEqual(filtered, ["p1", "p3"])

    def test_popular_fallback_for_cold_start_user(self):
        payload = BatchRecommendPayload(
            users_interactions=[
                UserInteraction(user_id="u1", path_id="pA", score=2.0),
                UserInteraction(user_id="u2", path_id="pB", score=5.0),
                UserInteraction(user_id="u3", path_id="pA", score=1.0),
            ],
            users_profiles=[
                UserProfile(user_id="cold_user", interests=""),
            ],
        )

        response = asyncio.run(self.service.compute_batch_recommendations(payload))

        self.assertTrue(response.success)
        self.assertEqual(response.data[0].user_id, "cold_user")
        self.assertEqual(response.data[0].recommended_paths[:2], ["pB", "pA"])


if __name__ == "__main__":
    unittest.main()
