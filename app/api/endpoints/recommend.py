from fastapi import APIRouter, Depends
import logging

from app.features.recommendation.schema import BatchRecommendPayload, BatchRecommendResponse
from app.features.recommendation.service import RecommendationService

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/batch-compute", response_model=BatchRecommendResponse)
async def compute_batch_recommendations(
    request: BatchRecommendPayload,
    service: RecommendationService = Depends()
):
    try:
        logger.info("Received batch computation request")
        response = await service.compute_batch_recommendations(request)
        return response
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        raise
