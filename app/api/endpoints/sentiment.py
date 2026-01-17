from fastapi import APIRouter, HTTPException, status, Depends
import logging

from app.features.sentiment_analysis.schema import SentimentRequest, SentimentResponse
from app.features.sentiment_analysis.service import SentimentService

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    service: SentimentService = Depends()
):
    try:
        logger.info(f"Analyzing reflection for: {request.what_learned}")
        result = await service.analyze_reflection(request)
        return result
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment analysis service is temporarily unavailable. Please try again later."
        )

# Debug endpoint to check collection info (reuse logic from search.py)
@router.get("/debug/collection/{collection_name}")
async def debug_collection(collection_name: str, service: SentimentService = Depends()):
    """Debug endpoint to check collection info for sentiment collection."""
    try:
        info = service.get_collection_info(collection_name)
        return info
    except Exception as e:
        logger.error(f"Debug error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to retrieve collection information. Vector database service may be unavailable."
        )