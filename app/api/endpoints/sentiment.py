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
        logger.info("Analyzing sentiment for input text")
        sentiment, score = service.analyze(request.text)
        return SentimentResponse(sentiment=sentiment, score=score)
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}"
        )
