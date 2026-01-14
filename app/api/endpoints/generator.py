from fastapi import APIRouter, status, Depends, HTTPException
from app.features.generator.schema import GenerateRequest, GenerateResponse
from app.features.generator.service import GeneratorService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/learning-path", response_model=GenerateResponse)
async def generate_learning_path(
    request: GenerateRequest,
    service: GeneratorService = Depends()
):
    try:
        logger.info(f"Generating path for topic: {request.topic}")
        response = await service.generate_learning_path(request)
        return response
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Learning path generation service is temporarily unavailable. Please try again later."
        )