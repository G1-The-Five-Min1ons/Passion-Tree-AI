from fastapi import APIRouter, status, Depends
from fastapi.responses import JSONResponse
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
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": False,
                "message": f"Generation failed",
                "data": None
            }
        )