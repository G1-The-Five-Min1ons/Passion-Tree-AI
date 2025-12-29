from fastapi import APIRouter, HTTPException, status, Depends
from app.features.search.schemas import SearchRequest, SearchResponse
from app.features.search.service import SearchService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/", response_model=SearchResponse)
async def search_learning_paths(
    request: SearchRequest,
    service: SearchService = Depends()
):

    try:
        logger.info(f"Search query: {request.query}")
        response = await service.search(request.query, request.top_k)
        return response
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )
