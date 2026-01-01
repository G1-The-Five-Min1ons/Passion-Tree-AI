from fastapi import APIRouter, HTTPException, status, Depends, Body
from app.features.search.schemas import SearchRequest, SearchResponse
from app.features.search.service import SearchService
from app.core.embedding import EmbeddingService
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

@router.post("/embed")
async def embed_text(text: str = Body(..., embed=True)):
    """Generate embedding vector from input text."""
    embedder = EmbeddingService()
    vector = embedder.generate_vector(text)
    return {"embedding": vector}