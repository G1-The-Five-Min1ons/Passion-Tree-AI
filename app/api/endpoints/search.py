from fastapi import APIRouter, HTTPException, status, Depends, Body
from app.features.search.schemas import (
    SearchRequest, SearchResponse, 
    SyncLearningPathRequest, SyncResponse,
    BulkSyncRequest, BulkSyncResponse
)
from app.features.search.service import SearchService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/init")
async def initialize_collections(service: SearchService = Depends()):
    """Initialize required Qdrant collections."""
    try:
        service.initialize_collections("learning_paths", vector_size=384)
        return {"success": True, "message": "Collections initialized successfully"}
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Initialization failed: {str(e)}"
        )

@router.post("/", response_model=SearchResponse)
async def search_learning_paths(
    request: SearchRequest,
    service: SearchService = Depends()
):

    try:
        logger.info(f"Search query: {request.query}")
        response = await service.search(
            query=request.query, 
            top_k=request.top_k,
            filters=request.filters,
            resource_type=request.resource_type or "learning_paths"
        )
        return response
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.post("/embed")
async def embed_text(text: str = Body(..., embed=True), service: SearchService = Depends()):
    """Generate embedding vector from input text."""
    vector = service.embedding.generate_vector(text)
    return {"embedding": vector}

@router.post("/sync", response_model=SyncResponse)
async def sync_learning_path(
    request: SyncLearningPathRequest,
    service: SearchService = Depends()
):
    """Sync single learning path (for CREATE or UPDATE operations)."""
    try:
        logger.info(f"Syncing learning path {request.path_id} to Qdrant")
        await service.sync_upsert(
            collection_name=request.collection_name,
            path_id=request.path_id,
            title=request.title,
            description=request.description,
            metadata=request.metadata
        )
        return SyncResponse(
            success=True,
            message=f"Learning path {request.path_id} synced successfully",
            path_id=request.path_id
        )
    except Exception as e:
        logger.error(f"Sync error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sync failed: {str(e)}"
        )

@router.post("/sync/bulk", response_model=BulkSyncResponse)
async def bulk_sync_learning_paths(
    request: BulkSyncRequest,
    service: SearchService = Depends()
):
    """Bulk sync multiple learning paths (for INITIAL SYNC from GO backend)."""
    total = len(request.learning_paths)
    succeeded = 0
    failed = 0
    errors = []
    
    logger.info(f"Bulk syncing {total} learning paths to Qdrant")
    
    for path_request in request.learning_paths:
        try:
            await service.sync_upsert(
                collection_name=request.collection_name,
                path_id=path_request.path_id,
                title=path_request.title,
                description=path_request.description,
                metadata=path_request.metadata
            )
            succeeded += 1
        except Exception as e:
            failed += 1
            error_msg = f"Failed to sync path_id {path_request.path_id}: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    return BulkSyncResponse(
        success=(failed == 0),
        message=f"Bulk sync completed: {succeeded}/{total} succeeded",
        total=total,
        succeeded=succeeded,
        failed=failed,
        errors=errors
    )

@router.delete("/sync/{path_id}", response_model=SyncResponse)
async def delete_learning_path(
    path_id: int,
    collection_name: str = "learning_paths",
    service: SearchService = Depends()
):
    """Delete learning path from Qdrant vector database."""
    try:
        logger.info(f"Deleting learning path {path_id} from Qdrant")
        await service.sync_delete(collection_name=collection_name, path_id=path_id)
        return SyncResponse(
            success=True,
            message=f"Learning path {path_id} deleted successfully",
            path_id=path_id
        )
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete failed: {str(e)}"
        )

@router.get("/debug/collection/{collection_name}")
async def debug_collection(collection_name: str, service: SearchService = Depends()):
    """Debug endpoint to check collection info."""
    try:
        info = service.get_collection_info(collection_name)
        return info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Debug failed: {str(e)}"
        )