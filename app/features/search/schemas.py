from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class SearchRequest(BaseModel):
    """Generic schema for search requests (any resource type)"""
    query: str = Field(
        ..., 
        min_length=1,
        example="Find beginner Go courses", 
        description="Text to search for (Semantic Search)"
    )
    top_k: int = Field(
        default=7, 
        ge=1, 
        le=20, 
        description="Number of desired results"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, 
        example={"category_id": 10},
        description="Flexible metadata filter conditions (e.g. category_id, is_active)"
    )
    resource_type: Optional[str] = Field(
        None,
        example="learning_path",
        description="Type of resource to search (e.g. learning_path, course, article)"
    )

class UpsertRequest(BaseModel):
    """Generic upsert schema for adding/updating vector data (can be extended per resource)"""
    id: Any = Field(..., description="Resource ID from SQL Database")
    payload: Dict[str, Any] = Field(..., description="Flexible resource data (e.g. title, description, category_id)")

class SyncLearningPathRequest(BaseModel):
    """Schema for syncing learning path from GO backend to Qdrant"""
    path_id: int = Field(..., description="Learning path ID from SQL Database")
    title: str = Field(..., description="Learning path title")
    description: str = Field(..., description="Learning path description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for filtering (e.g. category_id, difficulty, tags)")
    collection_name: str = Field(default="learning_paths", description="Target collection name in Qdrant")

class SyncResponse(BaseModel):
    """Response for sync operations"""
    success: bool
    message: str
    path_id: Optional[int] = None

class BulkSyncRequest(BaseModel):
    """Schema for bulk syncing multiple learning paths (initial sync)"""
    learning_paths: List[SyncLearningPathRequest] = Field(..., description="List of learning paths to sync")
    collection_name: str = Field(default="learning_paths", description="Target collection name in Qdrant")

class BulkSyncResponse(BaseModel):
    """Response for bulk sync operations"""
    success: bool
    message: str
    total: int
    succeeded: int
    failed: int
    errors: List[str] = Field(default_factory=list)


# --- Response Schemas ---

class GenericPayload(BaseModel):
    """Flexible payload for any resource type (fields can vary)"""
    data: Dict[str, Any] = Field(..., description="Resource data (e.g. title, description, etc.)")

class SearchResult(BaseModel):
    """Generic search result for any resource type"""
    id: Any = Field(..., description="ID of the matching resource from SQL Database")
    score: float = Field(..., description="Similarity score (0.0 - 1.0)")
    payload: Dict[str, Any] = Field(..., description="Flexible resource data (e.g. title, description, etc.)")

class SearchResponse(BaseModel):
    """Generic schema for sending search results (any resource type)"""
    query: str
    total: int
    results: List[SearchResult]

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Find beginner Go courses",
                "total": 1,
                "results": [
                    {
                        "id": 123,
                        "score": 0.895,
                        "payload": {
                            "title": "Go Fundamental",
                            "description": "Learn the basics of Go programming from scratch",
                            "category_id": 10
                        }
                    }
                ]
            }
        }