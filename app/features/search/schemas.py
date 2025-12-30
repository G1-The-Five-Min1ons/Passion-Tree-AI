from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Request Schema
class SearchRequest(BaseModel):
    query: str = Field(
        ..., 
        min_length=1,
        example="I want to learn data analysis with Python for my business.", 
        description="The question or learning intent"
    )
    top_k: int = Field(default=7, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = None

# Response Schema
class LearningPathPayload(BaseModel):
    title: str = Field(..., description="The title of the learning path")
    description: str = Field(..., description="The summary of the path content")

class SearchResult(BaseModel):
    id: int = Field(..., description="SQL Database ID")
    score: float = Field(..., description="Similarity score")
    payload: LearningPathPayload

class SearchResponse(BaseModel):
    query: str
    total: int
    results: List[SearchResult]

    class Config:
        json_schema_extra = {
            "example": {
                "query": "I want to start a career in AI",
                "total": 1,
                "results": [
                    {
                        "id": 501,
                        "score": 0.945,
                        "payload": {
                            "title": "AI Engineering Career Path",
                            "description": "Learn the essentials of Machine Learning and Deep Learning with Python."
                        }
                    }
                ]
            }
        }