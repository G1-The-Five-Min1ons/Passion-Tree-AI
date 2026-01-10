from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ReflectionRequest(BaseModel):
    """User's learning reflection input"""
    what_learned: str = Field(..., example="Reviewed graph traversal algorithms and implemented DFS in Go", description="What the user learned today")
    mood: int = Field(..., ge=1, le=5, description="User's mood after learning (1-5)")
    feelings_after_learning: str = Field(..., example="Feeling confident but still mix up recursion edge cases", description="How the user feels after learning")
    progress: int = Field(..., ge=1, le=5, description="Progress on learning (1-5)")
    challenge_level: int = Field(..., ge=1, le=5, description="How challenging is the material (1-5)")

class ReflectionResponse(BaseModel):
    """AI-generated analysis of the learning reflection"""
    summary: Dict[str, Any] = Field(..., description="Detailed analysis with keys: analysis, recommendation, next_steps")
    sentiment: str = Field(..., description="Detected sentiment from the reflection")
    overall_score: float = Field(..., ge=0, le=100, description="Overall reflection score (0-100)")
    reranked_results: List[str] = Field(..., description="Similar reflections from database for few-shot learning")
