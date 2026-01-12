from pydantic import BaseModel, Field
from typing import List, Dict, Any


class SentimentRequest(BaseModel):
    """User's learning reflection input"""
    what_learned: str = Field(
        ...,
        example="เรียนเรื่องฟังก์ชันใน Python และลองเขียนด้วยตัวเอง",
        description="What the user learned",
    )
    feelings_after_learning: str = Field(
        ...,
        example="รู้สึกสนุกและมั่นใจมากขึ้น",
        description="How the user feels after learning",
    )

class SentimentResponse(BaseModel):
    """AI-generated analysis of the learning reflection"""
    summary: Dict[str, Any] = Field(
        ..., description="Detailed analysis with keys: analysis, recommendation, next_steps, score, sentiment"
    )
    sentiment: str = Field(..., description="Detected sentiment from the reflection")
    reflection_score: float = Field(..., ge=0, le=10, description="Reflection score (1-10) predicted from the reflection")
    reranked_results: List[str] = Field(..., description="Similar reflections from database for few-shot learning")
