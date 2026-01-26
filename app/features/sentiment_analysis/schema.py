from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional


class Advanced(BaseModel):
    """Advanced analysis metrics"""
    primary_emotion: str = Field(..., description="Primary detected emotion")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score for the analysis")
    struggle_point: str = Field(..., description="Main struggle or challenge point")
    learning_disposition: str = Field(default="Growth Mindset", description="Learning disposition classification")
    consistency_check: Optional[str] = Field(default="Match", description="Consistency check result")


class DevelopmentPlan(BaseModel):
    """Development plan for the learner"""
    next_steps: List[str] = Field(..., description="Suggested next steps for improvement")


class LLMAnalysis(BaseModel):
    """Strict schema for validating LLM output"""
    analysis: str = Field(..., description="Detailed analysis of the learning reflection")
    recommendation: str = Field(..., description="Recommendations for improving learning")
    next_steps: str = Field(..., description="Suggested next steps for the learner")
    score: float = Field(..., ge=1, le=10, description="Reflection score from 1-10")
    sentiment: Literal["Positive", "Neutral", "Negative"] = Field(..., description="Sentiment classification")
    advanced: Optional[Advanced] = Field(default=None, description="Advanced analysis metrics")
    development_plan: Optional[DevelopmentPlan] = Field(default=None, description="Development plan")


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
    sentiment: str = Field(..., description="Detected sentiment from the reflection")
    reflection_score: float = Field(..., ge=0, le=10, description="Reflection score (1-10) predicted from the reflection")
    summary: str = Field(default="", description="Summary of the learning reflection")
    advanced: Optional[Advanced] = Field(default=None, description="Advanced analysis metrics")
    development_plan: Optional[DevelopmentPlan] = Field(default=None, description="Development plan for the learner")
    reranked_results: List[str] = Field(default=[], description="Similar reflections from database for few-shot learning")
