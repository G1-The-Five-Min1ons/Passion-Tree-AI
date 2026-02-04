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
    ai_confident_score: float = Field(..., ge=0, le=1, description="AI's confidence in the analysis (0-1)")
    advanced: Optional[Advanced] = Field(default=None, description="Advanced analysis metrics")
    development_plan: Optional[DevelopmentPlan] = Field(default=None, description="Development plan")


class SentimentRequest(BaseModel):
    """User's learning reflection input"""
    learning_reflect: str = Field(
        ...,
        example="เรียนเรื่องฟังก์ชันใน Python และลองเขียนด้วยตัวเอง",
        description="What the user learned",
    )
    mood_reflect: str = Field(
        ...,
        example="รู้สึกสนุกและมั่นใจมากขึ้น",
        description="How the user feels after learning",
    )
    feel_score: float = Field(
        ...,
        ge=0,
        le=5,
        example=4.0,
        description="Self-assessed feeling score (0-5)",
    )
    progress_score: float = Field(
        ...,
        ge=0,
        le=5,
        example=3.5,
        description="Self-assessed progress score (0-5)",
    )
    challenge_score: float = Field(
        ...,
        ge=0,
        le=5,
        example=3.0,
        description="Self-assessed challenge level score (0-5)",
    )

class SentimentResponse(BaseModel):
    """AI-generated analysis of the learning reflection"""
    summary: str = Field(..., description="Detailed analysis summary of the student's reflection")
    sentiment_analysis: str = Field(..., description="Sentiment classification: Positive, Neutral, or Negative")
    primary_emotion: str = Field(..., description="Primary detected emotion (e.g., Confident, Frustrated, Curious)")
    struggle_point: str = Field(..., description="Main struggle or challenge point identified")
    development_plan: List[str] = Field(..., description="Detailed development plan steps")
    ai_confident_score: float = Field(..., ge=0, le=1, description="AI confidence score for the analysis (0-1)")
    reflection_score: float = Field(..., ge=0, le=10, description="Text-based reflection quality score (0-10)")
    weighted_reflection_score: float = Field(..., ge=0, le=10, description="Weighted score combining text analysis and self-assessment scores")
    reranked_results: List[str] = Field(default=[], description="Similar reflections from database for few-shot learning")
