from pydantic import BaseModel
from typing import List


class UserInteraction(BaseModel):
    user_id: str
    path_id: str
    score: float


class UserProfile(BaseModel):
    user_id: str
    interests: str


class BatchRecommendPayload(BaseModel):
    users_interactions: List[UserInteraction]
    users_profiles: List[UserProfile]


class RecommendedPathScore(BaseModel):
    path_id: str
    score: float


class BatchRecommendationResult(BaseModel):
    user_id: str
    recommended_paths: List[RecommendedPathScore]


class BatchRecommendResponse(BaseModel):
    success: bool
    message: str
    data: List[BatchRecommendationResult]
