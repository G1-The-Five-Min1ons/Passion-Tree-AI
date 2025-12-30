from fastapi import FastAPI
from app.api.router import api_router

app = FastAPI(
    title="AI Inference Service",
    description="AI Microservice for Passion Tree - Topic Analysis, Sentiment Analysis, and Recommendations",
    version="1.0.0"
)

app.include_router(api_router, prefix="/api/v1")
