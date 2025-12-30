from fastapi import APIRouter, status
from app.core.config import settings
from app.core.redis import redis_client
from app.core.vector_database import qdrant_client
from app.api.endpoints import recommend, reflection, search
import logging

logger = logging.getLogger(__name__)

api_router = APIRouter()

# Default root endpoint
@api_router.get("/")
def root():
    return {
        "status": "ok",
        "service": "Passion Tree AI Service",
        "version": "1.0.0",
        "docs": "/docs"
    }

# Health check endpoint
@api_router.get("/health")
def health_check():
    health_status = {
        "status": "healthy",
        "service": "AI Inference Engine",
        "version": "1.0.0",
        "dependencies": {}
    }
    all_healthy = True
    # Check Redis connection
    try:
        redis_client.ping()
        health_status["dependencies"]["redis"] = "connected"
    except Exception as e:
        health_status["dependencies"]["redis"] = f"disconnected: {str(e)}"
        all_healthy = False
        logger.error(f"Redis health check failed: {e}")
    # Check Qdrant connection
    try:
        qdrant_client.get_collections()
        health_status["dependencies"]["qdrant"] = "connected"
    except Exception as e:
        health_status["dependencies"]["qdrant"] = f"disconnected: {str(e)}"
        all_healthy = False
        logger.error(f"Qdrant health check failed: {e}")
    # Check Groq API Key
    if settings.GROQ_API_KEY:
        health_status["dependencies"]["groq_api"] = "configured"
    else:
        health_status["dependencies"]["groq_api"] = "not_configured"
        all_healthy = False
    if not all_healthy:
        health_status["status"] = "degraded"
        return health_status, status.HTTP_503_SERVICE_UNAVAILABLE
    return health_status

# Include routers from different endpoints
api_router.include_router(
    recommend.router,
    prefix="/recommend",
    tags=["Recommendation"]
)

api_router.include_router(
    reflection.router,
    prefix="/reflection",
    tags=["Reflection & Sentiment Analysis"]
)

api_router.include_router(
    search.router,
    prefix="/search",
    tags=["Search"]
)
