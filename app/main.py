from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from app.api.router import api_router
from app.core.embedding import EmbeddingService
from app.core.reranker_store import get_reranker_service
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress verbose HTTP logs from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Inference Service",
    description="AI Microservice for Passion Tree - Topic Analysis, Sentiment Analysis, and Recommendations",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Starting AI Service - Preloading embedding model...")
        # Initialize EmbeddingService to trigger model download/cache
        _ = EmbeddingService()
        logger.info("Embedding model loaded successfully - Server ready!")
        # Preload Reranker API model
        _ = get_reranker_service()
        logger.info("Reranker API model loaded successfully - Server ready!")
    except Exception as e:
        logger.error(f"Failed to preload embedding model: {e}")
        raise

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "data": None
        }
    )

app.include_router(api_router, prefix="/api/v1")
