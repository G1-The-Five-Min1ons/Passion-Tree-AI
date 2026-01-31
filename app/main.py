import os, time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from app.api.router import api_router
from app.core.embedding import EmbeddingService
from app.core.reranker_model import get_reranker_service
from app.core.logger import setup_logger
from app.core.config import settings

ai_logger = setup_logger(settings.is_dev)
ai_logger.info("Service started", extra={"env": settings.APP_ENV})

app = FastAPI(
    title="AI Inference Service",
    description="AI Microservice for Passion Tree - Topic Analysis, Sentiment Analysis, and Recommendations",
    version="1.0.0"
)

# Add Middleware to Log Every Request Like Fiber
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Extract basic info
    path = request.url.path
    method = request.method

    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    
    # Log output as JSON (Prod) or Text (Dev) automatically
    ai_logger.info(
        "request handled",
        extra={
            "method": method,
            "path": path,
            "status_code": response.status_code,
            "latency_ms": f"{process_time:.2f}ms",
            "ip": request.client.host
        }
    )
    return response

@app.on_event("startup")
async def startup_event():
    try:
        ai_logger.info("Starting AI Service - Preloading models...")
        
        # Initialize EmbeddingService
        _ = EmbeddingService()
        ai_logger.info("Embedding model loaded successfully")
        
        # Preload Reranker API model
        _ = get_reranker_service()
        ai_logger.info("Reranker API model loaded successfully", extra={"status": "ready"})
        
    except Exception as e:
        ai_logger.error("Failed to preload models", extra={"error": str(e)})
        os._exit(1)

# Adjust Exception Handler to Log Structured Data
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    # Log error details to Azure Monitor before sending response
    ai_logger.warning(
        "application handled error",
        extra={
            "status_code": exc.status_code,
            "message": exc.detail,
            "path": request.url.path
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "data": None
        }
    )

app.include_router(api_router, prefix="/api/v1")