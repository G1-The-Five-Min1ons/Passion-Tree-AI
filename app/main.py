import os, time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from app.api.router import api_router
from app.core.embedding import EmbeddingService
from app.core.reranker_store import get_reranker_service
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
    client_ip = request.client.host if request.client else "unknown"

    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    status_code = response.status_code
    
    # ถ้าอยู่ใน Dev Mode อยากให้แสดงผลบรรทัดเดียวเหมือน Go
    if settings.is_dev:
        # ผลลัพธ์: INFO: | 200 | 34.35ms | 172.18.0.1 | POST | /api/v1/...
        ai_logger.info(
            f"| {status_code} | {process_time:.2f}ms | {client_ip} | {method} | {path}"
        )
    else:
        # ใน Prod จะพ่นเป็น JSON ตาม SlogStyleFormatter
        ai_logger.info(
            "request handled",
            extra={
                "method": method,
                "path": path,
                "status_code": status_code,
                "latency_ms": f"{process_time:.2f}ms",
                "ip": client_ip
            }
        )
    return response

@app.on_event("startup")
async def startup_event():
    start_time = time.perf_counter()
    try:
        _ = EmbeddingService()
        _ = get_reranker_service()
        
        duration = time.perf_counter() - start_time
        
        ai_logger.info(
            "AI Service ready - All models preloaded", 
            extra={
                "duration_sec": f"{duration:.2f}s",
                "models": ["Embedding", "Reranker"],
                "status": "ready"
            }
        )
        
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
            "error_detail": exc.detail,
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