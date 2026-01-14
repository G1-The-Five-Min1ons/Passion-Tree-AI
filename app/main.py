import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from app.api.router import api_router
from contextlib import asynccontextmanager
from app.core.reranker_store import RerankerModelStore

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(load_model_in_background())
    yield
    
async def load_model_in_background():
    try:
        await asyncio.to_thread(RerankerModelStore.load_model)
        print("Background Task: Reranker Model Ready!")
    except Exception as e:
        print(f"Background Task Reranker Model Failed: {e}")

app = FastAPI(
    title="AI Inference Service",
    description="AI Microservice for Passion Tree - Topic Analysis, Sentiment Analysis, and Recommendations",
    version="1.0.0",
    lifespan=lifespan
)

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Custom exception handler to return consistent response structure."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "data": None
        }
    )

app.include_router(api_router, prefix="/api/v1")
