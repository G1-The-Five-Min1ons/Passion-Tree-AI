from fastapi import FastAPI

app = FastAPI(
    title="AI Inference Service",
    description="Health check only for AI Microservice.",
)

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Start FastAPI AI",
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "AI Inference Engine",
    }