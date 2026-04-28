# Production Dockerfile for FastAPI AI service (ACA-ready)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app

# Create cache directory and pre-download ONNX model at build time
RUN mkdir -p /app/models/cache
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2', cache_dir='./models/cache')"

# Use non-root user
RUN useradd -u 10001 -r -s /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app/models
USER 10001

EXPOSE 8000

# Run without reload; scale workers via args or ACA replicas
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
