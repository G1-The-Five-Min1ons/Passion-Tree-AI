# Production Dockerfile for FastAPI AI service (Render / ACA compatible)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app

# Use non-root user
RUN useradd -u 10001 -r -s /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app
USER 10001

EXPOSE 8000

# Bind to $PORT so Render / ACA can inject the port at runtime.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]
