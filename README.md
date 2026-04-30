---
title: Passion Tree AI
emoji: 🌳
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# Passion-Tree AI (FastAPI)

## Dev: Run with Docker hot-reload

- Prereqs: Docker Desktop
- This uses uvicorn --reload to auto-restart on code changes.

### Start

From the infrastructure folder:

```powershell
cd ..\Passion-Tree-Infrastructure
./scripts/dev-up.ps1 -Rebuild
```

- AI service listens on http://localhost:8000.
- Health check: GET /health.

### How it works

- docker-compose.override.yml overrides ai-fastapi to build
	from ../Passion-Tree-AI/Dockerfile.dev, mounts the source,
	and runs uvicorn with --reload.
- Any saved Python changes under app/ trigger reload.

### Stop

```powershell
docker compose -f docker-compose.yml -f docker-compose.override.yml down
```

## Production: Azure Container Apps

- Do not use --reload in production. Use the production Dockerfile without reload and with non-root user.

### Build locally

```powershell
cd ..\Passion-Tree-AI
docker build -t ai-fastapi:prod -f Dockerfile .
```

### Push to ACR

```powershell
$Registry = "<yourRegistry>" # e.g. myregistry.azurecr.io
docker tag ai-fastapi:prod $Registry/ai-fastapi:latest
docker push $Registry/ai-fastapi:latest
```

### Point Terraform to your image

- Update ai_image in terraform.tfvars to your ACR path.
- Apply Terraform as documented in infrastructure README.

## Production: Render (recommended)

The repo ships a `render.yaml` blueprint that provisions:

- a Docker web service (`passion-tree-ai`) running this Dockerfile
- a Render Key Value (Redis-compatible) store, wired via `REDIS_URL`

### One-time setup

1. Push this repo to GitHub (Render reads the blueprint from the repo root).
2. In Render → New → Blueprint → connect the repo. Render parses `render.yaml`.
3. Fill the secrets marked `sync: false` in the dashboard:
   - `QDRANT_URL`, `QDRANT_API_KEY`
   - `GROQ_API_KEY`
   - `JINA_API_KEY`
4. Click Apply. First build takes ~3–5 min.

### After deploy

- Service URL: `https://passion-tree-ai.onrender.com` (Render assigns it).
- Health check: `GET /api/v1/health`.
- Update the Go backend's `AI_SERVICE_URL` env to that URL
  (see `.github/workflows/deploy-go.yml` and your runtime config).

### Notes

- The Dockerfile binds to `$PORT` (Render injects it) and falls back to 8000 locally.
- `fastembed` was removed; embeddings now go through Jina API
  (set `JINA_API_KEY`). Image is much smaller and cold start is faster.
- Render Free plan sleeps after 15 min of inactivity. Use Starter ($7/mo)
  for always-on if cold starts hurt.

## Notes

- Ensure GROQ_API_KEY, JINA_API_KEY, QDRANT_*, and REDIS_URL are provided by environment/secrets.