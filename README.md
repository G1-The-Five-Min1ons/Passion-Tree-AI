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

## Notes

- Ensure GROQ_API_KEY, REDIS_URL, and DB_URL are provided by environment/secrets in Terraform.