from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# create Singleton Instance
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL, 
    api_key=settings.QDRANT_API_KEY,
    timeout=settings.QDRANT_TIMEOUT
)

def get_qdrant_client() -> QdrantClient:
    return qdrant_client

async def verify_qdrant_connection():
    try:
        qdrant_client.get_collections()
        logger.info("Successfully connected to Qdrant Cloud/DB.")
    except UnexpectedResponse as e:
        logger.error(f"Qdrant connection failed (Unauthorized/Invalid URL): {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while connecting to Qdrant: {e}")