from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http import models
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

def create_collection_if_not_exists(collection_name: str, vector_size: int = 384):
    """Create a Qdrant collection if it doesn't exist."""
    try:
        collections = qdrant_client.get_collections().collections
        existing_names = [col.name for col in collections]
        
        if collection_name not in existing_names:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created collection '{collection_name}' with vector size {vector_size}")
        else:
            logger.info(f"Collection '{collection_name}' already exists")
    except Exception as e:
        logger.error(f"Failed to create collection '{collection_name}': {e}")
        raise

async def verify_qdrant_connection():
    try:
        qdrant_client.get_collections()
        logger.info("Successfully connected to Qdrant Cloud/DB.")
    except UnexpectedResponse as e:
        logger.error(f"Qdrant connection failed (Unauthorized/Invalid URL): {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while connecting to Qdrant: {e}")