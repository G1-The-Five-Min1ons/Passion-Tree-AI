from redis import Redis
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Create Singleton Redis Client Instance
redis_client = Redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5
)

def get_redis_client() -> Redis:
    """Get Redis client instance"""
    return redis_client

async def verify_redis_connection():
    """Verify Redis connection on startup"""
    try:
        redis_client.ping()
        logger.info("✅ Successfully connected to Redis.")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        raise
