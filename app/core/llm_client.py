import logging
from tenacity import retry, stop_after_attempt, wait_fixed

from app.core.config import settings
from app.core.network.groq_client import GroqClient

logger = logging.getLogger(__name__)

class GroqClientStore:
    _client: GroqClient | None = None

    @classmethod
    def get_client(cls) -> GroqClient:
        if cls._client is None:
            if not settings.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY is missing")
            cls._client = GroqClient()
        return cls._client

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def call_groq_api(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    try:
        client = GroqClientStore.get_client()
        response = await client.get_chat_completion(prompt=prompt, model=model)
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Groq API Error: {e}", exc_info=True)
        raise
