import logging
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_fixed

from app.core.config import settings

logger = logging.getLogger(__name__)

class GroqClientStore:
    _client = None

    @classmethod
    def get_client(cls) -> AsyncAnthropic:
        if cls._client is None:
            if not settings.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY is missing")

            cls._client = AsyncAnthropic(api_key=settings.GROQ_API_KEY)

        return cls._client

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def call_groq_api(prompt: str, model: str = "claude-haiku-4-5") -> str:
    try:
        client = GroqClientStore.get_client()

        message = await client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            max_tokens=1024,
        )

        return message.content[0].text

    except Exception as e:
        logger.error(f"Anthropic API Error: {e}", exc_info=True)
        raise e