import os
import logging
from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def call_groq_api(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    try:
        client = AsyncGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            temperature=0.3,
            max_tokens=1024,
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        logger.error(f"Groq API Error: {e}", exc_info=True)
        raise e 