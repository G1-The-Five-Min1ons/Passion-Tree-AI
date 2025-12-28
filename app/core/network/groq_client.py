from app.core.network.base_client import BaseClient
from app.core.config import settings

class GroqClient(BaseClient):
    def __init__(self):
        # ดึงค่าจาก Config ที่เตรียมไว้ใน core/config.py
        super().__init__(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.GROQ_API_KEY
        )

    async def get_chat_completion(self, prompt: str, model: str = "llama3-8b-8192"):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        # เรียกใช้ _send_request จาก Class แม่ได้เลย
        return await self._send_request(
            method="POST",
            endpoint="/chat/completions",
            data=payload
        )