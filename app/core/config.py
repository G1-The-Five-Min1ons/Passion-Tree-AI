from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    QDRANT_URL: str
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_TIMEOUT: int = 10
    GROQ_API_KEY: Optional[str] = None
    REDIS_URL: str = "redis://redis:6379"

    # การตั้งค่าที่ยืดหยุ่นที่สุดสำหรับทั้ง Local และ Production
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # 2. ถ้ามี Environment Variable ชื่อเดียวกันในระบบ (เช่น ใน Docker/Azure) 
        # ค่าในระบบจะ "เขียนทับ" ค่าในไฟล์ .env โดยอัตโนมัติ
        extra="ignore"
    )

settings = Settings()