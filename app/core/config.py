from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # default is development
    APP_ENV: str = "development" 
    
    QDRANT_URL: str
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_TIMEOUT: int = 10
    GROQ_API_KEY: Optional[str] = None
    JINA_API_KEY: Optional[str] = None
    REDIS_URL: str = "redis://redis:6379"
    MODEL_CACHE_DIR: str = "./models/cache"

    # Create a property to check if the environment is development
    @property
    def is_dev(self) -> bool:
        return self.APP_ENV.lower() != "production"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Create an instance for use
settings = Settings()