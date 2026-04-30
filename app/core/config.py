from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # default is development
    APP_ENV: str = "development" 
    
    QDRANT_URL: str
    QDRANT_API_KEY: Optional[str] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9sk0CaAuLIKcDVHep2pvyiCH0NP1mugNqng6HLSD_QY"
    QDRANT_TIMEOUT: int = 10
    GROQ_API_KEY: Optional[str] = "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"
    JINA_API_KEY: Optional[str] = "jina_978f1b54afbd4b69ae3fbda212034935ZHZgBFogV6BGnoRaxrqzAUxeG2qY"
    HF_TOKEN: Optional[str] = "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"
    REDIS_URL: str = "redis://redis:6379"
    MODEL_CACHE_DIR: str = "./models/cache"

    # Recommendation tuning
    REC_MIN_BEHAVIOR_WEIGHT: float = 0.2
    REC_MAX_BEHAVIOR_WEIGHT: float = 0.8
    REC_FULL_BEHAVIOR_DENSITY: float = 10.0

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