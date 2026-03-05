import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    APP_NAME: str = "embedding-service"
    APP_ENV: str = "development"
    
    MODEL_NAME: str = "BAAI/bge-m3"
    TEI_ENDPOINT: str = Field("http://localhost:8081", alias="TEI_ENDPOINT")
    BATCH_SIZE: int = 64
    
    REDIS_URL: str = Field(..., alias="REDIS_URL")
    RABBITMQ_URL: str = Field(..., alias="RABBITMQ_URL")
    
    RABBITMQ_QUEUE_CONSUME: str = Field("embedding_queue", alias="EMBEDDING_QUEUE")
    RABBITMQ_QUEUE_PUBLISH: str = Field("vector_queue", alias="VECTOR_QUEUE")

    model_config = SettingsConfigDict(
        env_file="../../.env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        populate_by_name=True
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()