from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8005
    debug: bool = True

    # LLM – Fastwork API (chuẩn OpenAI-Compatible)
    llm_provider: str = "openai"
    model_name: str = "Qwen/Qwen3-8B"
    openai_api_base: str = "https://aiapi.fastwork.vn/llm/v1"
    openai_api_key: str = "fastwork-api-key"

    # RAG Service (PDF lookup)
    rag_service_url: str = "http://localhost:8006"

    # Predict Service (Đối tác – 10.0.0.62)
    predict_service_url: str = "http://10.0.0.62:8000"
    predict_api_key: str = "hanoi-water-2026-secret"

    # API Gateway (nếu cần gọi ngược)
    api_gateway_url: str = "http://localhost:8000"

    # Database
    db_url: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
