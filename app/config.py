"""
API Gateway Configuration
"""
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = True
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./data/gateway.db"
    
    # JWT Authentication
    secret_key: str = "speedmaint-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    
    # RAGAnything Service
    rag_service_url: str = "http://localhost:8000"

    # Analytics/Predict Service
    analytics_service_url: str = "http://localhost:8007"
    
    # Water Forecast API
    water_api_url: str = "http://10.0.0.62:8000"
    
    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Storage
    upload_dir: str = "./data/uploads"
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
