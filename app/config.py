
import os
from functools import lru_cache
from typing import List, Union, Optional
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.utils.logger import RAGLogger  

class Settings(BaseSettings):
    """
    Application Settings managed by Pydantic.
    Reads values from environment variables or .env file.
    """
    
    # --- Server Configuration ---
    PROJECT_NAME: str = "EOVCopilot RAG Service"
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: str = "INFO"
    BACKEND_CORS_ORIGINS: List[Union[str, AnyHttpUrl]] = []
    PORT: int = 8006

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            from json import loads
            if isinstance(v, str):
                return loads(v)
            return v
        raise ValueError(v)

    # --- MinIO Configuration ---
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_SECURE: bool = False
    MINIO_BUCKET_OCR_RESULTS: str = "ocr-results"
    MINIO_BUCKET_DOCUMENTS: str = "documents"

    # --- PostgreSQL Configuration (Vector & KG) ---
    POSTGRES_HOST: str
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DATABASE: str
    POSTGRES_SCHEMA: str = "public"

    # --- Neo4j Configuration (Graph DB) ---
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str

    # --- LLM Configuration (Text Generation — Keyword Extraction & Indexing) ---
    LLM_BASE_URL: str
    LLM_API_KEY: str
    LLM_MODEL_NAME: str
    LLM_TIMEOUT: int = 60
    LLM_MAX_TOKENS: int = 4096

    # --- Response LLM (sinh câu trả lời — model nhẹ hơn, máy riêng) ---
    RESPONSE_LLM_BASE_URL: str = ""
    RESPONSE_LLM_API_KEY: str = "ollama"
    RESPONSE_LLM_MODEL_NAME: str = ""
    RESPONSE_LLM_TIMEOUT: int = 120


    # --- VLM Configuration (Vision Language Model) ---
    VLM_BASE_URL: str
    VLM_API_KEY: str
    VLM_MODEL_NAME: str

    # --- Reranker Configuration ---
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_DEVICE: str = "cuda"
    RERANKER_ENABLED: bool = True
    RERANKER_TOP_N: int = 3

    # --- Embedding Service Configuration ---
    EMBEDDING_SERVICE_URL: str
    EMBEDDING_MODEL_NAME: str
    EMBEDDING_DIM: int = 1024
    EMBEDDING_MAX_TOKEN_SIZE: int = 8192

    # --- OCR Service Configuration ---
    OCR_SERVICE_URL: str = "http://10.0.0.156:8001"
    OCR_POLL_TIMEOUT: int = 1200

    # --- RAG Core Configuration ---
    RAG_WORK_DIR: str = "./rag_workspace"
    PROMPTS_DIR: str = "./prompts"
    RAG_CHUNK_SIZE: int = 1200
    RAG_CHUNK_OVERLAP: int = 100
    RAG_ENABLE_LLM_CACHE: bool = True
    RAG_MAX_ASYNC_JOBS: int = 32

    # --- OCR Service Configuration ---
    OCR_SERVICE_URL: str = "http://10.0.0.156:8001"
    OCR_POLL_TIMEOUT: int = 1200
    
    # --- Storage Configuration ---
    STORAGE_TYPE: str = "json"  # "json" | "postgres"
    ENABLE_GRAPH_STORAGE: bool = False
    GRAPH_STORAGE_TYPE: str = "json"  # "json" | "neo4j"

    # --- Consensus Retriever Configuration ---
    CONSENSUS_ENABLE_RELATION_SEARCH: bool = True
    CONSENSUS_RELATION_TOP_K: int = 5
    CONSENSUS_WEIGHT_NAIVE: float = 0.50
    CONSENSUS_WEIGHT_LOCAL: float = 0.35
    CONSENSUS_WEIGHT_RELATION: float = 0.15

    # --- Startup Pre-warming ---
    # Danh sách workspace slug cần khởi tạo ngay khi service start (cách nhau bởi dấu phẩy)
    PRELOAD_WORKSPACES: str = ""  # ví dụ: "qtxd,workspace2"


    # Pydantic Config
    model_config = SettingsConfigDict(
        env_file=[".env", "/app/.env"],  # Check local .env first, then docker path
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True
    )

@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings object.
    Also initializes the logger based on the configured LOG_LEVEL.
    """
    settings_obj = Settings()
    
    # Initialize Global Logger configuration once settings are loaded
    RAGLogger.setup_logging(log_level=settings_obj.LOG_LEVEL)
    
    return settings_obj

# Global instance
settings = get_settings()