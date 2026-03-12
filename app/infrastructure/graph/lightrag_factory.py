"""
infrastructure/graph/lightrag_factory.py
Trách nhiệm:
  - _get_or_create_rag: Factory pattern tạo/cache LightRAG instance per workspace
  - Cấu hình storage backends (PG, Neo4j, VectorDB)
  - Initialize RAGAnything instances
"""
import os
import asyncio
import ujson as json
from typing import Dict, Tuple
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig
from app.config import settings
from app.utils.logger import get_logger
from app.infrastructure.llm.llm_func import llm_completion_func
from app.infrastructure.vlm.vlm_client import vlm_model_func
from app.infrastructure.embedding.embedding_func import embedding_func
from app.services.processing.prompt_loader import get_prompt_config
from app.services.indexing.lightrag_adapter import lightrag_chunking_adapter

logger = get_logger("LIGHTRAG FACTORY")
# Global caches
_rag_instances: Dict[str, LightRAG] = {}
_rag_locks: Dict[str, asyncio.Lock] = {}
_rag_anything_instances: Dict[str, RAGAnything] = {}

class RAGFactory:
    """Factory để khởi tạo và quản lý các instance của LightRAG / RAGAnything."""
    @classmethod
    async def get_or_create_rag(cls, workspace: str) -> Tuple[LightRAG, RAGAnything]:
        """
        Khởi tạo hoặc trả về instance cached theo workspace.
        Trả về tuple: (LightRAG instance, RAGAnything instance)
        """
        global _rag_instances, _rag_locks, _rag_anything_instances
        if workspace not in _rag_locks:
            _rag_locks[workspace] = asyncio.Lock()
        async with _rag_locks[workspace]:
            if workspace in _rag_instances and workspace in _rag_anything_instances:
                return _rag_instances[workspace], _rag_anything_instances[workspace]
            logger.info(f"Initializing LightRAG & RAGAnything cho workspace: '{workspace}'")
            # 1. Setup Data Directory
            rag_work_dir = os.path.join(settings.RAG_WORK_DIR, "lightrag_index", workspace)
            os.makedirs(rag_work_dir, exist_ok=True)
            # 2. Setup Storage Config
            storage_kwargs = {}
            if settings.STORAGE_TYPE == "postgres":
                os.environ["KV_STORAGE_CONFIG"] = json.dumps({
                    "host": settings.POSTGRES_HOST,
                    "port": settings.POSTGRES_PORT,
                    "user": settings.POSTGRES_USER,
                    "password": settings.POSTGRES_PASSWORD,
                    "database": settings.POSTGRES_DATABASE,
                })
                storage_kwargs["kv_storage"] = "PGKVStorage"
                storage_kwargs["vector_storage"] = "PGVectorStorage"
                logger.debug(f"Sử dụng PostgreSQL storage cho workspace '{workspace}'")
            if settings.ENABLE_GRAPH_STORAGE and settings.GRAPH_STORAGE_TYPE == "neo4j":
                os.environ["GRAPH_STORAGE_CONFIG"] = json.dumps({
                    "uri": settings.NEO4J_URI,
                    "username": settings.NEO4J_USERNAME,
                    "password": settings.NEO4J_PASSWORD
                })
                storage_kwargs["graph_storage"] = "Neo4JStorage"
                logger.debug(f"Sử dụng Neo4j storage cho workspace '{workspace}'")
            # CRITICAL: LightRAG đọc LLM_TIMEOUT qua os.getenv() trực tiếp
            # Worker timeout = LLM_TIMEOUT × 2, phải set env trước khi init
            os.environ["LLM_TIMEOUT"] = str(settings.LLM_TIMEOUT)

            # 3. Init LightRAG Core
            rag_instance = LightRAG(
                working_dir=rag_work_dir,
                workspace=workspace,
                llm_model_max_async=settings.RAG_MAX_ASYNC_JOBS,
                embedding_func_max_async=settings.RAG_MAX_ASYNC_JOBS,
                llm_model_func=llm_completion_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=settings.EMBEDDING_DIM,
                    max_token_size=settings.EMBEDDING_MAX_TOKEN_SIZE,
                    func=embedding_func
                ),
                # Truyền timeout trực tiếp để không phụ thuộc vào os.getenv
                default_llm_timeout=settings.LLM_TIMEOUT,
                # Tắt gleaning: chỉ dùng 1 LLM call/chunk thay vì 2-3 calls
                entity_extract_max_gleaning=0,
                **storage_kwargs
            )
            # 4. Apply Custom Prompts (từ prompt_loader)
            pc = get_prompt_config()
            if pc.get("entity_extract"): rag_instance.entity_extract_template = pc["entity_extract"]
            if pc.get("entity_summary"): rag_instance.entity_summary_template = pc["entity_summary"]
            if pc.get("rag_response"): rag_instance.rag_response_template = pc["rag_response"]
            if pc.get("naive_rag_response"): rag_instance.naive_rag_response_template = pc["naive_rag_response"]
            if pc.get("keywords"): rag_instance.keywords_extract_template = pc["keywords"]
            # 5. Apply Custom Chunker Adapter
            rag_instance.chunking_func = lightrag_chunking_adapter
            # Initialize storages (Tạo bảng DB, etc.)
            await rag_instance.initialize_storages()
            _rag_instances[workspace] = rag_instance
            # 6. Init RAGAnything (Multimodal extension)
            rag_anything = RAGAnything(
                lightrag=rag_instance,
                vision_model_func=vlm_model_func,
                config=RAGAnythingConfig(
                    working_dir=settings.RAG_WORK_DIR,
                    enable_image_processing=True,
                    enable_table_processing=True
                ),
                llm_model_func=llm_completion_func,
                embedding_func=embedding_func,
            )
            _rag_anything_instances[workspace] = rag_anything
            logger.info(f"LightRAG & RAGAnything đã sẵn sàng cho workspace '{workspace}'.")
            return rag_instance, rag_anything