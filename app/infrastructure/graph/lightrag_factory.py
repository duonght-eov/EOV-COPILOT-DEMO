"""
infrastructure/graph/lightrag_factory.py
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

_rag_instances: Dict[str, LightRAG] = {}
_rag_locks: Dict[str, asyncio.Lock] = {}
_rag_anything_instances: Dict[str, RAGAnything] = {}


class RAGFactory:
    @classmethod
    async def get_or_create_rag(cls, workspace: str) -> Tuple[LightRAG, RAGAnything]:
        global _rag_instances, _rag_locks, _rag_anything_instances
        if workspace not in _rag_locks:
            _rag_locks[workspace] = asyncio.Lock()
        async with _rag_locks[workspace]:
            if workspace in _rag_instances and workspace in _rag_anything_instances:
                return _rag_instances[workspace], _rag_anything_instances[workspace]

            logger.info(f"Initializing LightRAG & RAGAnything cho workspace: '{workspace}'")

            rag_work_dir = os.path.join(settings.RAG_WORK_DIR, "lightrag_index", workspace)
            os.makedirs(rag_work_dir, exist_ok=True)

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

            os.environ["LLM_TIMEOUT"] = str(settings.LLM_TIMEOUT)

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
                default_llm_timeout=settings.LLM_TIMEOUT,
                **storage_kwargs
            )

            # Apply Custom Prompts
            pc = get_prompt_config()
            if pc.get("entity_extract"): rag_instance.entity_extract_template = pc["entity_extract"]
            if pc.get("entity_summary"): rag_instance.entity_summary_template = pc["entity_summary"]
            if pc.get("rag_response"): rag_instance.rag_response_template = pc["rag_response"]
            if pc.get("naive_rag_response"): rag_instance.naive_rag_response_template = pc["naive_rag_response"]

            # Inject keywords_extraction vào global PROMPTS (giống phiên bản gốc)
            if pc.get("keywords") and len(pc["keywords"]) > 50:
                try:
                    from lightrag.prompt import PROMPTS
                    PROMPTS["keywords_extraction"] = pc["keywords"]
                    logger.info("RAGFactory: Injected Custom Vietnamese Keywords Extraction Prompt")
                except Exception as e:
                    logger.warning(f"RAGFactory: Failed to inject keywords prompt: {e}")

            rag_instance.chunking_func = lightrag_chunking_adapter
            await rag_instance.initialize_storages()
            _rag_instances[workspace] = rag_instance

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


class QueryRAGFactory:
    """
    Factory riêng cho Query Engine — dùng query_embedding_func có request-scoped cache.
    Giống phiên bản gốc QueryEngine._get_or_create_rag.
    """
    _rag_instances: Dict[str, LightRAG] = {}
    _rag_locks: Dict[str, asyncio.Lock] = {}

    @classmethod
    async def get_or_create_rag(cls, workspace: str) -> LightRAG:
        if workspace not in cls._rag_locks:
            cls._rag_locks[workspace] = asyncio.Lock()
        async with cls._rag_locks[workspace]:
            if workspace in cls._rag_instances:
                return cls._rag_instances[workspace]

            logger.info(f"Initializing Query Engine (Read-Only) for workspace: {workspace}...")

            from app.infrastructure.embedding.embedding_func import query_embedding_func

            rag_work_dir = os.path.join(settings.RAG_WORK_DIR, "lightrag_index", workspace)

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
                logger.info("QueryRAGFactory: Connected to PostgreSQL")

            if settings.ENABLE_GRAPH_STORAGE and settings.GRAPH_STORAGE_TYPE == "neo4j":
                os.environ["GRAPH_STORAGE_CONFIG"] = json.dumps({
                    "uri": settings.NEO4J_URI,
                    "username": settings.NEO4J_USERNAME,
                    "password": settings.NEO4J_PASSWORD
                })
                storage_kwargs["graph_storage"] = "Neo4JStorage"
                logger.info("QueryRAGFactory: Connected to Neo4j Graph")

            from app.infrastructure.llm.llm_func import query_llm_func as _query_llm_for_rag
            rag = LightRAG(
                working_dir=rag_work_dir,
                workspace=workspace,
                llm_model_max_async=settings.RAG_MAX_ASYNC_JOBS,
                embedding_func_max_async=settings.RAG_MAX_ASYNC_JOBS,
                llm_model_func=_query_llm_for_rag,
                embedding_func=EmbeddingFunc(
                    embedding_dim=settings.EMBEDDING_DIM,
                    max_token_size=settings.EMBEDDING_MAX_TOKEN_SIZE,
                    func=query_embedding_func
                ),
                **storage_kwargs
            )

            pc = get_prompt_config()
            if pc.get("rag_response"): rag.rag_response_template = pc["rag_response"]
            if pc.get("naive_rag_response"): rag.naive_rag_response_template = pc["naive_rag_response"]

            if pc.get("keywords") and len(pc["keywords"]) > 50:
                try:
                    from lightrag.prompt import PROMPTS
                    PROMPTS["keywords_extraction"] = pc["keywords"]
                    logger.info("QueryRAGFactory: Injected Custom Vietnamese Keywords Extraction Prompt")
                except Exception:
                    pass

            await rag.initialize_storages()
            cls._rag_instances[workspace] = rag
            logger.info(f"QueryRAGFactory: Query Engine Fully Initialized for {workspace}")
            return rag