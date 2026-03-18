import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils.logger import get_logger
from app.api.routes import ingest
from app.api.routes import chat
from app.api.routes import image


logger = get_logger("MAIN_APP")

def create_application() -> FastAPI:
    application = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url=f"{settings.API_V1_STR}/docs",
    )

    if settings.BACKEND_CORS_ORIGINS:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    application.include_router(
        ingest.router,
        prefix=f"{settings.API_V1_STR}/ingest",
        tags=["ingestion"]
    )

    application.include_router(
        chat.router,
        prefix="/api/v1",
        tags=["RAG Chat"]
    )

    application.include_router(
        image.router,
        prefix="/api/v1",
        tags=["Images"]
    )

    return application

app = create_application()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "rag-service", "version": "1.0.0"}

@app.on_event("startup")
async def startup_event():
    logger.info(">>> STARTING RAG SERVICE <<<")

    # 0. Khởi tạo persistent HTTP clients (tránh TCP overhead)
    from app.utils.http_client import get_embedding_client, get_indexing_client
    get_embedding_client()
    get_indexing_client()
    logger.info("[Startup] Persistent HTTP clients initialized")

    # 1. Pre-load Reranker model vào GPU
    if settings.RERANKER_ENABLED:
        try:
            from app.infrastructure.reranker.bge_reranker import _get_reranker
            logger.info("[Startup] Pre-loading Reranker model...")
            await _get_reranker()
            logger.info("[Startup] Reranker ready")
        except Exception as e:
            logger.warning(f"[Startup] Reranker pre-load failed: {e}")

    # 2. Pre-warm Embedding workers bằng 1 dummy request
    try:
        import numpy as np
        from app.infrastructure.embedding.embedding_func import embedding_func
        logger.info("[Startup] Pre-warming Embedding workers...")
        await embedding_func(["warmup"])
        logger.info("[Startup] Embedding workers ready")
    except Exception as e:
        logger.warning(f"[Startup] Embedding warm-up failed: {e}")

    # 3. Pre-initialize LLM clients (tránh lazy-init trên request đầu tiên)
    try:
        from app.infrastructure.llm.openai_client import _get_llm_client, _get_response_llm_client
        logger.info("[Startup] Pre-initializing LLM clients...")
        _get_llm_client()          # init singleton indexing LLM client
        _get_response_llm_client() # init singleton response LLM client
        logger.info("[Startup] LLM clients initialized")
    except Exception as e:
        logger.warning(f"[Startup] LLM client pre-init failed: {e}")

    # 4. Pre-warm Response LLM bằng 1 dummy call (khởi tạo HTTP connection pool)
    try:
        from app.infrastructure.llm.llm_func import response_llm_func
        logger.info("[Startup] Pre-warming Response LLM connection pool...")
        await response_llm_func("hi", system_prompt=None)
        logger.info("[Startup] Response LLM connection pool ready")
    except Exception as e:
        logger.warning(f"[Startup] Response LLM warm-up failed: {e}")

    # 4b. Pre-warm local LLM (keyword extraction) — force load model vào GPU
    # max_tokens=1: chỉ cần ping để init connection pool, không generate thực sự
    try:
        from app.infrastructure.llm.llm_func import query_llm_func
        logger.info("[Startup] Pre-warming local LLM (keyword extraction model)...")
        await query_llm_func("warmup", system_prompt=None, max_tokens=1)
        logger.info("[Startup] Local LLM model loaded into GPU memory")
    except Exception as e:
        logger.warning(f"[Startup] Local LLM warm-up failed (non-critical): {e}")

    # 5. Pre-initialize Query Engine cho các workspace đã khai báo (tùy chọn)
    workspaces = [w.strip() for w in settings.PRELOAD_WORKSPACES.split(",") if w.strip()]
    if workspaces:
        from app.infrastructure.graph.lightrag_factory import RAGFactory, QueryRAGFactory
        async def _init_ws(ws: str):
            try:
                logger.info(f"[Startup] Pre-initializing workspace: {ws}...")
                await RAGFactory.get_or_create_rag(ws)
                await QueryRAGFactory.get_or_create_rag(ws)
                logger.info(f"[Startup] Workspace '{ws}' ready (indexing + query)")
            except Exception as e:
                logger.warning(f"[Startup] Workspace '{ws}' init failed: {e}")
        await asyncio.gather(*[_init_ws(ws) for ws in workspaces])
    else:
        logger.info("[Startup] No PRELOAD_WORKSPACES configured — skipping pre-init")

    logger.info(">>> RAG SERVICE READY <<<")

@app.on_event("shutdown")
async def shutdown_event():
    from app.utils.http_client import close_all
    await close_all()
    logger.info(">>> SHUTTING DOWN RAG SERVICE <<<")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True
    )