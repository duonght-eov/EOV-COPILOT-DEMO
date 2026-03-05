"""
Singleton persistent httpx.AsyncClient cho toàn bộ service.
Tránh overhead TCP handshake + SSL mỗi lần gọi embedding/LLM.
"""
import httpx
from app.utils.logger import get_logger

logger = get_logger("HTTP_CLIENT")

# Persistent client với connection pooling
_embedding_client: httpx.AsyncClient | None = None
_indexing_client: httpx.AsyncClient | None = None


def get_embedding_client() -> httpx.AsyncClient:
    """Client dùng cho query embedding (timeout ngắn hơn)."""
    global _embedding_client
    if _embedding_client is None or _embedding_client.is_closed:
        _embedding_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        logger.info("[HTTP_CLIENT] Embedding client initialized (persistent)")
    return _embedding_client


def get_indexing_client() -> httpx.AsyncClient:
    """Client dùng cho indexing embedding (timeout dài hơn do batch lớn)."""
    global _indexing_client
    if _indexing_client is None or _indexing_client.is_closed:
        _indexing_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        logger.info("[HTTP_CLIENT] Indexing client initialized (persistent)")
    return _indexing_client


async def close_all():
    """Đóng tất cả client khi shutdown service."""
    global _embedding_client, _indexing_client
    if _embedding_client and not _embedding_client.is_closed:
        await _embedding_client.aclose()
        logger.info("[HTTP_CLIENT] Embedding client closed")
    if _indexing_client and not _indexing_client.is_closed:
        await _indexing_client.aclose()
        logger.info("[HTTP_CLIENT] Indexing client closed")
