import hashlib
import numpy as np
from app.config import settings
from app.utils.logger import get_logger
from app.utils.http_client import get_indexing_client, get_embedding_client
from app.utils.cache import MemoryCache

logger = get_logger("EMBEDDING")


async def embedding_func(texts: list[str]) -> np.ndarray:
    """Embedding function dùng cho indexing — batch call, không cache."""
    if not texts:
        return np.array([])
    try:
        client = get_indexing_client()
        url = f"{settings.EMBEDDING_SERVICE_URL.rstrip('/')}/api/v1/embed/batch"
        response = await client.post(url, json={"texts": texts, "model": settings.EMBEDDING_MODEL_NAME})
        response.raise_for_status()
        embeddings = response.json().get("vectors", [])
        return np.array(embeddings)
    except Exception as e:
        logger.error(f"Embedding Service Failed: {e}")
        raise e


async def query_embedding_func(texts: list[str]) -> np.ndarray:
    """
    Embedding function dùng cho query-time.
    Tích hợp request-scoped cache TTL 5s: cùng text trong 1 request cycle
    (asyncio.gather của ConsensusRetriever) → trả vector đã tính, tránh gọi 3 lần.
    """
    if not texts:
        return np.array([])

    results = []
    client = get_embedding_client()
    url = f"{settings.EMBEDDING_SERVICE_URL.rstrip('/')}/api/v1/embed/text"

    for text in texts:
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cached = MemoryCache.get_embed(cache_key)
        if cached is not None:
            logger.debug(f"Embedding Cache HIT: '{text[:30]}...'")
            results.append(cached)
            continue
        try:
            payload = {"text": text, "model": settings.EMBEDDING_MODEL_NAME}
            logger.info(f"Embedding Query: '{text[:30]}...' -> {url}")
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            vector = data.get("vector") or data.get("embedding")
            if vector:
                MemoryCache.set_embed(cache_key, vector)
                results.append(vector)
            else:
                logger.warning(f"Empty embedding for query: {text[:20]}...")
                results.append([0.0] * settings.EMBEDDING_DIM)
        except Exception as e:
            logger.error(f"Query Embedding Error: {e}")
            results.append([0.0] * settings.EMBEDDING_DIM)

    return np.array(results)