"""
infrastructure/embedding/embedding_func.py
Trách nhiệm:
  - embedding_func: Gọi embedding service qua HTTP
  - Kết nối tới embedding service (bge-m3) qua persistent HTTP client
"""
import numpy as np
from app.config import settings
from app.utils.logger import get_logger
from app.utils.http_client import get_indexing_client

logger = get_logger("EMBEDDING")

async def embedding_func(texts: list[str]) -> np.ndarray:
  if not texts:
    return np.array([])
  
  try:
    client = get_indexing_client()
    payload = {
      "texts": texts,
      "model": settings.EMBEDDING_MODEL_NAME,
    }
    url = f"{settings.EMBEDDING_SERVICE_URL.rstrip('/')}/api/v1/embed/batch"

    response = await client.post(url, json=payload)
    response.raise_for_status()

    embeddings = response.json().get("vectors", [])
    return np.array(embeddings)

  except Exception as e:
    logger.error(f"Embedding Service Failed: {e}")
    raise e