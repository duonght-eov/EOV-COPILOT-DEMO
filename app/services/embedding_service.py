import json
import hashlib
import logging
from typing import List
import redis
from app.config import get_settings
from app.core.embedding_model import model_instance

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.settings = get_settings()
        self.model = model_instance
        self.redis_client = None 
        
    def _get_connection(self):
        """Hàm đảm bảo luôn lấy được kết nối Redis sống"""
        if self.redis_client:
            return self.redis_client
        
        try:
            client = redis.from_url(
                self.settings.REDIS_URL,
                decode_responses=True,
                socket_timeout=2
            )
            client.ping()
            logger.info("Connect to Redis Successfully!")
            self.redis_client = client
            return client
        except Exception as e:
            logger.warning(f"Failed to Connect Redis: {e}")
            return None
        
    def _generate_cache_key(self, text: str) -> str:
        clean_text = text.strip()
        text_hash = hashlib.md5(clean_text.encode("utf-8")).hexdigest()
        return f"embed:{self.settings.MODEL_NAME}:{text_hash}"
    
    async def embed_text(self, text: str) -> List[float]:
        client = self._get_connection()
        if client:
            cache_key = self._generate_cache_key(text)
            try:
                cached = client.get(cache_key)
                if cached:
                    return json.loads(cached)   # fix: json.loads thay vì json.load
            except Exception:
                pass

        vectors = await self.model.encode([text])
        vector_list = vectors[0]

        if client:
            try:
                client.setex(
                    self._generate_cache_key(text),
                    86400,
                    json.dumps(vector_list)
                )
            except Exception as e:
                logger.warning(f"Failed to save Redis Cache: {e}")
        return vector_list
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        client = self._get_connection()
        results = [None] * len(texts)
        indices_to_compute = []
        texts_to_compute = []
        
        if client:
            try:
                pipe = client.pipeline()
                for text in texts:
                    pipe.get(self._generate_cache_key(text))
                cached_values = pipe.execute()
                
                for i, val in enumerate(cached_values):
                    if val:
                        results[i] = json.loads(val)
                    else:
                        indices_to_compute.append(i)
                        texts_to_compute.append(texts[i])
            except Exception:
                indices_to_compute = list(range(len(texts)))
                texts_to_compute = texts
        else:
            indices_to_compute = list(range(len(texts)))
            texts_to_compute = texts
            
        if texts_to_compute:
            logger.info(f"Computing embeddings for {len(texts_to_compute)}/{len(texts)} texts via TEI...")
            computed_vectors = await self.model.encode(texts_to_compute)
            
            for i, vector in enumerate(computed_vectors):
                original_index = indices_to_compute[i]
                results[original_index] = vector

            if client:
                try:
                    pipe = client.pipeline()
                    for i, vector in enumerate(computed_vectors):
                        key = self._generate_cache_key(texts_to_compute[i])
                        pipe.setex(key, 86400, json.dumps(vector))
                    pipe.execute()
                except Exception as e:
                    logger.warning(f"Failed to save Batch Cache: {e}")
                    
        return results


# Singleton instance — reuse Redis connection pool
_service_instance: "EmbeddingService | None" = None

def get_embedding_service() -> "EmbeddingService":
    global _service_instance
    if _service_instance is None:
        _service_instance = EmbeddingService()
    return _service_instance