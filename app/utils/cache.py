"""
utils/cache.py
Trách nhiệm:
  - Cache câu trả lời (TTL dài)
  - Cache embedding theo cụm từ (TTL ngắn)
"""
import time
from typing import Dict, Any, Tuple

# Cấu trúc cache câu trả lời: {md5_key: (response_data, expire_ts)}
_ANSWER_CACHE: Dict[str, Tuple[Dict[str, Any], float]] = {}
_ANSWER_CACHE_TTL = 3600  # Lưu cache 60 phút

# Request-scoped embedding cache: {md5(text): (vector, expire_ts)}
# TTL ngắn 5 giây — đủ cho 1 request cycle.
_EMBED_CACHE: Dict[str, Tuple[Any, float]] = {}
_EMBED_CACHE_TTL = 5.0

class MemoryCache:
    """Utility quản lý in-memory caches"""
    
    @staticmethod
    def get_answer(cache_key: str) -> Any:
        now = time.time()
        if cache_key in _ANSWER_CACHE:
            cached_data, expire_ts = _ANSWER_CACHE[cache_key]
            if now < expire_ts:
                return cached_data
            else:
                del _ANSWER_CACHE[cache_key]
        return None

    @staticmethod
    def set_answer(cache_key: str, data: Any, ttl: int = _ANSWER_CACHE_TTL):
        _ANSWER_CACHE[cache_key] = (data, time.time() + ttl)

    @staticmethod
    def get_embed(cache_key: str) -> Any:
        now = time.time()
        if cache_key in _EMBED_CACHE:
            cached_vec, expire_ts = _EMBED_CACHE[cache_key]
            if now < expire_ts:
                return cached_vec
            else:
                del _EMBED_CACHE[cache_key]
        return None

    @staticmethod
    def set_embed(cache_key: str, vector: Any, ttl: float = _EMBED_CACHE_TTL):
        _EMBED_CACHE[cache_key] = (vector, time.time() + ttl)