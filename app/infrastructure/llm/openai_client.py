"""
infrastructure/llm/openai_client.py
"""
import threading
import httpx
from openai import AsyncOpenAI
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("LLM CLIENT")

_llm_client: AsyncOpenAI | None = None
_response_llm_client: AsyncOpenAI | None = None
_lock = threading.Lock()


def _get_llm_client() -> AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        with _lock:
            if _llm_client is None:
                _llm_client = AsyncOpenAI(
                    api_key=settings.LLM_API_KEY,
                    base_url=settings.LLM_BASE_URL,
                    timeout=httpx.Timeout(
                        connect=10.0,
                        read=settings.LLM_TIMEOUT,
                        write=10.0,
                        pool=5.0,
                    ),
                )
                logger.info("[LLM CLIENT] Initialized LLM client")
    return _llm_client


def _get_response_llm_client() -> AsyncOpenAI:
    global _response_llm_client
    if _response_llm_client is None:
        with _lock:
            if _response_llm_client is None:
                _response_llm_client = AsyncOpenAI(
                    api_key=settings.RESPONSE_LLM_API_KEY,
                    base_url=settings.RESPONSE_LLM_BASE_URL,
                    timeout=httpx.Timeout(
                        connect=10.0,
                        read=float(settings.RESPONSE_LLM_TIMEOUT),
                        write=10.0,
                        pool=5.0,
                    ),
                )
                logger.info("[LLM CLIENT] Initialized Response LLM client")
    return _response_llm_client