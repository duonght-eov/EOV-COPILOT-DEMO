"""
infrastructure/llm/stream_func.py
"""
from app.config import settings
from app.utils.logger import get_logger
from app.infrastructure.llm.openai_client import _get_llm_client, _get_response_llm_client

logger = get_logger("STREAM FUNC")

# Tắt thinking mode của Qwen3 — block <think>...</think> buffer toàn bộ phần suy nghĩ
# trước khi yield token đầu tiên, gây ra cảm giác "đơ" dài trước khi stream bắt đầu.
_NO_THINK_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}


async def stream_llm_func(prompt: str, system_prompt: str = None, **kwargs):
    client = _get_llm_client()
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": prompt})

    try:
        stream = await client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=messages,
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
            stream=True,
            extra_body=_NO_THINK_EXTRA,
        )
        async for chunk in stream:
            token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if token:
                yield token
    except Exception as e:
        logger.warning(f"[StreamLLM] Error: {e}")


async def stream_response_llm_func(prompt: str, system_prompt: str = None, **kwargs):
    model_name = settings.RESPONSE_LLM_MODEL_NAME
    if not model_name:
        async for token in stream_llm_func(prompt, system_prompt, **kwargs):
            yield token
        return

    client = _get_response_llm_client()
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": prompt})

    try:
        stream = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
            stream=True,
            extra_body=_NO_THINK_EXTRA,
        )
        async for chunk in stream:
            token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if token:
                yield token
    except Exception:
        async for token in stream_llm_func(prompt, system_prompt, **kwargs):
            yield token