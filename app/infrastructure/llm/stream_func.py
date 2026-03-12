"""
infrastructure/llm/stream_func.py
Trách nhiệm:
  - stream_llm_func: Stream từ LLM chính token by token
  - stream_response_llm_func: Stream từ Response LLM (qwen2.5:7b riêng)
  - Xử lý <think>...</think> blocks trong streaming
"""
from app.config import settings
from app.utils.logger import get_logger
from app.infrastructure.llm.openai_client import _get_llm_client, _get_response_llm_client

logger = get_logger("STREAM FUNC")

async def stream_llm_func(prompt: str, system_prompt: str = None, **kwargs):
    client = _get_llm_client()
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": prompt})
    think_buf = ""
    in_think = False

    try:
        stream = await client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=messages,
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
            stream=True,
        )
        async for chunk in stream:
            token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if not token: continue
            
            if in_think:
                think_buf += token
                if "</think>" in think_buf:
                    after = think_buf.split("</think>", 1)[1]
                    in_think, think_buf = False, ""
                    if after: yield after
                continue

            if "<think>" in token:
                parts = token.split("<think>", 1)
                if parts[0]: yield parts[0]
                think_buf = parts[1]
                in_think = True
                continue
            
            yield token
    except Exception as e:
        logger.warning(f"[StreamLLM] Error: {e}")

async def stream_response_llm_func(prompt: str, system_prompt: str = None, **kwargs):
    model_name = settings.RESPONSE_LLM_MODEL_NAME
    if not model_name:
        async for token in stream_llm_func(prompt, system_prompt, **kwargs): yield token
        return
    client = _get_response_llm_client()
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": prompt})
    think_buf = ""
    in_think = False

    try:
        stream = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
            stream=True,
        )
        async for chunk in stream:
            token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if not token: continue
            if in_think:
                think_buf += token
                if "</think>" in think_buf:
                    after = think_buf.split("</think>", 1)[1]
                    in_think, think_buf = False, ""
                    if after: yield after
                continue
            if "<think>" in token:
                parts = token.split("<think>", 1)
                if parts[0]: yield parts[0]
                think_buf, in_think = parts[1], True
                continue
            yield token
    except Exception:
        async for token in stream_llm_func(prompt, system_prompt, **kwargs): yield token