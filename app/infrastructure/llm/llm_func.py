"""
infrastructure/llm/llm_func.py
"""
import re
import asyncio
from app.config import settings
from app.utils.logger import get_logger
from app.infrastructure.llm.openai_client import _get_llm_client

logger = get_logger("LLM_FUNC")

TUPLE_DELIMITER = "<|#|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"


def _is_extraction_task(system_prompt: str | None, prompt: str | None) -> bool:
    return TUPLE_DELIMITER in (system_prompt or "") or TUPLE_DELIMITER in (prompt or "")


def _fix_entity_fields(parts: list[str]) -> list[str] | None:
    if not parts or "entity" not in parts[0]:
        return None
    if len(parts) == 4:
        return parts
    if len(parts) < 4:
        return None
    return [parts[0], parts[1], parts[2], " ".join(parts[3:])]


def _fix_relation_fields(parts: list[str]) -> list[str] | None:
    if not parts or "relation" not in parts[0]:
        return None
    if len(parts) == 5:
        return parts
    if len(parts) == 4:
        return [parts[0], parts[1], parts[2], "related", parts[3]]
    if len(parts) == 6:
        potential_type = parts[3]
        if potential_type and len(potential_type.split()) <= 2 and potential_type[0].isupper():
            logger.debug(f"[FIX] Dropping suspected entity type '{potential_type}' from relation")
            return [parts[0], parts[1], parts[2], parts[4], parts[5]]
        return [parts[0], parts[1], parts[2], parts[3], " ".join(parts[4:])]
    if len(parts) > 6:
        return [parts[0], parts[1], parts[2], parts[3], " ".join(parts[4:])]
    return None


def _postprocess_extraction_output(content: str) -> str:
    lines = content.strip().splitlines()
    fixed_lines = []
    has_fix = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line in (COMPLETION_DELIMITER, "<|complete|>", "<| COMPLETE |>"):
            fixed_lines.append(COMPLETION_DELIMITER)
            continue
        if line.startswith(("```", "---")):
            continue
        if not (line.startswith("entity") or line.startswith("relation")):
            continue

        parts = [p.strip() for p in line.split(TUPLE_DELIMITER)]
        row_type = parts[0].lower() if parts else ""

        if row_type == "entity":
            fixed = _fix_entity_fields(parts)
            if fixed is None:
                has_fix = True
                logger.debug(f"[FIX] Dropped malformed entity: {line[:100]}")
                continue
            if fixed != parts:
                has_fix = True
            fixed_lines.append(TUPLE_DELIMITER.join(fixed))
        elif "relation" in row_type:
            fixed = _fix_relation_fields(parts)
            if fixed is None:
                has_fix = True
                logger.debug(f"[FIX] Dropped malformed relation: {line[:100]}")
                continue
            fixed[0] = "relation"
            if fixed != parts:
                has_fix = True
            fixed_lines.append(TUPLE_DELIMITER.join(fixed))

    if has_fix:
        logger.warning("[POST-PROCESS] Fixed field count errors in extraction output")

    if fixed_lines and fixed_lines[-1] != COMPLETION_DELIMITER:
        fixed_lines.append(COMPLETION_DELIMITER)

    return "\n".join(fixed_lines)


async def _llm_call_with_retry(client, model, messages, temperature, max_tokens, max_retries=3, extra_body=None):
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait = 2 ** attempt
                logger.warning(f"[RETRY] Attempt {attempt+1}/{max_retries} after {wait}s backoff...")
                await asyncio.sleep(wait)

            logger.info(f"[LLM CALL] Attempt {attempt+1}/{max_retries}...")
            call_kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            if extra_body:
                call_kwargs["extra_body"] = extra_body

            response = await client.chat.completions.create(**call_kwargs)
            content = response.choices[0].message.content
            logger.info(f"[LLM RESPONSE] Length: {len(content) if content else 0} chars")

            if not content or len(content.strip()) < 10:
                if attempt < max_retries - 1:
                    logger.warning("[RETRY TRIGGER] Empty/short response, retrying...")
                    continue
                logger.error(f"[RETRY FAILED] Empty response after {max_retries} attempts")
                return ""

            logger.info(f"[LLM SUCCESS] Valid response on attempt {attempt+1}")
            return content

        except Exception as e:
            logger.error(f"[LLM ERROR] Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** (attempt + 1))
                continue
            raise e

    return ""


async def llm_completion_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    **kwargs
) -> str:
    """LLM wrapper cho indexing + entity extraction. Có retry và post-processing."""
    client = _get_llm_client()
    is_extraction = _is_extraction_task(system_prompt, prompt)

    combined = (system_prompt or "") + (prompt or "")
    if TUPLE_DELIMITER in combined:
        task_type = "GLEANING" if history_messages else "EXTRACT"
    elif "summarize" in combined.lower() or "existing descriptions" in combined.lower():
        task_type = "ENTITY_MERGE"
    else:
        task_type = "OTHER"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    
    final_prompt = prompt + " /no_think" if is_extraction else prompt
    messages.append({"role": "user", "content": final_prompt})

    logger.info(f"[{task_type}] prompt_len={len(prompt)}")

    content = await _llm_call_with_retry(
        client=client,
        model=settings.LLM_MODEL_NAME,
        messages=messages,
        temperature=kwargs.get("temperature", 0),
        max_tokens=kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
        max_retries=1 if is_extraction else 3,
        extra_body={"think": False} if is_extraction else None,
    )

    if not content:
        return ""

    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    content = content.replace("```json", "").replace("```", "").strip()
    if content.startswith("Based on"):
        content = content.split("\n", 1)[-1].strip()

    if is_extraction and TUPLE_DELIMITER in content:
        content = _postprocess_extraction_output(content)

    return content


async def query_llm_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    **kwargs
) -> str:
    """
    LLM wrapper dùng cho query-time.
    - Dùng cho keyword extraction (max 256 tokens, timeout ngắn, fail-fast).
    - KHÔNG dùng cho sinh câu trả lời dài — dùng response_llm_func cho việc đó.
    """
    client = _get_llm_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt + " /no_think"})

    try:
        import time as _time
        t0 = _time.perf_counter()
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.LLM_MODEL_NAME,
                messages=messages,
                temperature=0,
                max_tokens=min(kwargs.get("max_tokens", 256), 256),
                extra_body={"think": False},
            ),
            timeout=settings.LLM_TIMEOUT,
        )
        elapsed_ms = (_time.perf_counter() - t0) * 1000
        content = response.choices[0].message.content or ""
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        content = content.replace("```json", "").replace("```", "").strip()

        usage = getattr(response, 'usage', None)
        if usage:
            tok_per_sec = getattr(usage, 'completion_tokens', 0) / max(elapsed_ms / 1000, 0.001)
            logger.info(
                f"[QueryLLM] {elapsed_ms:.0f}ms | "
                f"prompt={getattr(usage, 'prompt_tokens', 0)}tok, "
                f"completion={getattr(usage, 'completion_tokens', 0)}tok @ {tok_per_sec:.1f} tok/s"
            )
        else:
            logger.info(f"[QueryLLM] {elapsed_ms:.0f}ms | {len(content)} chars")

        return content
    except Exception as e:
        logger.warning(f"[QueryLLM] Failed/Timeout: {e} – returning empty")
        return ""


async def response_llm_func(
    prompt: str,
    system_prompt: str = None,
    **kwargs
) -> str:
    """
    LLM wrapper dùng để sinh câu trả lời RAG đầy đủ (non-streaming).
    Dùng LLM_MAX_TOKENS, không hard-cap 256 tokens như query_llm_func.
    Sử dụng RESPONSE_LLM_MODEL_NAME nếu có, ngược lại dùng LLM mặc định.
    """
    from app.infrastructure.llm.openai_client import _get_response_llm_client

    model_name = settings.RESPONSE_LLM_MODEL_NAME or settings.LLM_MODEL_NAME
    if settings.RESPONSE_LLM_MODEL_NAME:
        client = _get_response_llm_client()
        timeout = float(settings.RESPONSE_LLM_TIMEOUT)
    else:
        client = _get_llm_client()
        timeout = float(settings.LLM_TIMEOUT)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=kwargs.get("temperature", 0),
                max_tokens=kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
            ),
            timeout=timeout,
        )
        content = response.choices[0].message.content or ""
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return content
    except Exception as e:
        logger.warning(f"[ResponseLLM] Failed/Timeout: {e}")
        return ""
