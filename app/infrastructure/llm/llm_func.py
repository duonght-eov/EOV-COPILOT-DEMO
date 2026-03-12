"""
infrastructure/llm/llm_func.py
Trách nhiệm:
  - llm_completion_func: LLM cho indexing + entity extraction
  - query_llm_func: LLM cho query-time (keyword extraction, fast-fail)
  - _llm_call_with_retry: retry logic
"""
import re
import asyncio
from app.config import settings
from app.utils.logger import get_logger
from app.infrastructure.llm.openai_client import _get_llm_client

logger = get_logger("LLM_FUNC")

# Delimiter chính xác mà LightRAG sử dụng (lightrag/prompt.py line 8)
TUPLE_DELIMITER = "<|#|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"


def _is_extraction_task(system_prompt: str | None, prompt: str | None) -> bool:
    """Kiểm tra đây có phải extraction task không dựa vào sự có mặt của delimiter."""
    return TUPLE_DELIMITER in (system_prompt or "") or TUPLE_DELIMITER in (prompt or "")


def _fix_entity_fields(parts: list[str]) -> list[str] | None:
    """
    Chuẩn hóa entity record về đúng 4 fields:
    [entity, name, type, description]
    
    LightRAG parse: record_attributes[0]=entity, [1]=name, [2]=type, [3]=description
    Lỗi phổ biến: LLM trả về 5 fields (thêm 1 field thừa ở cuối)
    → nối fields thừa vào description
    """
    if not parts or "entity" not in parts[0]:
        return None
    if len(parts) == 4:
        return parts  # Đúng format
    if len(parts) < 4:
        # Thiếu field → không thể fix, bỏ qua
        return None
    # Thừa fields → nối description lại
    return [parts[0], parts[1], parts[2], " ".join(parts[3:])]


def _fix_relation_fields(parts: list[str]) -> list[str] | None:
    """
    Chuẩn hóa relation record về đúng 5 fields:
    [relation, src, tgt, keywords, description]
    
    LightRAG parse: [0]=relation, [1]=src, [2]=tgt, [3]=keywords, [4]=description
    Lỗi phổ biến LLM hay gặp:
      - 6 fields: chèn entity_type vào giữa → [relation, src, tgt, TYPE, keywords, description]
      - 4 fields: thiếu keywords → [relation, src, tgt, description]
    """
    if not parts or "relation" not in parts[0]:
        return None
    if len(parts) == 5:
        return parts  # Đúng format

    if len(parts) == 4:
        # Thiếu keywords field (4 fields) → thêm keyword mặc định
        return [parts[0], parts[1], parts[2], "related", parts[3]]

    if len(parts) == 6:
        # 6 fields: LLM hay chèn entity_type vào field[3]
        # Heuristic: nếu field[3] là 1 từ viết hoa (CamelCase) → là entity type → bỏ
        potential_type = parts[3]
        if potential_type and len(potential_type.split()) <= 2 and potential_type[0].isupper():
            logger.debug(f"[FIX] Dropping suspected entity type '{potential_type}' from relation")
            return [parts[0], parts[1], parts[2], parts[4], parts[5]]
        # Không xác định được → nối descriptions lại
        return [parts[0], parts[1], parts[2], parts[3], " ".join(parts[4:])]

    if len(parts) > 6:
        # Nhiều hơn 6: nối từ field[4] trở đi vào description
        return [parts[0], parts[1], parts[2], parts[3], " ".join(parts[4:])]

    return None


def _postprocess_extraction_output(content: str) -> str:
    """
    Post-process LLM output để đảm bảo đúng format LightRAG mong đợi:
    - Delimiter: <|#|>
    - Entity: đúng 4 fields
    - Relation: đúng 5 fields
    - Kết thúc bằng <|COMPLETE|>
    
    LightRAG đã có `fix_tuple_delimiter_corruption` xử lý <|> → <|#|>,
    nhưng KHÔNG xử lý số fields sai. Ta bổ sung phần đó.
    """
    lines = content.strip().splitlines()
    fixed_lines = []
    has_fix = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Completion delimiter
        if line in (COMPLETION_DELIMITER, "<|complete|>", "<| COMPLETE |>"):
            fixed_lines.append(COMPLETION_DELIMITER)
            continue

        # Bỏ dòng markdown/header
        if line.startswith(("```", "---")):
            continue

        # Chỉ xử lý dòng entity/relation
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
            # Normalize: đảm bảo field[0] luôn là "relation"
            fixed[0] = "relation"
            if fixed != parts:
                has_fix = True
            fixed_lines.append(TUPLE_DELIMITER.join(fixed))

    if has_fix:
        logger.warning("[POST-PROCESS] Fixed field count errors in extraction output")

    # Đảm bảo kết thúc bằng completion delimiter
    if fixed_lines and fixed_lines[-1] != COMPLETION_DELIMITER:
        fixed_lines.append(COMPLETION_DELIMITER)

    return "\n".join(fixed_lines)


async def _llm_call_with_retry(client, model, messages, temperature, max_tokens, max_retries=3, extra_body=None):
    """LLM call với retry mechanism."""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait = 2 ** attempt
                logger.warning(f"[RETRY] Attempt {attempt+1}/{max_retries} after {wait}s backoff...")
                await asyncio.sleep(wait)

            logger.info(f"[LLM CALL] Attempt {attempt+1}/{max_retries}...")

            call_kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
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
    """Wrapper cho LightRAG: gọi LLM và post-process output."""
    client = _get_llm_client()
    is_extraction = _is_extraction_task(system_prompt, prompt)

    # Phân loại call để log rõ ràng
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
    messages.append({"role": "user", "content": prompt})

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

    # Xóa thinking tags (DeepSeek-R1 / QwQ style)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    # Xóa markdown code fences
    content = content.replace("```json", "").replace("```", "").strip()
    # Xóa prefix rác phổ biến từ một số LLM
    if content.startswith("Based on"):
        content = content.split("\n", 1)[-1].strip()

    # Post-process: chỉ áp dụng cho extraction output (có delimiter <|#|>)
    if is_extraction and TUPLE_DELIMITER in content:
        content = _postprocess_extraction_output(content)

    return content


async def query_llm_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    **kwargs
) -> str:
    """LLM wrapper dùng riêng cho query time (fast-fail, timeout ngắn)."""
    client = _get_llm_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt + " /no_think"})

    try:
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
        content = response.choices[0].message.content or ""
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return content.replace("```json", "").replace("```", "").strip()
    except Exception as e:
        logger.warning(f"[QueryLLM] Failed/Timeout: {e} – returning empty")
        return ""
