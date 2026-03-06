import os
import ujson as json
import httpx
import numpy as np
from datetime import datetime
import asyncio
import base64
import re
from typing import Dict, Any, List, Optional, Union
from minio import Minio
from lightrag import LightRAG
from openai import AsyncOpenAI
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig
from app.config import settings
from app.utils.logger import get_logger
from app.utils.http_client import get_indexing_client
from app.services.text_chunker import CustomChunker, tokenizer
from app.services.context_builder import ContextBuilder


logger = get_logger("INDEXING ENGINE")

async def embedding_func(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.array([])
    try:
        client = get_indexing_client()
        payload = {
            "texts": texts,
            "model": settings.EMBEDDING_MODEL_NAME,
        }
        base_url = settings.EMBEDDING_SERVICE_URL.rstrip("/")
        url = f"{base_url}/api/v1/embed/batch"
        logger.info(f"Calling Embedding: {len(texts)} texts → {url}")
        response = await client.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        embeddings = result.get("vectors", [])
        if embeddings and isinstance(embeddings[0], list):
            logger.info(f"Embedding response: {len(embeddings)} vectors received")
            return np.array(embeddings)
        else:
            logger.warning(f"Unexpected embedding format: {type(embeddings)}")
            return np.array(embeddings)
    except Exception as e:
        logger.error(f"Embedding Service Failed: {e}")
        raise e

# LLM client dùng cho indexing + keyword extraction
_llm_client: Optional[AsyncOpenAI] = None

def _get_llm_client() -> AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            timeout=httpx.Timeout(
                connect=10.0,
                read=settings.LLM_TIMEOUT,
                write=10.0,
                pool=5.0
            )
        )
    return _llm_client


# Response LLM client dùng riêng để sinh câu trả lời (qwen2.5:7b trên máy 10.0.0.125)
_response_llm_client: Optional[AsyncOpenAI] = None

def _get_response_llm_client() -> AsyncOpenAI:
    global _response_llm_client
    if _response_llm_client is None:
        base_url = settings.RESPONSE_LLM_BASE_URL or settings.LLM_BASE_URL
        api_key = settings.RESPONSE_LLM_API_KEY or settings.LLM_API_KEY
        timeout = settings.RESPONSE_LLM_TIMEOUT or settings.LLM_TIMEOUT
        _response_llm_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(timeout),
                write=10.0,
                pool=5.0
            )
        )
    return _response_llm_client


# --- LLM FUNCTION ---
async def llm_completion_func(prompt: str, system_prompt: str=None, history_messages: list=[], **kwargs) -> str:
    """
    Custom OpenAI wrapper to ensure compatibility independent of LightRAG internals.
    Includes hack to swap <|#|> delimiter with #### for better Qwen compatibility.
    """
    client = _get_llm_client()

    # --- HACK: Delimiter Swapping for Qwen ---
    # Qwen tokenizer might eat <|#|> or treat it as special/garbage.
    # We allow the library to use <|#|> but tell the model to use ####.
    original_delimiter = "<|#|>"
    temp_delimiter = "####"
    is_extraction_task = False

    if (system_prompt and original_delimiter in system_prompt) or (prompt and original_delimiter in prompt):
        is_extraction_task = True
        if system_prompt:
            system_prompt = system_prompt.replace(original_delimiter, temp_delimiter)
        if prompt:
            prompt = prompt.replace(original_delimiter, temp_delimiter)
            guidance = (
                f"\n\nCRITICAL FORMAT RULES:"
                f"\n- Use '{temp_delimiter}' as separator (NOT commas, colons, pipes, or markdown)"
                f"\n- Each entity line must have EXACTLY 4 fields: entity{temp_delimiter}Name{temp_delimiter}Type{temp_delimiter}Description"
                f"\n- Each relation line must have EXACTLY 5 fields: relation{temp_delimiter}Source{temp_delimiter}Target{temp_delimiter}Keywords{temp_delimiter}Description"
                f"\n- Example entity: entity{temp_delimiter}Solar Panel{temp_delimiter}Equipment{temp_delimiter}A photovoltaic device that converts sunlight to electricity"
                f"\n- Example relation: relation{temp_delimiter}Solar Panel{temp_delimiter}Power Grid{temp_delimiter}connects to{temp_delimiter}Solar panels supply power to the grid"
                f"\n- Do NOT use Markdown code blocks or bullet points. Plain text only."
            )
            if "CRITICAL FORMAT RULES" not in prompt:
                prompt += guidance
    # -----------------------------------------
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    try:
        content = await _llm_call_with_retry(
            client=client,
            model=settings.LLM_MODEL_NAME,
            messages=messages,
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
            max_retries=3
        )
        


        if content:
            # --- GLOBAL CLEANING (ALL TASKS) ---
            # 1. Clean <think> blocks (DeepSeek R1/V3 spew)
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            # 2. Clean Markdown Wrappers
            content = content.replace("```json", "").replace("```csv", "").replace("```xml", "").replace("```", "").strip()

            # 3. Remove conversational prefixes (common in DeepSeek/Qwen)
            if content.startswith("Based on the input") or content.startswith("Based on the provided"):
                if "\n" in content:
                    content = content.split("\n", 1)[-1].strip()

            # 4. JSON rescue: nếu content chứa JSON object nhưng có text thừa phía trước/sau
            # Điều này xảy ra khi Qwen thêm giải thích trước JSON (keyword extraction task)
            if not content.startswith("{") and "{" in content and "}" in content:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0).strip()

        # --- SPECIALIZED CLEANING FOR ENTITY EXTRACTION ---
        if is_extraction_task and content:
            # DEBUG LOGGING (RAW)
            logger.info(f"\n [DEBUG EXTRACTION REPAIR]: Processing {len(content)} chars...\n--------------------------------------------------")
            
            # 1. Swap delimiter Back (#### -> <|#|>)
            content = content.replace(temp_delimiter, original_delimiter)
            
            # --- ROBUST OUTPUT REPAIR STRATEGY ---
            
            # 1. Normalize Delimiters (Regex Power)
            # Fix ####, <###>, <|>, |#| and accidental wrappers like <<|#|>> or >####<
            # Step A: Convert known variants to a temporary unique placeholder
            content = re.sub(r'#{4,}', '<|#|>', content)  # ####
            content = content.replace("<###>", "<|#|>")
            content = content.replace("<|>", "<|#|>")
            content = content.replace("|#|", "<|#|>")
            
            # Step B: Fix doubled/wrapped delimiters (e.g. <<|#|>> -> <|#|>)
            # This regex looks for <|#|> surrounded by extra <, >, or spaces
            content = re.sub(r'[<>\s]*<\|#\|>[<>\s]*', '<|#|>', content)

            # Step C: Fix Missing Newlines (e.g. ...<|#|>entity... on same line)
            # If 'entity' or 'relation' appears in the middle of a line, we force a newline.
            # Be careful not to break descriptions containing the word "entity".
            # Pattern: <|#|> followed by 'entity' or 'relation'
            content = content.replace("<|#|>entity", "<|#|>\nentity")
            content = content.replace("<|#|>relation", "<|#|>\nrelation")
            
            # Also handle if they used ####entity (before normalization) but we already prioritized normalization.
            # Let's handle generic case: normalized delimiter + start keyword
            
            fixed_lines = []
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if not line:
                    continue

                # 0. ROBUST DELIMITER NORMALIZATION (Fix LLM Hallucinations like ### instead of <|#|>)
                # If line starts with "entity" or "relation" and contains "###", replace "###" with proper delimiter
                if (line.startswith("entity") or line.startswith("relation")) and "###" in line:
                    line = line.replace("###", original_delimiter)
                
                # 0b. Normalize "relationship" -> "relation" (LLM often outputs this)
                if line.startswith("relationship" + original_delimiter):
                    line = "relation" + line[len("relationship"):]
                    logger.debug(f"[Normalize] relationship → relation")

                # DEBUG PARSING
                if line.startswith("relation"):
                    d_parts = line.split(original_delimiter)
                    if len(d_parts) != 5:
                        logger.warning(f" [Fixing Relation] Original ({len(d_parts)} parts): {line[:100]}...")
                
                # 1. Fix "entity<Type>Name" hallucination (DeepSeek specific)
                # ... (Keep existing logic)
                if line.startswith("entity<") and ">" in line and original_delimiter not in line[:20]:
                    try:
                        match = re.match(r"entity<([^>]+)>([^<]+)(.*)", line)
                        if match:
                            e_type, e_name, rest = match.groups()
                            # Reconstruct
                            line = f"entity{original_delimiter}{e_name.strip()}{original_delimiter}{e_type.strip()}{original_delimiter}{rest.strip().replace('<###>', '')}"
                    except:
                        pass
                
                # 3a. Auto-fix Entity Fields (Ensure 4 fields: entity, name, type, desc)
                if line.startswith("entity" + original_delimiter):
                    parts = line.split(original_delimiter)
                    if len(parts) == 2:
                        # Only name, missing type and description
                        line += original_delimiter + "Concept" + original_delimiter + " "
                        logger.debug(f"[Auto-fix] Added type+desc to 2-field entity: {parts[1]}")
                    elif len(parts) == 3:
                        # Missing Description only
                        line += original_delimiter + " "
                    elif len(parts) > 4:
                        # Extra fields -> Merge into Description
                        merged_desc = " - ".join(parts[3:])
                        line = original_delimiter.join(parts[:3] + [merged_desc])
                
                # 3b. Auto-fix Relation Fields (Ensure 5 fields)
                # Fix cases where LLM returns 3 or 4 fields only
                # 3b. Auto-fix Relation Fields (Ensure 5 fields: relation, src, tgt, keywords, description)
                if line.startswith("relation" + original_delimiter):
                    parts = line.split(original_delimiter)
                    
                    # Case: Found 4 parts (relation, src, tgt, description) -> Insert keywords at position 3
                    if len(parts) == 4:
                        # Rebuild: relation|src|tgt|keywords|description
                        line = original_delimiter.join([
                            parts[0],  # relation
                            parts[1],  # src
                            parts[2],  # tgt
                            "related", # keywords (inserted)
                            parts[3]   # description
                        ])
                        logger.debug(f"[Auto-fix] Inserted keywords into 4-field relation: {parts[1]} → {parts[2]}")
                    
                    # Case: Valid 5 parts -> Do nothing
                    
                    # Case: Extra fields (>5) -> Keep first 5
                    elif len(parts) > 5:
                         # parts[0]=relation, [1]=src, [2]=tgt, [3]=keywords, [4]=desc, [5...]=extra
                         # Keep first 5 fields
                         line = original_delimiter.join(parts[:5])
                         logger.debug(f"[Auto-fix] Trimmed {len(parts)}-field relation to 5 fields")
                        
                    # Case: Highly incomplete (only relation, src, tgt)
                    elif len(parts) == 3:
                         # Rebuild: relation|src|tgt|keywords|description
                         line = original_delimiter.join([
                             parts[0],  # relation
                             parts[1],  # src
                             parts[2],  # tgt
                             "related", # keywords
                             f"Quan hệ giữa {parts[1]} và {parts[2]}"  # non-empty description
                         ])
                         logger.debug(f"[Auto-fix] Added keywords+desc to 3-field relation: {parts[1]} → {parts[2]}")

                
                if line.startswith("relation"):
                    check_parts = line.split(original_delimiter)
                    if len(check_parts) != 5:
                        logger.error(f"[Still Wrong Relation] ({len(check_parts)}): {line}")

                fixed_lines.append(line)
            
            content = "\n".join(fixed_lines)
            
            # DEBUG LOGGING
            logger.info(f"\n [DEBUG PROCESSED RESPONSE (First 300 chars)]:\n{content[:300]}...\n--------------------------------------------------")
            
        return content
    except Exception as e:
        logger.error(f"LLM Call Failed: {e}")
        return ""


# --- QUERY-TIME LLM FUNCTION ---
async def query_llm_func(prompt: str, system_prompt: str = None, history_messages: list = [], **kwargs) -> str:
    """
    LLM wrapper dùng riêng cho query time (keyword extraction, RAG response generation).
    - Timeout ngắn (30s) để fail fast, không block chat streaming.
    - Không retry: nếu LLM chậm/lỗi → trả "" để ConsensusRetriever fallback raw query.
    - Strip <think> blocks (DeepSeek-R1 / Qwen3 reasoning mode).
    """
    import asyncio

    client = _get_llm_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    max_tokens = min(kwargs.get("max_tokens", 256), 256)  # extraction chỉ cần ~20 tokens

    # Tắt thinking mode cho qwen3 — ngăn model generate hàng trăm <think> tokens ẩn
    # Ollama hỗ trợ /no_think suffix; OpenAI-compat hỗ trợ extra_body={"think": False}
    final_messages = list(messages)
    if final_messages and final_messages[-1]["role"] == "user":
        final_messages[-1] = {
            "role": "user",
            "content": final_messages[-1]["content"] + " /no_think"
        }

    try:
        import time as _time
        t0 = _time.perf_counter()
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.LLM_MODEL_NAME,
                messages=final_messages,
                temperature=0,
                max_tokens=max_tokens,
                extra_body={"think": False},  # Ollama/vLLM: tắt reasoning
            ),
            timeout=settings.LLM_TIMEOUT,
        )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        content = response.choices[0].message.content or ""
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        content = content.replace("```json", "").replace("```", "").strip()

        # Tính tốc độ từ usage nếu có
        usage = getattr(response, 'usage', None)
        if usage:
            prompt_tok = getattr(usage, 'prompt_tokens', 0)
            completion_tok = getattr(usage, 'completion_tokens', 0)
            tok_per_sec = completion_tok / max(elapsed_ms / 1000, 0.001)
            logger.info(
                f"[QueryLLM] {elapsed_ms:.0f}ms | "
                f"prompt={prompt_tok}tok, completion={completion_tok}tok @ {tok_per_sec:.1f} tok/s"
            )
        else:
            logger.info(f"[QueryLLM] {elapsed_ms:.0f}ms | {len(content)} chars")

        return content

    except asyncio.TimeoutError:
        logger.warning(f"[QueryLLM] Timeout {settings.LLM_TIMEOUT}s – returning empty")
        return ""
    except Exception as e:
        logger.warning(f"[QueryLLM] Failed: {e} – returning empty")
        return ""


async def stream_llm_func(prompt: str, system_prompt: str = None, **kwargs):
    """
    Stream LLM response token by token. Strip <think>...</think> blocks on the fly.
    Yields str tokens. On error/timeout yields nothing (caller handles gracefully).
    """
    client = _get_llm_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    max_tokens = min(kwargs.get("max_tokens", settings.LLM_MAX_TOKENS), settings.LLM_MAX_TOKENS)

    # State machine để bỏ <think>...</think> trong streaming
    think_buf = ""
    in_think = False

    try:
        stream = await client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=messages,
            temperature=kwargs.get("temperature", 0),
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if not token:
                continue

            # Xử lý <think>...</think> blocks
            if in_think:
                think_buf += token
                if "</think>" in think_buf:
                    after = think_buf.split("</think>", 1)[1]
                    in_think = False
                    think_buf = ""
                    if after:
                        yield after
                continue

            if "<think>" in token:
                parts = token.split("<think>", 1)
                if parts[0]:
                    yield parts[0]
                think_buf = parts[1]
                in_think = True
                continue

            yield token

    except Exception as e:
        logger.warning(f"[StreamLLM] Error: {e}")


async def stream_response_llm_func(prompt: str, system_prompt: str = None, **kwargs):
    """
    Stream câu trả lời dùng Response LLM (qwen2.5:7b trên 10.0.0.125).
    Nếu RESPONSE_LLM_MODEL_NAME chưa cấu hình thì fallback về stream_llm_func.
    """
    model_name = settings.RESPONSE_LLM_MODEL_NAME
    if not model_name:
        logger.warning("[StreamResponseLLM] RESPONSE_LLM_MODEL_NAME chưa set — fallback về LLM chính")
        async for token in stream_llm_func(prompt, system_prompt, **kwargs):
            yield token
        return

    client = _get_response_llm_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    max_tokens = min(kwargs.get("max_tokens", settings.LLM_MAX_TOKENS), settings.LLM_MAX_TOKENS)

    think_buf = ""
    in_think = False

    try:
        logger.info(f"[StreamResponseLLM] Using {model_name} @ {settings.RESPONSE_LLM_BASE_URL}")
        stream = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0),
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if not token:
                continue

            if in_think:
                think_buf += token
                if "</think>" in think_buf:
                    after = think_buf.split("</think>", 1)[1]
                    in_think = False
                    think_buf = ""
                    if after:
                        yield after
                continue

            if "<think>" in token:
                parts = token.split("<think>", 1)
                if parts[0]:
                    yield parts[0]
                think_buf = parts[1]
                in_think = True
                continue

            yield token

    except Exception as e:
        logger.warning(f"[StreamResponseLLM] Error: {e} — fallback về LLM chính")
        async for token in stream_llm_func(prompt, system_prompt, **kwargs):
            yield token

def lightrag_chunking_adapter(*args, **kwargs) -> List[Dict[str, Any]]:
    # Adapt to 6-argument changes from lightrag.py
    text = args[1] if len(args) >= 2 and isinstance(args[1], str) else kwargs.get("content", args[0] if args else "")
    chunk_token_size = args[5] if len(args) >= 6 else kwargs.get("chunk_token_size", 1200)
    chunk_overlap_token_size = args[4] if len(args) >= 5 else kwargs.get("chunk_overlap_token_size", 100)

    chunker = CustomChunker(
        target_chunk_size=chunk_token_size,
        chunk_overlap=chunk_overlap_token_size
    )

    chunks = chunker.process(text, doc_id="embedded_doc")

    results = []
    for idx, c in enumerate(chunks):
        content = c.get("content", "")
        tokens = tokenizer.count(content)
            
        results.append({
            "content": content,
            "tokens": tokens,
            "chunk_order_index": idx,
            "source_id": f"chunk-{idx}", 
            "metadata": {
                "page_idx": c.get("page_idx"), 
                "type": "text"
            }
        })
    
    return results

# Cache trạng thái VLM để không retry khi đã biết offline
_vlm_offline: bool = False
_vlm_last_check: float = 0.0
_VLM_RECHECK_INTERVAL = 120.0  # giây — thử lại VLM sau 2 phút

# --- VLM ADAPTER ---
async def vlm_model_func(prompt: str, images: list[str] = [], **kwargs) -> str:
    """
    Adapter cho Vision Language Model (Qwen-VL).
    Dùng để sinh caption cho ảnh hoặc mô tả bảng biểu.
    """
    global _vlm_offline, _vlm_last_check
    import time

    if not images:
        return await llm_completion_func(prompt, **kwargs)

    # Skip nhanh nếu VLM đã biết offline (tránh chờ timeout per-image)
    if _vlm_offline and (time.time() - _vlm_last_check) < _VLM_RECHECK_INTERVAL:
        logger.warning(f"[VLM] Service offline — skipping image caption")
        return "Hình ảnh minh họa (VLM service không khả dụng)."

    # Helper function to encode image to base64
    def encode_image(image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to read/encode image {image_path}: {e}")
            return None

    # Xây dựng payload theo chuẩn OpenAI Vision (hỗ trợ bởi vLLM/Ollama)
    content = [{"type": "text", "text": prompt}]
    for img_path in images:
        # Chuyển đổi file local sang base64 để gửi qua API
        base64_image = encode_image(img_path)
        if base64_image:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        else:
             logger.warning(f"Skipping unreadable image: {img_path}")

    # Thêm system message để force Vietnamese output
    system_prompt_use = kwargs.get("system_prompt", "Bạn là một trợ lý AI chuyên phân tích hình ảnh. LUÔN trả lời bằng Tiếng Việt. Tập trung vào mô tả chi tiết, KHÔNG giải thích quá trình suy nghĩ.")
    
    messages = [
        {
            "role": "system", 
            "content": system_prompt_use
        },
        {
            "role": "user", 
            "content": content
        }
    ]
    
    try:
        headers = {
            "Authorization": f"Bearer {settings.VLM_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": settings.VLM_MODEL_NAME,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2048,  # Tăng từ 1024 lên 2048 để đủ cho content
            "top_p": 0.9
        }
        
        base_url = settings.VLM_BASE_URL.rstrip("/")
        url = f"{base_url}/chat/completions"
        
        # connect timeout ngắn (5s) để fail nhanh khi máy VLM down
        # read timeout 360s để chờ model generate caption phức tạp
        vlm_timeout = httpx.Timeout(connect=5.0, read=360.0, write=10.0, pool=5.0)
        async with httpx.AsyncClient(timeout=vlm_timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            res_json = response.json()
            
            message = res_json["choices"][0]["message"]
            content_text = message.get("content", "")
            
            # Nếu content empty, KHÔNG dùng reasoning, mà retry với prompt đơn giản hơn
            if not content_text or len(content_text.strip()) < 20:
                logger.warning(f"VLM returned empty/short content. Retrying with simplified prompt...")
                
                # Retry với prompt đơn giản hơn
                simple_messages = [
                    {
                        "role": "system",
                        "content": "Mô tả hình ảnh bằng Tiếng Việt."
                    },
                    {
                        "role": "user",
                        "content": content  # Reuse same content
                    }
                ]
                
                retry_payload = {
                    "model": settings.VLM_MODEL_NAME,
                    "messages": simple_messages,
                    "temperature": 0.3,
                    "max_tokens": 2048,
                }
                
                retry_response = await client.post(url, json=retry_payload, headers=headers)
                retry_response.raise_for_status()
                retry_json = retry_response.json()
                
                content_text = retry_json["choices"][0]["message"].get("content", "")
                
                if content_text and len(content_text.strip()) > 20:
                    logger.info("Retry successful - got Vietnamese description")
                else:
                    logger.error(f"VLM still returned empty after retry! Raw response: {retry_json}")
                    return "Không thể tạo mô tả cho hình ảnh này."

            return content_text
            
    except asyncio.TimeoutError:
        logger.warning(f"[VLM] Timed out after 600s for images {[os.path.basename(p) for p in images]}, skipping.")
        return "Hình ảnh minh họa (VLM timeout — bỏ qua)."
    except Exception as e:
        err_str = str(e)
        if "connection" in err_str.lower() or "connect" in err_str.lower() or "All connection" in err_str:
            import time
            _vlm_offline = True
            _vlm_last_check = time.time()
            logger.warning(f"[VLM] Marked offline for {_VLM_RECHECK_INTERVAL}s. Will retry later.")
        logger.error(f"VLM Call Failed for images {images}: {e}")
        return "Hình ảnh minh họa (VLM service không khả dụng)."

# Removed duplicated function lightrag_chunking_adapter

# --- PROMPT LOADER ---
def load_jinja_prompts(file_path: str) -> Dict[str, str]:
    """
    Parses a Jinja2-like file and extracts blocks defined by {% block name %}...{% endblock %}.
    Returns a dict {block_name: block_content}.
    """
    prompts = {}
    if not os.path.exists(file_path):
        logger.warning(f"Prompt file not found: {file_path}")
        return prompts
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        pattern = re.compile(r"{% block (\w+) %}(.*?){% endblock %}", re.DOTALL)
        matches = pattern.findall(content)
        
        for block_name, block_content in matches:
            cleaned = block_content.strip()
            cleaned = cleaned.replace("{{ tuple_delimiter }}", "<|>")
            cleaned = cleaned.replace("{{ completion_delimiter }}", "<|COMPLETE|>")
            cleaned = re.sub(r"\{\{ '\{(\w+)\}' \}\}", r"{\1}", cleaned)
            
            prompts[block_name] = cleaned
            
        logger.info(f"Loaded {len(prompts)} custom prompts from {file_path}")
        return prompts
    except Exception as e:
        logger.error(f"Failed to load prompts: {e}")
        return {}

# Global caches
_prompt_config = None
_chunker = None
_rag_instances: Dict[str, LightRAG] = {}
_rag_locks: Dict[str, asyncio.Lock] = {}
_rag_anything_instances: Dict[str, RAGAnything] = {}  # Cache RAGAnything per workspace

class IndexingEngine:
    def __init__(self, doc_id: str = None):
        self.doc_id = doc_id
        self.context_builder = ContextBuilder(context_window=2, max_context_chars=500)
        
        self.minio_client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )
        logger.debug(f"MinIO client initialized: {settings.MINIO_ENDPOINT}")
        
        self._load_prompts_and_config()

    def _load_prompts_and_config(self):
        global _prompt_config, _chunker
        
        if _prompt_config is not None:
            self.vlm_prompts = _prompt_config["vlm_prompts"]
            return

        prompt_extractor_path = os.path.join(settings.RAG_WORK_DIR, "prompts", "prompt_extractor.jinja")
        if not os.path.exists(prompt_extractor_path):
            prompt_extractor_path = "/home/datpt/projects/EOVCopilot-Demo/services/rag-service/prompts/prompt_extractor.jinja"
        custom_prompts = load_jinja_prompts(prompt_extractor_path)

        processor_prompt_path = os.path.join(settings.RAG_WORK_DIR, "prompts", "processor_prompts.jinja")
        if not os.path.exists(processor_prompt_path):
            processor_prompt_path = "/home/datpt/projects/EOVCopilot-Demo/services/rag-service/prompts/processor_prompts.jinja"
        multimodal_prompts = load_jinja_prompts(processor_prompt_path)
        if multimodal_prompts:
            import raganything.prompt
            logger.info(f"Overriding RAGAnything prompts with {len(multimodal_prompts)} templates")
            raganything.prompt.PROMPTS.update(multimodal_prompts)

        vlm_prompt_path = os.path.join(settings.RAG_WORK_DIR, "prompts", "vlm_prompts.jinja")
        if not os.path.exists(vlm_prompt_path):
            vlm_prompt_path = "/home/datpt/projects/EOVCopilot-Demo/services/rag-service/prompts/vlm_prompts.jinja"
        vlm_prompts = load_jinja_prompts(vlm_prompt_path)

        p_entity_extract = ""
        if "entity_extraction_system_prompt" in custom_prompts:
            p_entity_extract += custom_prompts["entity_extraction_system_prompt"] + "\n"
        if "entity_extraction_user_prompt" in custom_prompts:
            p_entity_extract += custom_prompts["entity_extraction_user_prompt"]
        if "entity_extraction_examples" in custom_prompts and "{examples}" in p_entity_extract:
            p_entity_extract = p_entity_extract.replace("{examples}", custom_prompts["entity_extraction_examples"])

        _prompt_config = {
            "vlm_prompts": vlm_prompts,
            "entity_extract": p_entity_extract,
            "entity_summary": custom_prompts.get("summarize_entity_descriptions"),
            "rag_response": custom_prompts.get("rag_response"),
            "naive_rag_response": custom_prompts.get("naive_rag_response"),
            "keywords": custom_prompts.get("keywords_extraction"),
        }

        if not _chunker:
            _chunker = CustomChunker()

        self.vlm_prompts = vlm_prompts
        logger.info("Prompt config loaded.")

    async def _get_or_create_rag(self, workspace: str) -> LightRAG:
        global _rag_instances, _rag_locks

        if workspace not in _rag_locks:
            _rag_locks[workspace] = asyncio.Lock()

        async with _rag_locks[workspace]:
            if workspace in _rag_instances:
                return _rag_instances[workspace]

            logger.info(f"Initializing IndexingEngine LightRAG for workspace: {workspace}")

            rag_work_dir = os.path.join(settings.RAG_WORK_DIR, "lightrag_index", workspace)
            os.makedirs(rag_work_dir, exist_ok=True)

            storage_kwargs = {}
            if settings.STORAGE_TYPE == "postgres":
                os.environ["KV_STORAGE_CONFIG"] = json.dumps({
                    "host": settings.POSTGRES_HOST,
                    "port": settings.POSTGRES_PORT,
                    "user": settings.POSTGRES_USER,
                    "password": settings.POSTGRES_PASSWORD,
                    "database": settings.POSTGRES_DATABASE,
                })
                storage_kwargs["kv_storage"] = "PGKVStorage"
                storage_kwargs["vector_storage"] = "PGVectorStorage"
                logger.info(f"IndexingEngine: PostgreSQL storage for workspace '{workspace}'")

            if settings.ENABLE_GRAPH_STORAGE and settings.GRAPH_STORAGE_TYPE == "neo4j":
                os.environ["GRAPH_STORAGE_CONFIG"] = json.dumps({
                    "uri": settings.NEO4J_URI,
                    "username": settings.NEO4J_USERNAME,
                    "password": settings.NEO4J_PASSWORD
                })
                storage_kwargs["graph_storage"] = "Neo4JStorage"

            rag_instance = LightRAG(
                working_dir=rag_work_dir,
                workspace=workspace,
                llm_model_max_async=settings.RAG_MAX_ASYNC_JOBS,
                embedding_func_max_async=settings.RAG_MAX_ASYNC_JOBS,
                llm_model_func=llm_completion_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=settings.EMBEDDING_DIM,
                    max_token_size=settings.EMBEDDING_MAX_TOKEN_SIZE,
                    func=embedding_func
                ),
                **storage_kwargs
            )

            pc = _prompt_config or {}
            if pc.get("entity_extract"): rag_instance.entity_extract_template = pc["entity_extract"]
            if pc.get("entity_summary"): rag_instance.entity_summary_template = pc["entity_summary"]
            if pc.get("rag_response"): rag_instance.rag_response_template = pc["rag_response"]
            if pc.get("naive_rag_response"): rag_instance.naive_rag_response_template = pc["naive_rag_response"]
            if pc.get("keywords"): rag_instance.keywords_extract_template = pc["keywords"]

            rag_instance.chunking_func = lightrag_chunking_adapter

            await rag_instance.initialize_storages()
            _rag_instances[workspace] = rag_instance
            logger.info(f"IndexingEngine: LightRAG ready for workspace '{workspace}'")
            return rag_instance


    async def preprocess_content_for_chunking(self, full_content_list: List[Dict]) -> List[Dict]:
        """
        Processes raw content blocks to prepare them for chunking.
        - Enriches images with VLM captions.
        - Stores captions in self._last_caption_cache {original_img_path: caption}.
        - Converts image blocks to text paragraphs for Phase 3a (avoid double-index).
        """
        text_only_content = []
        self._last_caption_cache: Dict[str, str] = {}  # Reset mỗi lần index
        
        # Helper to get surrounding text
        def get_surrounding_text(curr_idx, all_items, window=2):
            prev_text = []
            next_text = []
            
            # Look back
            for k in range(1, window + 1):
                idx = curr_idx - k
                if idx >= 0:
                    item_type = all_items[idx].get("type", "unknown")
                    if item_type not in ["image", "table"]:
                        text_val = all_items[idx].get("text", "")
                        if text_val and len(text_val.strip()) > 0:
                            prev_text.insert(0, text_val)
            
            # Look forward
            for k in range(1, window + 1):
                idx = curr_idx + k
                if idx < len(all_items):
                    item_type = all_items[idx].get("type", "unknown")
                    if item_type not in ["image", "table"]:
                         text_val = all_items[idx].get("text", "")
                         if text_val and len(text_val.strip()) > 0:
                            next_text.append(text_val)
            
            return "\n".join(prev_text), "\n".join(next_text)

        logger.info("Phase 1.5: Inline Enrichment (Context-Aware Natural) - Parallelized...")

        semaphore = asyncio.Semaphore(5)  # Giới hạn 5 request VLM chạy đồng thời

        async def process_item(i, item):
            item_type = item.get("type", "unknown")
            
            # Case 1: Text Block -> Keep as is
            if item_type not in ["image", "table"]:
                return item
            
            # Case 2: Image Block -> Metadata Enrichment (VLM)
            elif item_type == "image":
                img_path = item.get("img_path", "")
                caption = item.get("text", "") 
                
                # Check for existing enrichment
                if caption and "[IMAGE_REF:" in caption and "Description:" in caption:
                    logger.info(f"Skipping VLM for {os.path.basename(img_path)} (Already enriched)")
                    return item
                
                # 1. Generate Description (VLM)
                if img_path:
                    local_img_path = img_path
                    if not os.path.isabs(img_path) and not os.path.exists(img_path):
                        if img_path.startswith("ocr-results/"):
                            object_path = img_path.replace("ocr-results/", "", 1)
                            local_img_path = f"/tmp/{object_path}"
                            
                            # Đoạn tải MinIO được giữ nguyên nhưng nằm ngoài semaphore cho nhẹ gánh
                            if not os.path.exists(local_img_path):
                                try:
                                    os.makedirs(os.path.dirname(local_img_path), exist_ok=True)
                                    self.minio_client.fget_object(
                                        settings.MINIO_BUCKET_OCR_RESULTS,
                                        object_path,
                                        local_img_path
                                    )
                                    logger.debug(f"Downloaded image from MinIO: {object_path}")
                                except Exception as e:
                                    logger.error(f"Failed to download image from MinIO: {object_path}: {e}")
                                    local_img_path = None
                    
                    if local_img_path and os.path.exists(local_img_path):
                        try:
                            prev_ctx, _ = get_surrounding_text(i, full_content_list)
                            context_str = ""
                            if prev_ctx: context_str += f"[Văn bản trước đó]:\n{prev_ctx}\n"
                            if not context_str: context_str = "Không có văn bản ngữ cảnh cụ thể."

                            template = self.vlm_prompts.get("inline_enrichment_narrative")
                            if template:
                                prompt = template.replace("{{ context_str }}", context_str)
                            else:
                                prompt = (
                                    f"Ngữ cảnh tài liệu:\n---\n{context_str}\n---\n"
                                    f"Phân tích hình ảnh chi tiết dựa vào ngữ cảnh này. Trả lời bằng Tiếng Việt."
                                )
                            
                            vlm_sys_prompt = self.vlm_prompts.get("vlm_system_prompt")
                            vlm_kwargs = {}
                            if vlm_sys_prompt:
                                vlm_kwargs["system_prompt"] = vlm_sys_prompt.strip()

                            # Chỉ bao bọc đoạn gọi call API căng thẳng vào Semaphore
                            async with semaphore:
                                logger.info(f"[VLM] Start processing image: {os.path.basename(local_img_path)}")
                                caption = await asyncio.wait_for(
                                    vlm_model_func(prompt, images=[local_img_path], **vlm_kwargs),
                                    timeout=480.0
                                )
                                logger.info(f"[VLM] Generated caption ({len(caption)} chars): {caption[:100]}...")

                        except asyncio.TimeoutError:
                            logger.warning(f"[VLM] Timeout 480s for {os.path.basename(local_img_path)}, skipping.")
                            caption = f"Hình ảnh minh họa: {os.path.basename(img_path)}"
                    else:
                        logger.warning(f"[VLM] Skipping - image not found: {img_path}")
                        caption = f"Hình ảnh minh họa: {os.path.basename(img_path) if img_path else 'unknown'}"

                if img_path:
                    self._last_caption_cache[img_path] = caption

                page_label = f"[Page {item.get('page_idx', '')}]" if item.get('page_idx') is not None else ""
                img_ref = f"[IMAGE_REF:{img_path}]" if img_path else ""
                return {
                    "type": "text",
                    "text": f"{page_label}{img_ref} {caption}".strip(),
                    "page_idx": item.get("page_idx"),
                }

            # Case 3: Table Block
            elif item_type == "table":
                table_body = item.get("table_body") or item.get("text") or item.get("html", "")
                return {
                    "type": "table", 
                    "text": table_body,
                    "table_caption": "Bảng dữ liệu",
                    "page_idx": item.get("page_idx"),
                    "bbox": item.get("bbox")
                }
            return None

        # Gửi toàn bộ job của các khối Data cho task manager
        tasks = [process_item(i, item) for i, item in enumerate(full_content_list)]
        results = await asyncio.gather(*tasks)
        
        # Append kết quả đã xử lý vào mảng chính xác theo thứ tự
        for res in results:
            if res is not None:
                text_only_content.append(res)
        
        return text_only_content

    def _extract_multimodal_items(self, full_content_list: List[Dict]) -> List[Dict]:
        """
        Bóc tách Image/Table từ full_content_list (tránh parse OCR JSON lần 2).
        - Giữ img_path gốc trong _ocr_img_path để match với caption_cache.
        - Download từ MinIO về /tmp/ và lưu vào img_path để RAGAnything dùng.
        """
        items = []
        try:
            for b in full_content_list:
                if b.get("type") not in ["image", "table"]:
                    continue

                item = b.copy()
                if "img_path" not in item and "image_path" in item:
                    item["img_path"] = item["image_path"]

                img_path = item.get("img_path", "")
                item["_ocr_img_path"] = img_path  # Giữ key gốc để lookup caption_cache

                if img_path:
                    if os.path.isabs(img_path) and os.path.exists(img_path):
                        pass  # Local path hợp lệ
                    elif img_path.startswith("ocr-results/"):
                        object_path = img_path.replace("ocr-results/", "", 1)
                        local_path = f"/tmp/{object_path}"
                        if os.path.exists(local_path):
                            item["img_path"] = local_path
                        else:
                            try:
                                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                                self.minio_client.fget_object(
                                    settings.MINIO_BUCKET_OCR_RESULTS,
                                    object_path,
                                    local_path
                                )
                                item["img_path"] = local_path
                                logger.debug(f"Downloaded for RAGAnything: {object_path}")
                            except Exception as e:
                                logger.warning(f"Cannot download {object_path}: {e}")
                                item["img_path"] = None
                    else:
                        logger.warning(f"Unknown img_path, skipping: {img_path[:80]}")
                        item["img_path"] = None

                items.append(item)

        except Exception as e:
            logger.error(f"Multimodal extraction error: {e}")

        valid = sum(1 for i in items if i.get("img_path"))
        logger.info(f"Extracted {len(items)} multimodal items ({valid} with valid local image)")
        return items


    async def index_document(self, ocr_data: Dict[str, Any], workspace: str = "default", job_id: str = "unknown", original_filename: str = ""):
        """
        Enhanced indexing with Hybrid Approach: Content Fusion + Manual Graph Enhancement
        
        Phases:
        1. Content Separation & Context Extraction
        2. Content Fusion (enrich multimodal with context)
        3. Standard RAG Processing (text + enriched multimodal)
        4. Manual Graph Enhancement (create custom relationships)
        """
        rag_instance = await self._get_or_create_rag(workspace)

        # Cache RAGAnything theo workspace thay vì tạo mới mỗi call
        if workspace not in _rag_anything_instances:
            _rag_anything_instances[workspace] = RAGAnything(
                lightrag=rag_instance,
                vision_model_func=vlm_model_func,
                config=RAGAnythingConfig(
                    working_dir=settings.RAG_WORK_DIR,
                    enable_image_processing=True,
                    enable_table_processing=True
                ),
                llm_model_func=llm_completion_func,
                embedding_func=embedding_func,
            )
        rag_anything = _rag_anything_instances[workspace]

        logger.info(f"Processing Ingestion Job: {job_id} | Workspace: {workspace}")
        graph_enhanced = False
        try:
            # ===== PHASE 1: Content Separation & Context Extraction =====
            logger.info("Phase 1: Extracting content and building context map...")
            
            full_content_list = self.context_builder.extract_full_content_list(ocr_data)
            context_map = self.context_builder.build_context_map(full_content_list)
            
            text_only_content = await self.preprocess_content_for_chunking(full_content_list)
            
            # Extract multimodal items từ full_content_list (tránh parse lại OCR JSON)
            mm_items = self._extract_multimodal_items(full_content_list)

            logger.info(
                f"   ✓ Extracted {len(text_only_content)} text blocks, "
                f"{len(mm_items)} multimodal items"
            )

            # Inject VLM caption từ Phase 1.5 vào mm_items
            # _last_caption_cache key = original OCR img_path
            # mm_items có _ocr_img_path = original path để lookup chính xác
            caption_cache = getattr(self, "_last_caption_cache", {})
            injected = 0
            for item in mm_items:
                ocr_path = item.get("_ocr_img_path", "")
                if ocr_path and ocr_path in caption_cache:
                    item["image_caption"] = caption_cache[ocr_path]  # field mà enrich_multimodal_items đọc
                    item["description"] = caption_cache[ocr_path]   # field RAGAnything image processor đọc
                    injected += 1
            if injected:
                logger.info(f"   ✓ Injected VLM captions into {injected}/{len(mm_items)} multimodal items")

            # ===== PHASE 2: Content Fusion =====
            enriched_mm_items = []
            if len(mm_items) > 0:
                logger.info("Phase 2: Enriching multimodal items with context...")

                enriched_mm_items = self.context_builder.enrich_multimodal_items(
                    mm_items, context_map
                )

                logger.info(f"   ✓ Enriched {len(enriched_mm_items)} multimodal items with context")

            
            # ===== PHASE 3: Standard RAG Processing =====
            logger.info("Phase 3: Inserting content into RAG system...")
            
            # 3a. Insert text-only content
            if len(text_only_content) > 0:
                text_json = json.dumps({"content": text_only_content})
                display_name = original_filename or job_id
                await rag_instance.ainsert(text_json, file_paths=display_name)
                logger.info(f"   ✓ Inserted {len(text_only_content)} text chunks into workspace '{workspace}' (filename: {display_name})")
            
            # 3b. Process enriched multimodal content
            if len(enriched_mm_items) > 0:
                logger.info(f"Processing {len(enriched_mm_items)} enriched multimodal items...")
                
                await rag_anything._process_multimodal_content(
                    multimodal_items=enriched_mm_items,
                    file_path=job_id,
                    doc_id=f"doc_{job_id}"
                )
                
                logger.info(f"   ✓ Multimodal processing complete")
            
            logger.info("Phase 4: Manual Graph Enhancement skipped (Module Disabled)")
            
            # ===== Summary =====
            logger.info(f"Ingestion Complete for {job_id} in workspace '{workspace}'")
            return {
                "status": "success",
                "job_id": job_id,
                "workspace": workspace,
                "text_chunks": len(text_only_content),
                "multimodal_items": len(mm_items),
                "context_enriched": len(enriched_mm_items),
                "graph_enhanced": graph_enhanced
            }
            
        except Exception as e:
            logger.exception(f"Indexing Failed for {job_id}")
            raise e

# Helper function for LLM retry
async def _llm_call_with_retry(client, model, messages, temperature, max_tokens, max_retries=3):
    """LLM call với retry mechanism"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.warning(f"RETRY] Attempt {attempt+1}/{max_retries} after {2**attempt}s backoff...")
                await asyncio.sleep(2 ** attempt)
            
            logger.info(f"[LLM CALL] Attempt {attempt+1}/{max_retries}...")
            
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            
            logger.info(f"LLM RESPONSE] Length: {len(content) if content else 0} chars")
            
            if not content or len(content.strip()) < 10:
                logger.warning(f"RETRY TRIGGER] Empty/short response ({len(content) if content else 0} chars)")
                if attempt < max_retries - 1:
                    logger.warning(f"Retrying after {2**(attempt+1)}s...")
                    continue
                else:
                    logger.error(f"[RETRY FAILED] Empty after {max_retries} attempts")
                    return ""
            
            
            logger.info(f"[LLM SUCCESS] Valid response on attempt {attempt+1}")
            # Detailed logging for entity extraction debugging
            if content and "entity" in content.lower():
                logger.info("\n" + "="*60)
                logger.info("ENTITY EXTRACTION DEBUG] Full LLM Response:")
                logger.info("="*60)
                
                lines = content.strip().split("\n")
                entities = []
                relations = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("entity<|#|>") or line.startswith("entity|"):
                        entities.append(line)
                    elif line.startswith("relation<|#|>") or line.startswith("relation|"):
                        relations.append(line)
                
                if entities:
                    logger.info(f"\n🔹 ENTITIES ({len(entities)}):")
                    for i, ent in enumerate(entities[:10], 1):  # Show first 10
                        logger.info(f"  [{i}] {ent}")
                    if len(entities) > 10:
                        logger.info(f"  ... and {len(entities) - 10} more entities")
                
                if relations:
                    logger.info(f"\nRELATIONS ({len(relations)}):")
                    for i, rel in enumerate(relations, 1):
                        parts = rel.split("<|#|>")
                        if len(parts) == 5:
                            logger.info(f"  [{i}] {parts[1]} → {parts[2]} | Keywords: '{parts[3]}'")
                        else:
                            logger.error(f"  [{i}] WRONG FORMAT ({len(parts)} fields): {rel}")
                
                logger.info("="*60 + "\n")
            
            # Auto-fix delimiter and format issues line-by-line
            if content and ("####" in content or "###" in content or "entity|" in content or "relation|" in content or ("<|#|>" not in content and "#" in content)):
                logger.warning("[AUTO-FIX] Detected wrong delimiter, normalizing line-by-line...")
                lines = content.strip().split("\n")
                fixed_lines = []
                relations_fixed_count = 0
                entities_fixed_count = 0
                
                for line in lines:
                    line = line.strip()
                    if not line or line == "<|COMPLETE|>":
                        fixed_lines.append(line)
                        continue
                    
                    # Skip header lines (common LLM mistakes)
                    if line.lower().startswith("type<|#|>description") or line.lower().startswith("name<|#|>type"):
                        logger.debug(f"Skipping header line: {line[:50]}")
                        continue
                    
                    # First, normalize all delimiter variants to <|#|>
                    # Handle both #### (4 chars) and ### (3 chars)
                    line = line.replace("####", "<|#|>").replace("###", "<|#|>")
                    line = line.replace("entity|", "entity<|#|>").replace("relation|", "relation<|#|>")
                    
                    if "<|#|>" not in line and line.count("#") >= 3:
                        line = line.replace("#", "<|#|>")
                    
                    # Auto-fix missing entity/relation prefix
                    # If line has 3+ delimiters but no prefix, it's likely an entity
                    if not line.startswith("entity<|#|>") and not line.startswith("relation<|#|>"):
                        parts = line.split("<|#|>")
                        if len(parts) >= 3:
                            # Likely an entity without prefix
                            line = f"entity<|#|>{line}"
                            logger.debug(f"Auto-added entity prefix to: {parts[0][:30]}")
                    
                    # Handle entity lines
                    if line.startswith("entity<|#|>"):
                        parts = line.split("<|#|>")
                        parts = [p.strip() for p in parts]
                        
                        # Entity should have 4 fields: entity, name, type, description
                        if len(parts) == 3:
                            # Missing description
                            fixed_line = f"entity<|#|>{parts[1]}<|#|>{parts[2]}<|#|>Mentioned in the document"
                            entities_fixed_count += 1
                            fixed_lines.append(fixed_line)
                        elif len(parts) == 4 and not parts[3]:
                            # Empty description
                            fixed_line = f"entity<|#|>{parts[1]}<|#|>{parts[2]}<|#|>Mentioned in the document"
                            entities_fixed_count += 1
                            fixed_lines.append(fixed_line)
                        elif len(parts) >= 4:
                            # Valid entity
                            fixed_lines.append(line)
                        else:
                            logger.warning(f"Skipping invalid entity with {len(parts)} parts")
                            continue
                    
                    # Handle relation lines
                    elif line.startswith("relation<|#|>"):
                        parts = line.split("<|#|>")
                        parts = [p.strip() for p in parts if p.strip()]
                        
                        # Relation should have 5 fields: relation, source, target, keywords, description
                        if len(parts) >= 4:
                            if len(parts) == 4:
                                # Missing keywords, need to split field 4
                                combined = parts[3]
                                keywords = "related"
                                description = combined
                                
                                # Try to extract keywords from beginning
                                if "," in combined:
                                    first_part = combined.split(",")[0].strip()
                                    if len(first_part.split()) <= 5:
                                        keywords = first_part
                                        description = combined[len(first_part)+1:].strip()
                                
                                fixed_line = f"relation<|#|>{parts[1]}<|#|>{parts[2]}<|#|>{keywords}<|#|>{description}"
                                relations_fixed_count += 1
                            elif len(parts) == 5:
                                # Already has 5 parts, just use as is
                                fixed_line = line
                            else:
                                # More than 5 parts, take first 5
                                fixed_line = f"relation<|#|>{parts[1]}<|#|>{parts[2]}<|#|>{parts[3]}<|#|>{parts[4]}"
                            
                            fixed_lines.append(fixed_line)
                        else:
                            logger.warning(f"Skipping invalid relation with {len(parts)} parts")
                            continue
                    else:
                        # Other lines, keep as is
                        fixed_lines.append(line)
                
                content = "\n".join(fixed_lines)
                logger.info(f"[AUTO-FIX] Processed {len(lines)} lines (Entities: {entities_fixed_count}, Relations: {relations_fixed_count})")
                
                # Log samples
                entities_fixed = [l for l in fixed_lines if l.startswith("entity<|#|>")]
                relations_fixed = [l for l in fixed_lines if l.startswith("relation<|#|>")]
                
                if entities_fixed:
                    logger.info(f"Entities: {len(entities_fixed)} total")
                    for i, ent in enumerate(entities_fixed[:2], 1):
                        parts = ent.split("<|#|>")
                        if len(parts) >= 3:
                            logger.info(f"  [{i}] {parts[1][:30]} ({parts[2]})")
                
                if relations_fixed:
                    logger.info(f"Relations: {len(relations_fixed)} total")
                    for i, rel in enumerate(relations_fixed[:3], 1):
                        parts = rel.split("<|#|>")
                        if len(parts) >= 5:
                            logger.info(f"  [{i}] {parts[1][:25]} → {parts[2][:25]} | KW: '{parts[3][:35]}'")
            
            return content
        except Exception as e:
            logger.error(f"[LLM ERROR] Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                logger.warning(f"Retrying after {2**(attempt+1)}s...")
                continue
            else:
                logger.error(f"[RETRY FAILED] All {max_retries} attempts exhausted")
                raise e
    return ""

default_engine = IndexingEngine()