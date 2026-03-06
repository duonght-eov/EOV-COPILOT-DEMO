import os
import time
import ujson as json
import httpx
import numpy as np
import re
from typing import Dict, Any, List, Optional
import hashlib

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from app.config import settings
from app.utils.logger import logger
from app.utils.http_client import get_embedding_client
from app.services.indexing_engine import query_llm_func, stream_llm_func, stream_response_llm_func
from app.services.consensus_retriever import ConsensusRetriever
from app.services.reranker import rerank_chunks


# Cấu trúc cache câu trả lời: {md5_key: (response_data, expire_ts)}
_ANSWER_CACHE: Dict[str, tuple] = {}
_ANSWER_CACHE_TTL = 3600  # Lưu cache trong 60 phút

IMAGE_REF_PATTERN = re.compile(r'\[IMAGE_REF:\s*([^\]]+)\]')
PAGE_CITE_PATTERN = re.compile(r'\[Page\s+(\d+)\]', re.IGNORECASE)


def _parse_obj_key(raw_path: str) -> str:
    """Chuẩn hóa path về object key cho MinIO bucket ocr-results."""
    if raw_path.startswith('ocr-results/'):
        return raw_path[len('ocr-results/'):]
    return raw_path


# Tầng 1: LLM copy trực tiếp từ mô tả VLM (n-gram match)
# Cần 12 từ liên tiếp để tránh false positive từ cụm chung như "hệ thống điện trong tòa nhà"
_IMG_NGRAM_SIZE = 10

# Tầng 2: LLM tham chiếu ảnh theo cách tự nhiên ("hình ảnh", "sơ đồ", "biểu đồ"...)
# Dùng khi LLM không copy text VLM nhưng rõ ràng đang nói về ảnh
_VISUAL_KEYWORDS = re.compile(
    r'(hình\s*ảnh|sơ\s*đồ|biểu\s*đồ|hình\s*vẽ|ảnh\s*minh\s*họa|minh\s*họa|hình\s*dưới|bảng\s*sau)',
    re.IGNORECASE
)


def _normalize(text: str) -> str:
    """Chuẩn hóa text để so sánh: lowercase, loại bỏ ký tự đặc biệt."""
    text = text.lower()
    text = re.sub(r'[\[\]\(\)\{\}"\',.:;!?\-_]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _extract_img_description(content: str, img_ref_match: re.Match) -> str:
    """
    Trích xuất phần text mô tả ảnh nằm ngay sau [IMAGE_REF:...] trong chunk content.
    Trả về chuỗi rỗng nếu không tìm thấy mô tả.
    """
    start = img_ref_match.end()
    # Lấy tối đa 1000 ký tự sau IMAGE_REF làm mô tả
    desc_raw = content[start:start + 1000].strip()
    # Dừng ở IMAGE_REF tiếp theo hoặc [Page X] tiếp theo nếu có
    stop = re.search(r'\[IMAGE_REF:|\[Page\s+\d+\]', desc_raw)
    if stop:
        desc_raw = desc_raw[:stop.start()].strip()
    return desc_raw


def _image_desc_used_in_answer(description: str, answer: str, ngram_size: int = _IMG_NGRAM_SIZE) -> bool:
    """
    Kiểm tra xem answer có chứa ít nhất 1 cụm từ (phrase) từ description không.
    Dùng sliding-window n-gram trên words sau khi normalize.
    """
    if not description or not answer:
        return False

    norm_desc = _normalize(description)
    norm_ans = _normalize(answer)

    words_desc = norm_desc.split()
    if len(words_desc) < ngram_size:
        # Mô tả quá ngắn → kiểm tra substring trực tiếp
        return norm_desc in norm_ans

    # Sliding window: mỗi phrase gồm ngram_size từ liên tiếp
    for i in range(len(words_desc) - ngram_size + 1):
        phrase = ' '.join(words_desc[i:i + ngram_size])
        if phrase in norm_ans:
            return True
    return False


def _answer_visually_references_page(answer: str, page_num: int) -> bool:
    """
    Kiểm tra xem LLM có dùng từ khóa trực quan ("hình ảnh", "sơ đồ"...)
    kết hợp với cite đúng page đó không.
    """
    norm_ans = answer.lower()
    if not _VISUAL_KEYWORDS.search(norm_ans):
        return False
    # Kiểm tra page được nhắc đến trong câu trả lời
    cited = {int(m.group(1)) for m in PAGE_CITE_PATTERN.finditer(answer)}
    return page_num in cited


def extract_image_refs_from_answer(
    chunks: List[Dict], answer: str, context_text: str = ""
) -> List[str]:
    """
    Trả về danh sách IMAGE_REF của các ảnh cần hiển thị, theo logic 2 tầng:

    Tầng 1 (ưu tiên): LLM copy ≥12 từ liên tiếp từ mô tả VLM → hiển thị ảnh.
    Tầng 2 (fallback): LLM dùng từ khóa trực quan ("hình ảnh", "sơ đồ"...)
                       VÀ cite đúng page có ảnh đó → hiển thị ảnh.
    """
    seen = set()
    refs = []

    sources = [(chunk.get('content', '') or '') for chunk in chunks]
    if context_text:
        sources.append(context_text)

    for content in sources:
        for img_match in IMAGE_REF_PATTERN.finditer(content):
            obj_key = _parse_obj_key(img_match.group(1).strip())
            if not obj_key or obj_key in seen:
                continue

            description = _extract_img_description(content, img_match)

            # Tầng 1: n-gram match với mô tả VLM
            if _image_desc_used_in_answer(description, answer):
                seen.add(obj_key)
                refs.append(obj_key)
                logger.info(f"[ImageFilter] ✅ [T1-ngram] {obj_key.split('/')[-1]}")
                continue

            # Tầng 2: LLM dùng từ khóa trực quan + cite đúng page
            pre_text = content[max(0, img_match.start() - 30):img_match.start()]
            page_m = PAGE_CITE_PATTERN.search(pre_text)
            if page_m:
                img_page = int(page_m.group(1))
                if _answer_visually_references_page(answer, img_page):
                    seen.add(obj_key)
                    refs.append(obj_key)
                    logger.info(f"[ImageFilter] ✅ [T2-visual] {obj_key.split('/')[-1]} (page {img_page})")
                    continue

            logger.debug(f"[ImageFilter] ❌ Bỏ qua: {obj_key.split('/')[-1]}")

    return refs


def load_prompt(filename: str) -> str:
    """Load prompt template from prompts directory."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        service_root = os.path.dirname(os.path.dirname(current_dir))
        prompt_path = os.path.join(service_root, "prompts", filename)
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load prompt '{filename}': {e}")
        return "{context_data}\n\n{question}" # Fallback minimal prompt

RAG_RESPONSE_TEMPLATE = load_prompt("response_system_prompt.jinja")
NAIVE_RAG_RESPONSE_TEMPLATE = RAG_RESPONSE_TEMPLATE


def _format_chunks_as_sources(chunks: List[Dict]) -> List[Dict]:
    """
    Convert LightRAG chunks → AnythingLLM Citations format.
    Citations component expects: {id, title, text, chunkSource, score}
    """
    import uuid as _uuid

    def _get_page(chunk: Dict) -> Optional[int]:
        meta = chunk.get("metadata", {})
        if isinstance(meta, dict):
            try:
                return int(meta["page_idx"]) + 1  # 0-indexed → 1-indexed
            except (KeyError, ValueError, TypeError):
                pass
        # Fallback: parse [Page X] from content
        m = PAGE_CITE_PATTERN.search(chunk.get("content", ""))
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        return None

    def _get_doc_name(chunk: Dict) -> str:
        # LightRAG lưu original_filename vào file_path qua ainsert(file_paths=...)
        # Ưu tiên file_path → full_doc_id → fallback
        for key in ("file_path", "full_doc_id", "doc_id", "source"):
            val = chunk.get(key, "")
            if not val:
                continue
            # Bỏ path prefix
            name = val.split("/")[-1].split("\\")[-1]
            # Bỏ UUID-like prefix (e.g. "abc12345_filename.pdf" → "filename.pdf")
            # Nhận dạng: prefix dài ≥ 8 ký tự hex + dấu "_"
            if "_" in name:
                parts = name.split("_", 1)
                import re as _re
                if _re.match(r'^[a-f0-9]{8,}$', parts[0]):
                    name = parts[1]
            # Bỏ phần mở rộng .pdf, .docx, ... để title gọn hơn
            if "." in name:
                name = name.rsplit(".", 1)[0]
            return name.replace("_", " ").replace("-", " ")
        return "Tài liệu"

    sources = []
    for chunk in chunks:
        page = _get_page(chunk)
        doc_name = _get_doc_name(chunk)
        title = f"{doc_name} — Trang {page}" if page else doc_name
        content = chunk.get("content", chunk.get("content_with_weight", ""))
        # Strip [Page X] prefix from display text
        clean_content = PAGE_CITE_PATTERN.sub("", content).strip()

        sources.append({
            "id": chunk.get("id", str(_uuid.uuid4())),
            "title": title,
            "text": clean_content,
            "chunkSource": "",
            "score": chunk.get("total_score", chunk.get("score", None)),
        })

    return sources

async def query_embedding_func(texts: list[str]) -> np.ndarray:
    """
    Embedding function cho QUERY — dùng singleton persistent HTTP client.
    Tránh TCP handshake overhead mỗi lần gọi.
    """
    if not texts:
        return np.array([])

    results = []
    client = get_embedding_client()
    base_url = settings.EMBEDDING_SERVICE_URL.rstrip("/")
    url = f"{base_url}/api/v1/embed/text"

    for text in texts:
        try:
            payload = {
                "text": text,
                "model": settings.EMBEDDING_MODEL_NAME,
            }
            logger.info(f"Embedding Query: '{text[:30]}...' -> {url}")
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            vector = data.get("vector") or data.get("embedding")
            if vector:
                results.append(vector)
            else:
                logger.warning(f"Empty embedding for query: {text[:20]}...")
                results.append([0.0] * settings.EMBEDDING_DIM)
        except Exception as e:
            logger.error(f"Query Embedding Error: {e}")
            results.append([0.0] * settings.EMBEDDING_DIM)

    return np.array(results)

class QueryEngine:
    def __init__(self):
        self.rags = {}
        self.rag_locks = {}

    async def _get_or_create_rag(self, workspace: str) -> LightRAG:
        import asyncio
        if workspace not in self.rag_locks:
            self.rag_locks[workspace] = asyncio.Lock()
            
        async with self.rag_locks[workspace]:
            if workspace in self.rags:
                return self.rags[workspace]

            logger.info(f"Initializing Query Engine (Read-Only) for workspace: {workspace}...")
            
            # --- Storage Configuration (Must match IndexingEngine) ---
            storage_kwargs = {}
            rag_work_dir = os.path.join(settings.RAG_WORK_DIR, "lightrag_index", workspace)
            
            # Postgres Configuration
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
                logger.info("QueryEngine: Connected to PostgreSQL")
            
            # Neo4j Configuration
            if settings.ENABLE_GRAPH_STORAGE and settings.GRAPH_STORAGE_TYPE == "neo4j":
                os.environ["GRAPH_STORAGE_CONFIG"] = json.dumps({
                    "uri": settings.NEO4J_URI,
                    "username": settings.NEO4J_USERNAME,
                    "password": settings.NEO4J_PASSWORD
                })
                storage_kwargs["graph_storage"] = "Neo4JStorage"
                logger.info("QueryEngine: Connected to Neo4j Graph")

            # --- LightRAG Initialization ---
            rag = LightRAG(
                working_dir=rag_work_dir,
                workspace=workspace,
                llm_model_max_async=settings.RAG_MAX_ASYNC_JOBS,
                embedding_func_max_async=settings.RAG_MAX_ASYNC_JOBS,
                llm_model_func=query_llm_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=settings.EMBEDDING_DIM,
                    max_token_size=settings.EMBEDDING_MAX_TOKEN_SIZE,
                    func=query_embedding_func
                ),
                **storage_kwargs
            )

            # --- INJECT CUSTOM PROMPTS ---
            # Overwrite default LightRAG prompts with Vietnamese strict prompts
            
            # 1. RAG Response Prompts
            rag.rag_response_template = RAG_RESPONSE_TEMPLATE
            rag.naive_rag_response_template = NAIVE_RAG_RESPONSE_TEMPLATE
            
            # 2. Keywords Extraction Prompt (Critical for Local Search)
            from lightrag.prompt import PROMPTS
            keywords_extraction_template = load_prompt("keywords_extraction.jinja")
            if keywords_extraction_template and len(keywords_extraction_template) > 50:
                PROMPTS["keywords_extraction"] = keywords_extraction_template
                logger.info("QueryEngine: Injected Custom Vietnamese Keywords Extraction Prompt")
            else:
                logger.warning("QueryEngine: Failed to load Keywords Extraction Prompt (using default)")

            logger.info("QueryEngine: Injected Custom Vietnamese Prompts (Strict Citation Mode)")

            # Initialize storages
            await rag.initialize_storages()
            self.rags[workspace] = rag
            logger.info(f"QueryEngine: Query Engine Fully Initialized for {workspace}")
            return rag

    async def query(self, question: str, mode: str = "mix", workspace: str = "default") -> Dict[str, Any]:
        """
        Execute RAG Query with Manual Control for Citations & Anti-Hallucination.
        
        Available Modes:
        - 'naive': Simple vector search only
        - 'local': Entity-based local search
        - 'global': Community-based global search
        - 'mix': Hybrid of local + global (default)
        - 'consensus': Intersection of naive + local (high precision)
        """
        question_norm = question.strip().lower()
        cache_key = hashlib.md5(f"{workspace}_{mode}_{question_norm}".encode()).hexdigest()

        if cache_key in _ANSWER_CACHE:
            cached_data, expire_ts = _ANSWER_CACHE[cache_key]
            if time.time() < expire_ts:
                logger.info(f"Answer Cache HIT for question: '{question[:30]}...'")
                return cached_data

        rag = await self._get_or_create_rag(workspace)

        logger.info(f"Processing Query [Mode: {mode}, Workspace: {workspace}]: {question}")

        # ===== MODE: CONSENSUS =====
        if mode == "consensus":
            try:
                # 1. Init Consensus Module
                consensus_retriever = ConsensusRetriever(rag)

                # 2. Execute Consensus Search
                retrieved_chunks = await consensus_retriever.consensus_search(
                    query=question,
                    top_k_each_method=5,
                    final_k=5
                )
                
                if not retrieved_chunks:
                    return {"answer": "Không tìm thấy thông tin phù hợp từ các nguồn dữ liệu (Consensus mode)."}

                # 3. Rerank chunks bằng BGE cross-encoder
                if len(retrieved_chunks) >= 2:
                    retrieved_chunks = await rerank_chunks(question, retrieved_chunks)

                # 4. Context Builder
                contexts = []
                for chunk in retrieved_chunks:
                    c_text = chunk.get('content_with_weight', chunk.get('content', ''))
                    # The chunk is likely a fragment of a JSON list. We extract useful fields.
                    if "{" in c_text and "}" in c_text and '"' in c_text:
                        cleaned_parts = []
                        
                        # Extract Page Index if present (Best effort)
                        page_matches = re.findall(r'"page_idx"\s*:\s*(\d+)', c_text)
                        current_page = page_matches[0] if page_matches else "?"
                        
                        # 1. Extract Text Paragraphs
                        # Match "text": "..." content
                        text_matches = re.findall(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', c_text)
                        for t in text_matches:
                            clean_t = t.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                            if len(clean_t.strip()) > 5: # Skip empty/noise
                                cleaned_parts.append(f"[Page {current_page}] {clean_t}")
                                
                        # 2. Extract Tables
                        # Match "table_body": "..." content
                        table_matches = re.findall(r'"table_body"\s*:\s*"((?:[^"\\]|\\.)*)"', c_text)
                        for t in table_matches:
                            clean_t = t.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                            cleaned_parts.append(f"\n[Page {current_page}] [TABLE DATA]:\n{clean_t}\n")
                            
                        # If we extracted meaningful content, use it. Otherwise fallback to raw (maybe it's not JSON)
                        if cleaned_parts:
                            c_text = "\n".join(cleaned_parts)
                        else:
                            # Fallback: If regex found nothing but it looks like JSON, it might be an image chunk or empty
                            # Just keep it as is, or strip lines? Let's keep distinct JSON chars out
                            pass 
                            
                    contexts.append(c_text)
                
                context_text = "\n\n------\n\n".join(contexts)
                
                # --- DEBUG: Log Final Context Preview ---
                logger.info(f"Final Context sent to LLM ({len(context_text)} chars):\n{context_text[:500]}...\n[...]\n{context_text[-200:]}")

                
                # 4. Generate Answer
                prompt = rag.rag_response_template.format(
                    question=question,
                    context_data=context_text
                )
                
                answer = await query_llm_func(prompt)

                # Hiển thị ảnh chỉ khi LLM thực sự dùng mô tả ảnh trong câu trả lời.
                image_refs = extract_image_refs_from_answer(
                    retrieved_chunks, answer, context_text
                )
                logger.info(f"Consensus: {len(image_refs)} ảnh được hiển thị: {image_refs}")

                result = {
                    "answer": answer,
                    "retrieved_chunks": retrieved_chunks,
                    "sources": _format_chunks_as_sources(retrieved_chunks),
                    "mode": "consensus",
                    "images": image_refs
                }
                _ANSWER_CACHE[cache_key] = (result, time.time() + _ANSWER_CACHE_TTL)
                return result

            except Exception as e:
                logger.error(f"Consensus Query Error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return {"answer": "Đã xảy ra lỗi trong quá trình xử lý Consensus Query."}

        # Mode 'hybrid' = alias của 'mix' trong LightRAG
        effective_mode = "mix" if mode == "hybrid" else mode
        query_param = QueryParam(mode=effective_mode, only_need_context=True, top_k=5)

        try:
            context_text = await rag.aquery(question, param=query_param)
        except Exception as e:
            err_str = str(e)
            # Keyword extraction thất bại (LLM output JSON sai format) → fallback naive
            if "and end with" in err_str or "Keyword" in err_str or "json" in err_str.lower():
                logger.warning(f"Keyword extraction failed ({err_str[:80]}), falling back to naive mode")
                try:
                    fallback_param = QueryParam(mode="naive", only_need_context=True, top_k=5)
                    context_text = await rag.aquery(question, param=fallback_param)
                    effective_mode = "naive (fallback)"
                except Exception as e2:
                    logger.error(f"Naive fallback also failed: {e2}")
                    return {"answer": "Đã xảy ra lỗi khi truy xuất dữ liệu.", "mode": mode, "images": []}
            else:
                logger.error(f"Retrieval failed: {e}")
                return {"answer": "Đã xảy ra lỗi khi truy xuất dữ liệu.", "mode": mode, "images": []}

        # 2. VALIDATION (Anti-Hallucination Level 1)
        # If context is empty or too short, don't even ask LLM.
        if not context_text or len(context_text.strip()) < 10:
            logger.warning("Empty context retrieved. Returning fallback response.")
            result = {
                "answer": "Xin lỗi, tôi không tìm thấy thông tin nào liên quan trong tài liệu để trả lời câu hỏi này.",
                "mode": mode,
                "context": ""
            }
            _ANSWER_CACHE[cache_key] = (result, time.time() + _ANSWER_CACHE_TTL)
            return result

        # 3. GENERATION (Strict Prompting)
        # We manually construct the prompt using our STRICT template logic.
        prompt = RAG_RESPONSE_TEMPLATE.format(
            context_data=context_text,
            question=question
        )

        try:
            # Call LLM directly using the function registered in LightRAG
            answer = await rag.llm_model_func(prompt, system_prompt=None)
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            answer = "Xin lỗi, đã xảy ra lỗi trong quá trình tổng hợp câu trả lời."

        # Hiển thị ảnh chỉ khi LLM dùng mô tả ảnh trong câu trả lời.
        image_refs = extract_image_refs_from_answer([], answer, context_text)

        result = {
            "answer": answer,
            "sources": [],
            "mode": mode,
            "question": question,
            "images": image_refs,
        }
        _ANSWER_CACHE[cache_key] = (result, time.time() + _ANSWER_CACHE_TTL)
        return result


    async def query_stream(self, question: str, mode: str = "consensus", workspace: str = "default"):
        """
        Streaming version của query().
        Yields SSE-style dicts:
          {"type": "token", "content": "<text>"}
          {"type": "done",  "images": [...], "mode": "...", "sources": [...]}
          {"type": "error", "content": "<msg>"}
        """
        question_norm = question.strip().lower()
        cache_key = hashlib.md5(f"{workspace}_{mode}_{question_norm}".encode()).hexdigest()

        if cache_key in _ANSWER_CACHE:
            cached_data, expire_ts = _ANSWER_CACHE[cache_key]
            if time.time() < expire_ts:
                logger.info(f"[StreamQuery] Answer Cache HIT for question: '{question[:30]}...'")
                # Trả về câu trả lời đã lưu trong 1 token duy nhất (fake streaming rất nhanh)
                yield {"type": "token", "content": cached_data["answer"]}
                yield {"type": "done", "images": cached_data["images"], "mode": mode, "sources": cached_data["sources"]}
                return

        rag = await self._get_or_create_rag(workspace)
        logger.info(f"[StreamQuery] Start [{mode}]: {question[:80]}")

        retrieved_chunks = []
        context_text = ""

        try:
            if mode == "consensus":
                t_total = time.perf_counter()

                t0 = time.perf_counter()
                consensus_retriever = ConsensusRetriever(rag)
                retrieved_chunks = await consensus_retriever.consensus_search(
                    query=question, top_k_each_method=5, final_k=5
                )
                logger.info(f"[StreamQuery][TIMING] consensus_search={1000*(time.perf_counter()-t0):.0f}ms, chunks={len(retrieved_chunks)}")

                if not retrieved_chunks:
                    yield {"type": "error", "content": "Không tìm thấy thông tin phù hợp."}
                    return

                if len(retrieved_chunks) >= 2:
                    t0 = time.perf_counter()
                    retrieved_chunks = await rerank_chunks(question, retrieved_chunks)
                    logger.info(f"[StreamQuery][TIMING] rerank={1000*(time.perf_counter()-t0):.0f}ms")

                contexts = []
                for chunk in retrieved_chunks:
                    c_text = chunk.get("content_with_weight", chunk.get("content", ""))
                    if "{" in c_text and "}" in c_text and '"' in c_text:
                        cleaned_parts = []
                        page_matches = re.findall(r'"page_idx"\s*:\s*(\d+)', c_text)
                        current_page = page_matches[0] if page_matches else "?"
                        text_matches = re.findall(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', c_text)
                        for t in text_matches:
                            clean_t = t.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                            if len(clean_t.strip()) > 5:
                                cleaned_parts.append(f"[Page {current_page}] {clean_t}")
                        table_matches = re.findall(r'"table_body"\s*:\s*"((?:[^"\\]|\\.)*)"', c_text)
                        for t in table_matches:
                            clean_t = t.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                            cleaned_parts.append(f"\n[Page {current_page}] [TABLE DATA]:\n{clean_t}\n")
                        if cleaned_parts:
                            c_text = "\n".join(cleaned_parts)
                    contexts.append(c_text)
                context_text = "\n\n------\n\n".join(contexts)
                prompt = rag.rag_response_template.format(question=question, context_data=context_text)

            else:
                effective_mode = "mix" if mode == "hybrid" else mode
                query_param = QueryParam(mode=effective_mode, only_need_context=True, top_k=5)
                try:
                    context_text = await rag.aquery(question, param=query_param)
                except Exception:
                    fallback_param = QueryParam(mode="naive", only_need_context=True, top_k=5)
                    context_text = await rag.aquery(question, param=fallback_param)

                if not context_text or len(context_text.strip()) < 10:
                    yield {"type": "error", "content": "Không tìm thấy thông tin liên quan."}
                    return

                prompt = RAG_RESPONSE_TEMPLATE.format(context_data=context_text, question=question)

        except Exception as e:
            logger.error(f"[StreamQuery] Retrieval error: {e}")
            yield {"type": "error", "content": "Lỗi truy xuất dữ liệu."}
            return

        # Stream LLM tokens
        full_answer = ""
        t0 = time.perf_counter()
        async for token in stream_response_llm_func(prompt):
            full_answer += token
            yield {"type": "token", "content": token}
        llm_ms = 1000 * (time.perf_counter() - t0)

        # Đo tốc độ tok/s bằng tiktoken
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            output_tokens = len(enc.encode(full_answer))
        except Exception:
            output_tokens = max(1, len(full_answer) // 4)

        tok_per_sec = output_tokens / max(llm_ms / 1000, 0.001)

        # Sau khi có full answer, tính image refs
        if mode == "consensus":
            image_refs = extract_image_refs_from_answer(retrieved_chunks, full_answer, context_text)
        else:
            image_refs = extract_image_refs_from_answer([], full_answer, context_text)

        total_ms = 1000 * (time.perf_counter() - t_total) if mode == "consensus" else llm_ms
        logger.info(
            f"[StreamQuery][TIMING] llm_generate={llm_ms:.0f}ms "
            f"| {output_tokens} tokens @ {tok_per_sec:.1f} tok/s "
            f"| total={total_ms:.0f}ms"
        )
        logger.info(f"[StreamQuery] Done: {len(full_answer)} chars, {len(image_refs)} images")
        formatted_sources = _format_chunks_as_sources(retrieved_chunks) if mode == "consensus" else []
        
        # Lưu vào cache
        _ANSWER_CACHE[cache_key] = (
            {
                "answer": full_answer,
                "images": image_refs,
                "sources": formatted_sources
            },
            time.time() + _ANSWER_CACHE_TTL
        )
        
        yield {"type": "done", "images": image_refs, "mode": mode, "sources": formatted_sources}


# Singleton Instance
query_engine = QueryEngine()