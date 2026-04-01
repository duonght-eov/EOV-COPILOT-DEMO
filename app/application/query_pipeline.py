"""
application/query_pipeline.py
Orchestrate tiến trình trả lời câu hỏi RAG (Query Pipeline).
"""
import re
import time
import hashlib
from typing import Dict, Any

from lightrag import QueryParam
from app.utils.logger import logger
from app.utils.cache import MemoryCache

from app.infrastructure.graph.lightrag_factory import QueryRAGFactory
from app.infrastructure.llm.llm_func import response_llm_func
from app.infrastructure.llm.stream_func import stream_response_llm_func

from app.services.retrieval.consensus_retriever import ConsensusRetriever
from app.services.retrieval.multi_hop_retriever import detect_and_decompose, multi_hop_retrieve_labeled
from app.infrastructure.reranker.reranker import rerank_chunks
from app.services.processing.image_resolver import extract_image_refs_from_answer
from app.services.generation.rag_generator import format_chunks_as_sources, RAG_RESPONSE_TEMPLATE

PAGE_CITE_PATTERN = re.compile(r'\[Page\s+(\d+)\]', re.IGNORECASE)


def _clean_vlm_content(text: str) -> str:
    """
    Loại bỏ triệt để các mô tả hình ảnh rườm rà (màu sắc, hình khối, mũi tên) từ VLM.
    Regex mới hỗ trợ cả các trường hợp có nhãn trong ngoặc kép ở giữa.
    """
    if "[IMAGE_REF:" not in text:
        return text

    # Regex mạnh mẽ hơn để quét sạch: (từ chỉ nút/ô) + (nhãn tùy chọn) + (hình khối tùy chọn) + (màu sắc tùy chọn)
    fluff_patterns = [
        # Nhận diện: nút "ABC" hình thoi màu xanh / ô hình thoi màu lục / nút bắt đầu màu đỏ
        r'(?:tại\s+)?(?:nút|ô|hộp|khung|hình|khối|điểm)\s+(?:["\'][\w\s]+?["\']\s+)?(?:hình\s+)?(?:thoi|chữ\s+nhật|tròn|vuông|oval|bình\s+hành|thang|trụ|tròn)\b(?:\s+màu\s+[\w\s]+?(?=[,.]|\s|$))?',
        # Màu sắc đơn lẻ đi kèm text mô tả layout
        r'\b(?:màu\s+)?(?:xanh|đỏ|vàng|tím|cam|trắng|đen|xám|lục|lam|ngọc|hồng)(?:\s+(?:lá\s+cây|dương|biển|nhạt|đậm))?\b',
        # Các chi tiết layout/mũi tên
        r'mũi\s+tên\s+chỉ\s+hướng(?:\s+sang\s+(?:trái|phải|trên|dưới))?',
        r'đường\s+kẻ\s+(?:đứt\s+nét|liền\s+mạch|mũi\s+tên)',
        r'phía\s+(?:trên|dưới|trái|phải)\s+của\s+(?:hình|sơ\s+đồ)',
        # Loại bỏ các cụm từ mồi do VLM sinh ra
        r'được\s+lưu\s+trữ\s+tại\s+\[page\s+\d+\]',
        r'hình\s+ảnh\s+minh\s+họa\s+sơ\s+đồ\s+này'
    ]
    
    cleaned = text
    # Xoá tag IMAGE_REF khỏi text context gửi LLM (nhưng sẽ giữ ở bản gốc dùng cho Solver)
    cleaned = re.sub(r'\[IMAGE_REF:.*?\]', '', cleaned)
    
    for p in fluff_patterns:
        cleaned = re.sub(p, ' ', cleaned, flags=re.IGNORECASE)
    
    cleaned = re.sub(r'\s*,\s*,', ',', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _build_context_from_chunks(chunks, clean: bool = True) -> str:
    """
    Parse JSON chunks → text thuần có đánh số [1], [2] làm nguồn.
    clean=True: lọc nhiễu VLM (gửi cho LLM).
    clean=False: giữ nguyên (dùng cho Image Solver).
    """
    contexts = []
    for i, chunk in enumerate(chunks):
        c_text = chunk.get('content_with_weight', chunk.get('content', ''))
        if "{" in c_text and "}" in c_text and '"' in c_text:
            cleaned_parts = []
            page_matches = re.findall(r'"page_idx"\s*:\s*(\d+)', c_text)
            current_page = page_matches[0] if page_matches else "?"

            text_matches = re.findall(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', c_text)
            for t in text_matches:
                clean_t = t.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                if len(clean_t.strip()) > 5:
                    if clean:
                        clean_t = _clean_vlm_content(clean_t)
                    cleaned_parts.append(f"[Page {current_page}] {clean_t}")

            table_matches = re.findall(r'"table_body"\s*:\s*"((?:[^"\\]|\\.)*)"', c_text)
            for t in table_matches:
                clean_t = t.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                cleaned_parts.append(f"\n[Page {current_page}] [TABLE DATA]:\n{clean_t}\n")

            if cleaned_parts:
                c_text = "\n".join(cleaned_parts)

        contexts.append(f"[{i+1}] {c_text}")
    return "\n\n------\n\n".join(contexts)


def _build_mix_context(data: dict) -> str:
    """
    Build context string từ aquery_data() structured data.
    Giống format LightRAG: Entities → Relationships → Sources
    """
    parts = []

    entities = data.get("entities", [])
    if entities:
        ent_lines = []
        for e in entities:
            name = e.get("entity_name", "")
            desc = e.get("description", "")
            if name:
                ent_lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        if ent_lines:
            parts.append("-----Entities-----\n" + "\n".join(ent_lines))

    relations = data.get("relationships", [])
    if relations:
        rel_lines = []
        for r in relations:
            src = r.get("src_id", "")
            tgt = r.get("tgt_id", "")
            desc = r.get("description", "")
            if src and tgt:
                rel_lines.append(f"- {src} → {tgt}: {desc}" if desc else f"- {src} → {tgt}")
        if rel_lines:
            parts.append("-----Relationships-----\n" + "\n".join(rel_lines))

    chunks = data.get("chunks", [])
    if chunks:
        src_lines = []
        for i, c in enumerate(chunks):
            content = c.get("content", "")
            if content:
                src_lines.append(f"[{i+1}] {content}")
        if src_lines:
            parts.append("-----Sources-----\n" + "\n\n".join(src_lines))

    return "\n\n".join(parts)


class QueryPipeline:

    async def query(self, question: str, mode: str = "mix", workspace: str = "default") -> Dict[str, Any]:
        """Thực thi câu hỏi và đợi toàn bộ kết quả (Non-streaming)."""
        question_norm = question.strip().lower()
        cache_key = hashlib.md5(f"{workspace}_{mode}_{question_norm}".encode()).hexdigest()

        cached_data = MemoryCache.get_answer(cache_key)
        if cached_data:
            logger.info(f"Answer Cache HIT for question: '{question[:30]}...'")
            return cached_data

        rag = await QueryRAGFactory.get_or_create_rag(workspace)
        logger.info(f"Processing Query [Mode: {mode}, Workspace: {workspace}]: {question}")

        retrieved_chunks = []
        context_text = ""

        # --- Multi-hop detection (Đã tắt - nhảy thẳng vào single-hop) ---
        try:
            is_multi_hop, sub_queries = False, [] # await detect_and_decompose(question)
            if is_multi_hop and len(sub_queries) >= 2:
                logger.info(f"[MultiHop] Detected {len(sub_queries)} sub-queries for mode {mode}")
                retrieved_chunks, labeled_context = await multi_hop_retrieve_labeled(
                    sub_queries=sub_queries,
                    rag_instance=rag,
                    top_k_each=3,
                    mode=mode,
                )
                if not retrieved_chunks:
                    return {"answer": "Không tìm thấy thông tin phù hợp (multi-hop)."}

                from app.services.generation.rag_generator import load_prompt
                multi_hop_template = load_prompt("multi_hop_synthesis.jinja")
                _parts = multi_hop_template.split("---", 1)
                sys_prompt = _parts[0].strip() if len(_parts) > 1 else multi_hop_template
                user_msg = f"DỮ LIỆU NGỮ CẢNH ĐƯỢC CẤP:\n{labeled_context}\n\nCÂU HỎI:\n{question}"

                answer = await response_llm_func(prompt=user_msg, system_prompt=sys_prompt)

                image_refs = extract_image_refs_from_answer(retrieved_chunks, answer, labeled_context)
                result = {
                    "answer": answer,
                    "retrieved_chunks": retrieved_chunks,
                    "sources": format_chunks_as_sources(retrieved_chunks),
                    "mode": f"{mode}+multi_hop",
                    "images": image_refs,
                    "sub_queries": sub_queries,
                }
                MemoryCache.set_answer(cache_key, result)
                return result
        except Exception as e:
            logger.warning(f"MultiHop phase failed: {e}, falling back to single-hop")

        # ── SINGLE-HOP MODE ───────────────────────────────────────────────
        if mode == "consensus":
            try:
                retriever = ConsensusRetriever(rag)
                # --- Single-hop consensus (luồng gốc) ---
                retrieved_chunks = await retriever.consensus_search(
                    query=question, top_k_each_method=5, final_k=5
                )
                if not retrieved_chunks:
                    return {"answer": "Không tìm thấy thông tin phù hợp từ các nguồn dữ liệu (Consensus mode)."}

                if len(retrieved_chunks) >= 2:
                    retrieved_chunks = await rerank_chunks(question, retrieved_chunks)

                cleaned_context = _build_context_from_chunks(retrieved_chunks, clean=True)
                original_context = _build_context_from_chunks(retrieved_chunks, clean=False)

                _parts = RAG_RESPONSE_TEMPLATE.split("---", 1)
                sys_prompt = _parts[0].strip() if len(_parts) > 1 else RAG_RESPONSE_TEMPLATE
                user_msg = f"DỮ LIỆU NGỮ CẢNH:\n{cleaned_context}\n\nCÂU HỎI CỦA NGƯỜI DÙNG:\n{question}"

                answer = await response_llm_func(prompt=user_msg, system_prompt=sys_prompt)

                image_refs = extract_image_refs_from_answer(retrieved_chunks, answer, original_context)
                logger.info(f"Consensus: {len(image_refs)} ảnh được hiển thị: {image_refs}")

                result = {
                    "answer": answer,
                    "retrieved_chunks": retrieved_chunks,
                    "sources": format_chunks_as_sources(retrieved_chunks),
                    "mode": "consensus",
                    "images": image_refs,
                }
                MemoryCache.set_answer(cache_key, result)
                return result

            except Exception as e:
                logger.error(f"Consensus Query Error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return {"answer": "Đã xảy ra lỗi trong quá trình xử lý Consensus Query."}

        # ── OTHER MODES (mix / local / global / naive) ───────────────────
        effective_mode = "mix" if mode == "hybrid" else mode
        mix_chunks = []
        context_text = ""
        try:
            # Chạy song song: LightRAG cho context + ConsensusRetriever cho sources
            import asyncio as _asyncio
            async def _get_context():
                try:
                    return await rag.aquery(question, param=QueryParam(mode=effective_mode, only_need_context=True, top_k=5))
                except Exception:
                    return await rag.aquery(question, param=QueryParam(mode="naive", only_need_context=True, top_k=5))

            async def _get_chunks():
                retriever = ConsensusRetriever(rag)
                return await retriever.consensus_search(query=question, top_k_each_method=4, final_k=6)

            context_text, mix_chunks = await _asyncio.gather(_get_context(), _get_chunks(), return_exceptions=True)
            if isinstance(context_text, BaseException):
                logger.error(f"Context retrieval failed: {context_text}")
                return {"answer": "Đã xảy ra lỗi khi truy xuất dữ liệu.", "mode": mode, "images": []}
            if isinstance(mix_chunks, BaseException):
                logger.warning(f"Chunk retrieval failed: {mix_chunks}")
                mix_chunks = []
            logger.info(f"[query][mix] context={len(context_text)} chars, chunks={len(mix_chunks)}")
        except Exception as e:
            logger.error(f"All retrieval failed: {e}")
            return {"answer": "Đã xảy ra lỗi khi truy xuất dữ liệu.", "mode": mode, "images": []}

        if not context_text or len(context_text.strip()) < 10:
            logger.warning("Empty context retrieved. Returning fallback response.")
            result = {
                "answer": "Xin lỗi, tôi không tìm thấy thông tin nào liên quan trong tài liệu để trả lời câu hỏi này.",
                "mode": mode,
                "context": ""
            }
            MemoryCache.set_answer(cache_key, result)
            return result

        # Build context từ chunks để LLM dùng đúng số [1],[2] đồng bộ với sources array
        cleaned_context = _build_context_from_chunks(mix_chunks, clean=True) if mix_chunks else ""
        original_context = _build_context_from_chunks(mix_chunks, clean=False) if mix_chunks else ""

        _parts = RAG_RESPONSE_TEMPLATE.split("---", 1)
        sys_prompt = _parts[0].strip() if len(_parts) > 1 else RAG_RESPONSE_TEMPLATE
        user_msg = f"DỮ LIỆU NGỮ CẢNH:\n{cleaned_context}\n\nCÂU HỎI CỦA NGƯỜI DÙNG:\n{question}"
        try:
            answer = await response_llm_func(prompt=user_msg, system_prompt=sys_prompt)
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            answer = "Xin lỗi, đã xảy ra lỗi trong quá trình tổng hợp câu trả lời."

        image_refs = extract_image_refs_from_answer([], answer, original_context)
        sources = format_chunks_as_sources(mix_chunks) if mix_chunks else []

        result = {
            "answer": answer,
            "sources": sources,
            "mode": mode,
            "question": question,
            "images": image_refs,
        }
        MemoryCache.set_answer(cache_key, result)
        return result

    async def query_stream(self, question: str, mode: str = "mix", workspace: str = "default"):
        """
        Streaming response pipeline SSE.
        Yields: {"type": "token"|"done"|"error", ...}
        """
        question_norm = question.strip().lower()
        cache_key = hashlib.md5(f"{workspace}_{mode}_{question_norm}".encode()).hexdigest()

        cached_data = MemoryCache.get_answer(cache_key)
        if cached_data:
            logger.info(f"[StreamQuery] Answer Cache HIT for question: '{question[:30]}...'")
            yield {"type": "token", "content": cached_data["answer"]}
            yield {"type": "done", "images": cached_data["images"], "mode": mode, "sources": cached_data["sources"]}
            return

        rag = await QueryRAGFactory.get_or_create_rag(workspace)
        logger.info(f"[StreamQuery] Start [{mode}]: {question[:80]}")

        retrieved_chunks = []
        context_text = ""
        t_total = time.perf_counter()

        try:
            # --- Multi-hop detection (Đã tắt - nhảy thẳng vào single-hop) ---
            t0 = time.perf_counter()
            is_multi_hop, sub_queries = False, [] # await detect_and_decompose(question)

            if is_multi_hop and len(sub_queries) >= 2:
                logger.info(f"[StreamQuery][MultiHop] {len(sub_queries)} sub-queries detected for mode {mode}")
                retrieved_chunks, labeled_context = await multi_hop_retrieve_labeled(
                    sub_queries=sub_queries,
                    rag_instance=rag,
                    top_k_each=3,
                    mode=mode,
                )
                logger.info(
                    f"[StreamQuery][TIMING] multi_hop_retrieve={1000*(time.perf_counter()-t0):.0f}ms, "
                    f"chunks={len(retrieved_chunks)}"
                )
                if not retrieved_chunks:
                    yield {"type": "error", "content": "Không tìm thấy thông tin phù hợp."}
                    return

                from app.services.generation.rag_generator import load_prompt
                multi_hop_template = load_prompt("multi_hop_synthesis.jinja")
                _parts = multi_hop_template.split("---", 1)
                sys_prompt = _parts[0].strip() if len(_parts) > 1 else multi_hop_template
                user_msg = f"DỮ LIỆU NGỮ CẢNH ĐƯỢC CẤP:\n{labeled_context}\n\nCÂU HỎI:\n{question}"
                context_text = labeled_context

            elif mode == "consensus":
                # --- Single-hop consensus ---
                retriever = ConsensusRetriever(rag)
                retrieved_chunks = await retriever.consensus_search(
                    query=question, top_k_each_method=5, final_k=5
                )
                logger.info(
                    f"[StreamQuery][TIMING] consensus_search={1000*(time.perf_counter()-t0):.0f}ms, "
                    f"chunks={len(retrieved_chunks)}"
                )

                if not retrieved_chunks:
                    yield {"type": "error", "content": "Không tìm thấy thông tin phù hợp."}
                    return

                if len(retrieved_chunks) >= 2:
                    t0 = time.perf_counter()
                    retrieved_chunks = await rerank_chunks(question, retrieved_chunks)
                    logger.info(f"[StreamQuery][TIMING] rerank={1000*(time.perf_counter()-t0):.0f}ms")

                cleaned_context = _build_context_from_chunks(retrieved_chunks, clean=True)
                original_context = _build_context_from_chunks(retrieved_chunks, clean=False)
                
                _parts = RAG_RESPONSE_TEMPLATE.split("---", 1)
                sys_prompt = _parts[0].strip() if len(_parts) > 1 else RAG_RESPONSE_TEMPLATE
                user_msg = f"DỮ LIỆU NGỮ CẢNH:\n{cleaned_context}\n\nCÂU HỎI CỦA NGƯỜI DÙNG:\n{question}"

            else:
                effective_mode = "mix" if mode == "hybrid" else mode
                try:
                    import asyncio as _asyncio
                    async def _get_ctx():
                        try:
                            return await rag.aquery(question, param=QueryParam(mode=effective_mode, only_need_context=True, top_k=5))
                        except Exception:
                            return ""

                    async def _get_cks():
                        retriever = ConsensusRetriever(rag)
                        return await retriever.consensus_search(query=question, top_k_each_method=4, final_k=6)

                    context_text, retrieved_chunks = await _asyncio.gather(_get_ctx(), _get_cks(), return_exceptions=True)
                    if isinstance(context_text, BaseException) or not context_text:
                        context_text = ""
                    if isinstance(retrieved_chunks, BaseException):
                        retrieved_chunks = []
                    logger.info(f"[StreamQuery][mix] context={len(context_text)} chars, chunks={len(retrieved_chunks)}")
                except Exception as e:
                    logger.error(f"[StreamQuery] Retrieval failed: {e}")
                    context_text = ""
                    retrieved_chunks = []

                if not context_text or len(context_text.strip()) < 10:
                    yield {"type": "error", "content": "Không tìm thấy thông tin liên quan."}
                    return

                # Build context từ chunks để [1],[2] đồng bộ với sources
                if retrieved_chunks:
                    cleaned_context = _build_context_from_chunks(retrieved_chunks, clean=True)
                    original_context = _build_context_from_chunks(retrieved_chunks, clean=False)
                    logger.info(f"[StreamQuery][mix] rebuilt context from {len(retrieved_chunks)} chunks")

                _parts = RAG_RESPONSE_TEMPLATE.split("---", 1)
                sys_prompt = _parts[0].strip() if len(_parts) > 1 else RAG_RESPONSE_TEMPLATE
                user_msg = f"DỮ LIỆU NGỮ CẢNH:\n{cleaned_context}\n\nCÂU HỎI CỦA NGƯỜI DÙNG:\n{question}"

        except Exception as e:
            logger.error(f"[StreamQuery] Retrieval error: {e}")
            yield {"type": "error", "content": "Lỗi truy xuất dữ liệu."}
            return

        # Stream LLM tokens
        full_answer = ""
        t0 = time.perf_counter()
        async for token in stream_response_llm_func(prompt=user_msg, system_prompt=sys_prompt):
            full_answer += token
            yield {"type": "token", "content": token}
        llm_ms = 1000 * (time.perf_counter() - t0)

        # Tính tok/s
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            output_tokens = len(enc.encode(full_answer))
        except Exception:
            output_tokens = max(1, len(full_answer) // 4)

        tok_per_sec = output_tokens / max(llm_ms / 1000, 0.001)
        total_ms = 1000 * (time.perf_counter() - t_total)
        logger.info(
            f"[StreamQuery][TIMING] llm_generate={llm_ms:.0f}ms "
            f"| {output_tokens} tokens @ {tok_per_sec:.1f} tok/s "
            f"| total={total_ms:.0f}ms"
        )

        image_refs = extract_image_refs_from_answer(
            retrieved_chunks, full_answer, original_context
        )
        formatted_sources = format_chunks_as_sources(retrieved_chunks)

        logger.info(f"[StreamQuery] Done: {len(full_answer)} chars, {len(image_refs)} images")

        MemoryCache.set_answer(cache_key, {
            "answer": full_answer,
            "images": image_refs,
            "sources": formatted_sources,
        })

        yield {"type": "done", "images": image_refs, "mode": mode, "sources": formatted_sources}


# Global Pipeline instance
query_pipeline = QueryPipeline()
