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
from app.infrastructure.reranker.bge_reranker import rerank_chunks
from app.services.processing.image_resolver import extract_image_refs_from_answer
from app.services.generation.rag_generator import format_chunks_as_sources, RAG_RESPONSE_TEMPLATE

PAGE_CITE_PATTERN = re.compile(r'\[Page\s+(\d+)\]', re.IGNORECASE)


def _build_context_from_chunks(chunks) -> str:
    """
    Parse JSON chunks → text thuần có [Page X] prefix.
    Giống logic trong query_engine.py phiên bản gốc.
    """
    contexts = []
    for chunk in chunks:
        c_text = chunk.get('content_with_weight', chunk.get('content', ''))
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
    return "\n\n------\n\n".join(contexts)


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

        # ── CONSENSUS MODE ───────────────────────────────────────────────
        if mode == "consensus":
            try:
                retriever = ConsensusRetriever(rag)
                retrieved_chunks = await retriever.consensus_search(
                    query=question, top_k_each_method=5, final_k=5
                )
                if not retrieved_chunks:
                    return {"answer": "Không tìm thấy thông tin phù hợp từ các nguồn dữ liệu (Consensus mode)."}

                if len(retrieved_chunks) >= 2:
                    retrieved_chunks = await rerank_chunks(question, retrieved_chunks)

                context_text = _build_context_from_chunks(retrieved_chunks)

                tier_summary = " | ".join(
                    f"[{c.get('consensus_source','?')}] {c.get('total_score', 0):.3f}"
                    for c in retrieved_chunks
                )
                logger.info(f"Consensus Tiers: {tier_summary}")
                logger.info(
                    f"Final Context sent to LLM ({len(context_text)} chars):\n"
                    f"{context_text[:500]}...\n[...]\n{context_text[-200:]}"
                )

                prompt = RAG_RESPONSE_TEMPLATE.format(question=question, context_data=context_text)
                answer = await response_llm_func(prompt)

                image_refs = extract_image_refs_from_answer(retrieved_chunks, answer, context_text)
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
        query_param = QueryParam(mode=effective_mode, only_need_context=True, top_k=5)
        try:
            context_text = await rag.aquery(question, param=query_param)
        except Exception as e:
            err_str = str(e)
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

        if not context_text or len(context_text.strip()) < 10:
            logger.warning("Empty context retrieved. Returning fallback response.")
            result = {
                "answer": "Xin lỗi, tôi không tìm thấy thông tin nào liên quan trong tài liệu để trả lời câu hỏi này.",
                "mode": mode,
                "context": ""
            }
            MemoryCache.set_answer(cache_key, result)
            return result

        prompt = RAG_RESPONSE_TEMPLATE.format(context_data=context_text, question=question)
        try:
            answer = await rag.llm_model_func(prompt, system_prompt=None)
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            answer = "Xin lỗi, đã xảy ra lỗi trong quá trình tổng hợp câu trả lời."

        image_refs = extract_image_refs_from_answer([], answer, context_text)

        result = {
            "answer": answer,
            "sources": [],
            "mode": mode,
            "question": question,
            "images": image_refs,
        }
        MemoryCache.set_answer(cache_key, result)
        return result

    async def query_stream(self, question: str, mode: str = "consensus", workspace: str = "default"):
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
            if mode == "consensus":
                t0 = time.perf_counter()
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

                context_text = _build_context_from_chunks(retrieved_chunks)

                tier_summary = " | ".join(
                    f"[{c.get('consensus_source','?')}] {c.get('total_score', 0):.3f}"
                    for c in retrieved_chunks
                )
                logger.info(f"[StreamQuery] Consensus Tiers: {tier_summary}")
                logger.info(
                    f"[StreamQuery] Final Context sent to LLM ({len(context_text)} chars):\n"
                    f"{context_text[:500]}...\n[...]\n{context_text[-200:]}"
                )
                prompt = RAG_RESPONSE_TEMPLATE.format(question=question, context_data=context_text)

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
            retrieved_chunks if mode == "consensus" else [], full_answer, context_text
        )
        formatted_sources = format_chunks_as_sources(retrieved_chunks) if mode == "consensus" else []

        logger.info(f"[StreamQuery] Done: {len(full_answer)} chars, {len(image_refs)} images")

        MemoryCache.set_answer(cache_key, {
            "answer": full_answer,
            "images": image_refs,
            "sources": formatted_sources,
        })

        yield {"type": "done", "images": image_refs, "mode": mode, "sources": formatted_sources}


# Global Pipeline instance
query_pipeline = QueryPipeline()
