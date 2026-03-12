"""
application/query_pipeline.py
Trách nhiệm:
  - Orchestrate tiến trình trả lời câu hỏi RAG (Query Pipeline)
  - Điều phối Retriever, Reranker, LLM Generation, và Image Resolver
"""
import os
import time
import hashlib
from typing import Dict, Any

from lightrag import QueryParam
from app.utils.logger import logger
from app.utils.cache import MemoryCache

# Tầng Infrastructure
from app.infrastructure.graph.lightrag_factory import RAGFactory
from app.infrastructure.llm.llm_func import query_llm_func
from app.infrastructure.llm.stream_func import stream_response_llm_func

# Tầng Services
from app.services.retrieval.consensus_retriever import ConsensusRetriever
from app.infrastructure.reranker.bge_reranker import rerank_chunks
from app.services.processing.image_resolver import extract_image_refs_from_answer
from app.services.generation.rag_generator import format_chunks_as_sources, RAG_RESPONSE_TEMPLATE


class QueryPipeline:
    
    async def query(self, question: str, mode: str = "mix", workspace: str = "default") -> Dict[str, Any]:
        """Thực thi câu hỏi và đợi toàn bộ kết quả (Non-streaming)."""
        question_norm = question.strip().lower()
        cache_key = hashlib.md5(f"{workspace}_{mode}_{question_norm}".encode()).hexdigest()

        cached_data = MemoryCache.get_answer(cache_key)
        if cached_data:
            logger.info(f"Answer Cache HIT for question: '{question[:30]}...'")
            return cached_data

        rag_instance, _ = await RAGFactory.get_or_create_rag(workspace)
        logger.info(f"Processing Query [Mode: {mode}, Workspace: {workspace}]: {question}")

        retrieved_chunks = []
        context_text = ""

        # ==========================================================
        # 1. RETRIEVAL PHASE
        # ==========================================================
        if mode == "consensus":
            try:
                retriever = ConsensusRetriever(rag_instance)
                retrieved_chunks = await retriever.consensus_search(
                    query=question, top_k_each_method=5, final_k=5
                )
                if not retrieved_chunks:
                    return {"answer": "Không tìm thấy thông tin phù hợp (Consensus mode)."}

                # Reranking
                if len(retrieved_chunks) >= 2:
                    retrieved_chunks = await rerank_chunks(question, retrieved_chunks)

                # Context Building
                from app.services.processing.context_builder import ContextBuilder
                c_builder = ContextBuilder()
                contexts = []
                for chunk in retrieved_chunks:
                    c_text = chunk.get('content_with_weight', chunk.get('content', ''))
                    # Tận dụng method clean text đã viết trong context_builder (nếu có bổ sung sau)
                    contexts.append(c_text)
                
                context_text = "\n\n------\n\n".join(contexts)

            except Exception as e:
                logger.error(f"Consensus Query Error: {e}")
                return {"answer": "Đã xảy ra lỗi trong quá trình xử lý Consensus Query."}

        else:
            effective_mode = "mix" if mode == "hybrid" else mode
            query_param = QueryParam(mode=effective_mode, only_need_context=True, top_k=5)
            try:
                context_text = await rag_instance.aquery(question, param=query_param)
            except Exception as e:
                logger.warning(f"Keyword extraction failed, falling back to naive mode: {e}")
                try:
                    fallback_param = QueryParam(mode="naive", only_need_context=True, top_k=5)
                    context_text = await rag_instance.aquery(question, param=fallback_param)
                except Exception as e2:
                    logger.error(f"Naive fallback also failed: {e2}")
                    return {"answer": "Đã xảy ra lỗi khi truy xuất dữ liệu.", "mode": mode, "images": []}

        # ==========================================================
        # 2. VALIDATION PHASE (Anti-Hallucination)
        # ==========================================================
        if not context_text or len(context_text.strip()) < 10:
            result = {
                "answer": "Xin lỗi, tôi không tìm thấy thông tin nào liên quan trong tài liệu để trả lời câu hỏi này.",
                "mode": mode, "context": ""
            }
            MemoryCache.set_answer(cache_key, result)
            return result

        # ==========================================================
        # 3. GENERATION PHASE
        # ==========================================================
        prompt = RAG_RESPONSE_TEMPLATE.format(context_data=context_text, question=question)
        try:
            answer = await query_llm_func(prompt)
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            answer = "Xin lỗi, đã xảy ra lỗi trong quá trình tổng hợp câu trả lời."

        # Resolution (Image Tags Extract)
        image_refs = extract_image_refs_from_answer(
            retrieved_chunks if mode == "consensus" else [], 
            answer, context_text
        )

        result = {
            "answer": answer,
            "sources": format_chunks_as_sources(retrieved_chunks) if mode == "consensus" else [],
            "mode": mode,
            "question": question,
            "images": image_refs,
        }
        MemoryCache.set_answer(cache_key, result)
        return result


    async def query_stream(self, question: str, mode: str = "consensus", workspace: str = "default"):
        """Streaming response pipeline SSE."""
        question_norm = question.strip().lower()
        cache_key = hashlib.md5(f"{workspace}_{mode}_{question_norm}".encode()).hexdigest()

        cached_data = MemoryCache.get_answer(cache_key)
        if cached_data:
            yield {"type": "token", "content": cached_data["answer"]}
            yield {"type": "done", "images": cached_data["images"], "mode": mode, "sources": cached_data["sources"]}
            return

        rag_instance, _ = await RAGFactory.get_or_create_rag(workspace)
        
        retrieved_chunks = []
        context_text = ""

        try:
            if mode == "consensus":
                retriever = ConsensusRetriever(rag_instance)
                retrieved_chunks = await retriever.consensus_search(
                    query=question, top_k_each_method=5, final_k=5
                )
                if not retrieved_chunks:
                    yield {"type": "error", "content": "Không tìm thấy thông tin phù hợp."}
                    return
                if len(retrieved_chunks) >= 2:
                    retrieved_chunks = await rerank_chunks(question, retrieved_chunks)

                contexts = [c.get("content_with_weight", c.get("content", "")) for c in retrieved_chunks]
                context_text = "\n\n------\n\n".join(contexts)
                prompt = RAG_RESPONSE_TEMPLATE.format(question=question, context_data=context_text)

            else:
                effective_mode = "mix" if mode == "hybrid" else mode
                query_param = QueryParam(mode=effective_mode, only_need_context=True, top_k=5)
                try:
                    context_text = await rag_instance.aquery(question, param=query_param)
                except Exception:
                    fallback_param = QueryParam(mode="naive", only_need_context=True, top_k=5)
                    context_text = await rag_instance.aquery(question, param=fallback_param)

                if not context_text or len(context_text.strip()) < 10:
                    yield {"type": "error", "content": "Không tìm thấy thông tin liên quan."}
                    return
                prompt = RAG_RESPONSE_TEMPLATE.format(context_data=context_text, question=question)

        except Exception as e:
            logger.error(f"[StreamQuery] Retrieval error: {e}")
            yield {"type": "error", "content": "Lỗi truy xuất dữ liệu."}
            return

        # LLM Streaming Output
        full_answer = ""
        async for token in stream_response_llm_func(prompt):
            full_answer += token
            yield {"type": "token", "content": token}

        # Resolution Process
        image_refs = extract_image_refs_from_answer(
            retrieved_chunks if mode == "consensus" else [], full_answer, context_text
        )
        formatted_sources = format_chunks_as_sources(retrieved_chunks) if mode == "consensus" else []
        
        # Save to Cache
        MemoryCache.set_answer(cache_key, {
            "answer": full_answer, "images": image_refs, "sources": formatted_sources
        })
        
        yield {"type": "done", "images": image_refs, "mode": mode, "sources": formatted_sources}

# Global Pipeline instance
query_pipeline = QueryPipeline()
