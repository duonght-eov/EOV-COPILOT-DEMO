"""
services/retrieval/multi_hop_retriever.py
Multi-hop retrieval: phát hiện câu hỏi cần nhiều lần retrieve, phân rã và tổng hợp.
"""
import re
import os
import json
import asyncio
from typing import List, Dict, Optional, Tuple

from app.utils.logger import get_logger
from app.config import settings

logger = get_logger("MULTI_HOP")

# Heuristic keywords — tránh gọi LLM nếu rõ ràng là single-hop
_MULTI_HOP_PATTERNS = re.compile(
    r"\b(so sánh|khác nhau|khác biệt|giống nhau|tương đồng|"
    r"đồng thời|cả hai|cả .+ và|giữa .+ và|"
    r"máy .+ và máy|hệ thống .+ và hệ thống|thiết bị .+ và thiết bị)\b",
    re.IGNORECASE,
)

_SINGLE_HOP_EXCLUDE = re.compile(
    r"\b(một|duy nhất|chỉ|riêng)\b",
    re.IGNORECASE,
)


def _heuristic_is_multi_hop(question: str) -> bool:
    """Kiểm tra nhanh bằng regex trước khi gọi LLM."""
    if _SINGLE_HOP_EXCLUDE.search(question) and not _MULTI_HOP_PATTERNS.search(question):
        return False
    return bool(_MULTI_HOP_PATTERNS.search(question))


def _load_prompt(filename: str) -> str:
    prompt_path = os.path.join(settings.PROMPTS_DIR, filename)
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


async def detect_and_decompose(question: str) -> Tuple[bool, List[str]]:
    """
    Phát hiện multi-hop và phân rã câu hỏi thành sub-queries.
    Returns: (is_multi_hop, sub_queries)
    """
    if not _heuristic_is_multi_hop(question):
        logger.debug(f"[MultiHop] Heuristic: single-hop — '{question[:60]}'")
        return False, []

    logger.info(f"[MultiHop] Heuristic triggered, calling LLM decomposer...")
    try:
        from app.infrastructure.llm.llm_func import llm_completion_func
        template = _load_prompt("query_decomposer.jinja")
        prompt = template.replace("{question}", question)

        raw = await asyncio.wait_for(
            llm_completion_func(prompt, system_prompt=None, max_tokens=400),
            timeout=15.0,
        )

        raw = raw.strip()
        # Tìm JSON trong output (LLM đôi khi thêm text thừa)
        match = re.search(r'\{[\s\S]*\}', raw)
        if not match:
            logger.warning("[MultiHop] LLM did not return valid JSON, fallback to single-hop")
            return False, []

        data = json.loads(match.group())
        is_multi = bool(data.get("is_multi_hop", False))
        sub_queries = [q.strip() for q in data.get("sub_queries", []) if q.strip()]

        logger.info(
            f"[MultiHop] is_multi_hop={is_multi} | reason={data.get('reason', '')} | "
            f"sub_queries={sub_queries}"
        )

        if not is_multi or len(sub_queries) < 2:
            return False, []

        return True, sub_queries[:3]  # Tối đa 3 sub-queries

    except asyncio.TimeoutError:
        logger.warning("[MultiHop] LLM decomposer timeout — fallback to single-hop")
        return False, []
    except Exception as e:
        logger.warning(f"[MultiHop] Decompose error: {e} — fallback to single-hop")
        return False, []


async def multi_hop_retrieve(
    sub_queries: List[str],
    rag_instance,
    top_k_each: int = 3,
) -> List[Dict]:
    """
    Chạy ConsensusRetriever song song cho từng sub-query.
    Returns: list chunks đã gắn nhãn sub_query_label.
    """
    from app.services.retrieval.consensus_retriever import ConsensusRetriever
    retriever = ConsensusRetriever(rag_instance)

    async def _retrieve_one(sub_q: str, label: str) -> List[Dict]:
        try:
            chunks = await retriever.consensus_search(
                query=sub_q,
                top_k_each_method=top_k_each,
                final_k=top_k_each,
            )
            for c in chunks:
                c["_sub_query_label"] = label
            logger.info(f"[MultiHop] '{label}' → {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"[MultiHop] Retrieve failed for '{sub_q}': {e}")
            return []

    tasks = [
        _retrieve_one(q, f"Nguồn {i+1}: {q[:60]}")
        for i, q in enumerate(sub_queries)
    ]
    results = await asyncio.gather(*tasks)

    # Merge + dedup theo content hash
    seen_ids = set()
    merged = []
    for chunk_list in results:
        for chunk in chunk_list:
            cid = chunk.get("id", chunk.get("content", "")[:50])
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(chunk)

    logger.info(f"[MultiHop] Merged {len(merged)} unique chunks from {len(sub_queries)} sub-queries")
    return merged


def build_labeled_context(sub_queries: List[str], chunk_sets: List[List[Dict]]) -> str:
    """
    Tạo context có nhãn rõ ràng theo từng sub-query để LLM so sánh.
    """
    from app.application.query_pipeline import _build_context_from_chunks
    parts = []
    for i, (sub_q, chunks) in enumerate(zip(sub_queries, chunk_sets), start=1):
        section_header = f"=== THÔNG TIN VỀ CHỦ THỂ {i}: {sub_q} ==="
        if chunks:
            context = _build_context_from_chunks(chunks)
        else:
            context = "(Không tìm thấy thông tin liên quan)"
        parts.append(f"{section_header}\n\n{context}")
    return "\n\n" + "\n\n".join(parts) + "\n"


async def multi_hop_retrieve_labeled(
    sub_queries: List[str],
    rag_instance,
    top_k_each: int = 3,
    mode: str = "consensus",
) -> Tuple[List[Dict], str]:
    """
    Retrieve song song và trả về (all_chunks, labeled_context) cho mọi mode truy vấn.
    """
    from app.services.retrieval.consensus_retriever import ConsensusRetriever
    from app.infrastructure.reranker.reranker import rerank_chunks
    from lightrag import QueryParam

    retriever = ConsensusRetriever(rag_instance) if mode == "consensus" else None

    async def _retrieve_one(sub_q: str) -> List[Dict]:
        try:
            if mode == "consensus":
                chunks = await retriever.consensus_search(
                    query=sub_q,
                    top_k_each_method=top_k_each,
                    final_k=top_k_each,
                )
            else:
                effective_mode = "mix" if mode == "hybrid" else mode
                raw = await rag_instance.aquery_data(sub_q, param=QueryParam(mode=effective_mode, top_k=top_k_each))
                data = raw.get("data", {}) if isinstance(raw, dict) else {}
                chunks = data.get("chunks", [])
            if chunks and len(chunks) >= 2:
                chunks = await rerank_chunks(sub_q, chunks)
            return chunks
        except Exception as e:
            logger.error(f"[MultiHop] Retrieve failed for '{sub_q[:50]}': {e}")
            return []

    chunk_sets = await asyncio.gather(*[_retrieve_one(q) for q in sub_queries])
    chunk_sets = list(chunk_sets)

    all_chunks = []
    seen_ids = set()
    for chunks in chunk_sets:
        for c in chunks:
            cid = c.get("id", c.get("content", "")[:50])
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(c)

    labeled_context = build_labeled_context(sub_queries, chunk_sets)
    return all_chunks, labeled_context
