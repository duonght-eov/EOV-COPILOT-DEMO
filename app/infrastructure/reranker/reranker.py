import time
import asyncio
from typing import List, Dict, Any, Optional
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("RERANKER")

# Singleton — load model 1 lần duy nhất khi khởi động
_reranker = None
_reranker_lock: asyncio.Lock | None = None
_reranker_device_info: str = "unknown"


def _load_reranker():
    """Load FlagReranker synchronously (gọi từ asyncio.to_thread)."""
    global _reranker_device_info
    from FlagEmbedding import FlagReranker
    import torch
    model = FlagReranker(
        settings.RERANKER_MODEL_NAME,
        use_fp16=True,
        device=settings.RERANKER_DEVICE,
    )
    # Detect actual device
    try:
        if torch.cuda.is_available() and settings.RERANKER_DEVICE == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            _reranker_device_info = f"GPU ({gpu_name})"
        else:
            _reranker_device_info = "CPU"
    except Exception:
        _reranker_device_info = settings.RERANKER_DEVICE
    logger.info(
        f"[Reranker] Model loaded: {settings.RERANKER_MODEL_NAME} "
        f"| device={_reranker_device_info}"
    )
    return model


async def _get_reranker():
    """Lấy singleton reranker, khởi tạo lazy nếu chưa có."""
    global _reranker, _reranker_lock
    if _reranker is not None:
        return _reranker
    # Tạo lock lazily trong event loop hiện tại, tránh deadlock khi lock được tạo ngoài loop
    if _reranker_lock is None:
        _reranker_lock = asyncio.Lock()
    async with _reranker_lock:
        if _reranker is None:
            _reranker = await asyncio.to_thread(_load_reranker)
    return _reranker


def _compute_scores(reranker, pairs: List[List[str]]) -> List[float]:
    """Tính relevance scores — synchronous, chạy trong thread riêng."""
    import numpy as np
    raw = reranker.compute_score(pairs, normalize=True)
    # Handle scalar vs list
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    # Replace NaN/Inf with 0.0
    return [float(np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)) for s in raw]


async def rerank_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rerank chunks bằng BGE cross-encoder (FlagEmbedding, chạy local).
    Dùng GPU nếu có (RERANKER_DEVICE=cuda), CPU nếu không.
    Fail-safe: trả về list gốc nếu reranker lỗi.
    """
    if not settings.RERANKER_ENABLED or not chunks:
        return chunks

    documents = [chunk.get("content", "") or "" for chunk in chunks]
    if not any(d.strip() for d in documents):
        return chunks

    try:
        reranker = await _get_reranker()

        pairs = [[query, doc] for doc in documents]

        t0 = time.perf_counter()
        scores: List[float] = await asyncio.to_thread(_compute_scores, reranker, pairs)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        scored = [
            (score, idx, {**chunk, "rerank_score": round(float(score), 6)})
            for idx, (chunk, score) in enumerate(zip(chunks, scores))
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        top_n = min(settings.RERANKER_TOP_N, len(scored))
        reranked = [c for _, _, c in scored[:top_n]]

        logger.info(
            f"[Reranker] {len(chunks)} → {len(reranked)} chunks "
            f"| device={_reranker_device_info} "
            f"| inference={elapsed_ms:.1f}ms "
            f"| scores: {[round(s, 4) for s, _, _ in scored[:top_n]]}"
        )
        return reranked

    except Exception as e:
        logger.warning(f"[Reranker] Lỗi: {e} – giữ thứ tự gốc")
        return chunks

_reranker_lock = asyncio.Lock()

async def lightrag_rerank_wrapper(query: str, documents: List[str], top_n: int = None) -> List[Dict[str, Any]]:
    """
    Wrapper tích hợp trực tiếp cho chế độ mix/local/global của LightRAG.
    LightRAG yêu cầu output là list dict: [{"index": i, "relevance_score": score}, ...]
    """
    if not settings.RERANKER_ENABLED or not documents:
        return [{"index": i, "relevance_score": 1.0} for i in range(len(documents))]

    # Filter empty documents
    valid_docs = [d if d and d.strip() else " " for d in documents]

    async with _reranker_lock:  # Chống concurrent access
        try:
            reranker = await _get_reranker()
            pairs = [[query, doc] for doc in valid_docs]
            
            t0 = time.perf_counter()
            scores: List[float] = await asyncio.to_thread(_compute_scores, reranker, pairs)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            scored = [
                {"index": idx, "relevance_score": float(score)}
                for idx, score in enumerate(scores)
            ]
            scored.sort(key=lambda x: x["relevance_score"], reverse=True)

            _top_n = top_n if top_n is not None else settings.RERANKER_TOP_N
            scored = scored[:_top_n]

            logger.info(
                f"[LightRAG-Reranker] {len(documents)} → {len(scored)} chunks "
                f"| inference={elapsed_ms:.1f}ms "
                f"| scores: {[round(x['relevance_score'], 4) for x in scored]}"
            )
            return scored

        except Exception as e:
            logger.warning(f"[LightRAG-Reranker] Lỗi: {e} – bỏ qua rerank")
            return [{"index": i, "relevance_score": 1.0} for i in range(len(documents))]
