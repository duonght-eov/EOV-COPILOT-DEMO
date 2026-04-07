"""
Intent Classifier Module

Phân loại câu hỏi người dùng thành 3 loại:
- rag: Câu hỏi về tài liệu, quy trình, SOP
- predict: Câu hỏi về dự báo, DMA, lịch sử
- general: Câu hỏi chung
"""
import re
from typing import Literal
from app.utils.logger import get_logger

logger = get_logger("IntentClassifier")

# Intent labels
INTENT_RAG: Literal["rag", "predict", "general"] = "rag"
INTENT_PREDICT: Literal["rag", "predict", "general"] = "predict"
INTENT_GENERAL: Literal["rag", "predict", "general"] = "general"

# Predict keywords (dựa trên domain knowledge)
PREDICT_KEYWORDS = [
    "dự báo", "dma", "mã dma", "tiêu thụ", "lịch sử", "forecast",
    "khu vực", "danh sách dma", "có tồn tại", "tồn tại không",
    "short-term", "long-term", "ngắn hạn", "dài hạn", "lượng nước"
]


def classify_intent(message: str) -> Literal["rag", "predict", "general"]:
    """
    Phân loại intent bằng keyword matching (không gọi LLM, nhanh hơn).

    Args:
        message: Câu hỏi của người dùng

    Returns:
        "rag", "predict", hoặc "general"
    """
    msg_lower = message.lower().strip()

    # Rỗng hoặc quá ngắn → general
    if len(msg_lower) < 3:
        logger.debug(f"Message too short, default to RAG: '{message[:50]}...'")
        return INTENT_RAG

    # Check predict keywords
    for keyword in PREDICT_KEYWORDS:
        if keyword in msg_lower:
            logger.info(f"Detected PREDICT intent (keyword: '{keyword}')")
            return INTENT_PREDICT

    # Mặc định: RAG (tài liệu)
    logger.debug(f"Default to RAG intent")
    return INTENT_RAG


# Alias cho backward compatibility
def _classify_intent(message: str) -> str:
    """Hàm alias để tương thích với code cũ."""
    return classify_intent(message)


__all__ = [
    "INTENT_RAG",
    "INTENT_PREDICT",
    "INTENT_GENERAL",
    "classify_intent",
    "_classify_intent",
]
