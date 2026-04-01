"""
router.py – Agentic Router (Intent Classification + Direct Service Proxy)

Luồng:
  1. LLM phân loại intent từ câu hỏi (chỉ 1 lần LLM, không tool-calling).
  2. Dựa trên intent → gọi thẳng RAG-service hoặc Predict-service.
  3. Proxy SSE stream trực tiếp về Frontend — không qua LLM lần nào nữa.

Kết quả:
  - Citation / sources từ RAG được giữ nguyên 100%
  - Streaming token-by-token nhanh hơn
  - Không có LLM thứ 3 làm méo nội dung
"""
import json
import uuid
from typing import Optional

import httpx
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core.config import get_settings
from app.tools.predict_tool import (
    _get_short_term_forecast,
    _get_long_term_forecast,
    _get_history,
    _get_dma_list,
)

router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])

# ── Intent labels ──────────────────────────────────────────────────────────
INTENT_RAG = "rag"
INTENT_SHORT_FORECAST = "short_forecast"    # get_short_term_forecast
INTENT_LONG_FORECAST = "long_forecast"      # get_long_term_forecast
INTENT_HISTORY = "history"                  # get_history
INTENT_DMA_LIST = "dma_list"               # get_dma_list
INTENT_GENERAL = "general"                  # general knowledge, chitchat


INTENT_SYSTEM_PROMPT = """Bạn là bộ phân loại intent. Phân loại câu hỏi thành đúng MỘT trong các nhãn sau:
- rag: câu hỏi về quy trình, SOP, tài liệu nội bộ, hướng dẫn kỹ thuật, tiêu chuẩn, xây dựng, thi công
- short_forecast: dự báo ngắn hạn (tháng tới, tuần tới) tiêu thụ nước DMA
- long_forecast: dự báo dài hạn (quý, 3 tháng tiếp theo) tiêu thụ nước DMA
- history: dữ liệu lịch sử/thực tế tiêu thụ nước DMA trong quá khứ
- dma_list: danh sách DMA, khu vực, ID DMA
- general: câu hỏi chung, chào hỏi, không thuộc nhóm trên

Chỉ trả về đúng một trong các từ khóa trên, không giải thích gì thêm."""


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    workspace_slug: Optional[str] = "default"
    connector: Optional[dict] = None


class ChatResponse(BaseModel):
    answer: str
    status: str = "ok"


def _build_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.openai_api_base,
        api_key=settings.openai_api_key,
        model=settings.model_name,
        temperature=0.0,
        streaming=False,
    )


async def _classify_intent(message: str) -> str:
    """Gọi LLM một lần duy nhất để phân loại intent câu hỏi."""
    llm = _build_llm()
    from langchain_core.messages import SystemMessage, HumanMessage
    result = await llm.ainvoke([
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=message),
    ])
    intent = result.content.strip().lower()
    # Validate
    valid = {INTENT_RAG, INTENT_SHORT_FORECAST, INTENT_LONG_FORECAST,
             INTENT_HISTORY, INTENT_DMA_LIST, INTENT_GENERAL}
    return intent if intent in valid else INTENT_RAG


async def _proxy_rag_stream(message: str, workspace_slug: str, stream_uuid: str):
    """
    Gọi RAG-service /chat/stream và chuyển đổi format SSE về format Frontend.

    Frontend (utils/chat/index.js) xử lý 2 loại event:
    - type=textResponseChunk: streaming token, sources tạm thời
    - type=textResponse (close=True): ghi sources VĨNH VIỄN vào UI

    → Phải phát textResponse ở cuối với đầy đủ sources.
    """
    settings = get_settings()
    url = f"{settings.rag_service_url}/api/v1/chat/stream"
    payload = {
        "workspace": workspace_slug,
        "messages": message,
        "mode": "mix",
        "stream": True,
    }
    sources = []
    images = []
    full_text = ""
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw:
                        continue
                    try:
                        event = json.loads(raw)
                    except Exception:
                        continue

                    etype = event.get("type", "")

                    if etype == "token":
                        token = event.get("content", "")
                        if token:
                            full_text += token
                            yield json.dumps({
                                "uuid": stream_uuid,
                                "type": "textResponseChunk",
                                "textResponse": token,
                                "close": False,
                                "sources": [],
                            }, ensure_ascii=False)

                    elif etype == "done":
                        sources = event.get("sources", [])
                        images = event.get("images", [])

                    elif etype == "error":
                        yield json.dumps({
                            "uuid": stream_uuid,
                            "type": "textResponseChunk",
                            "textResponse": f"\n[Lỗi RAG: {event.get('content', '')}]",
                            "close": False,
                            "sources": [],
                        }, ensure_ascii=False)

        # Phát event textResponse cuối cùng — đây là event quan trọng nhất
        yield json.dumps({
            "uuid": stream_uuid,
            "type": "textResponse",
            "textResponse": full_text,
            "close": True,
            "sources": sources,
            "images": images,
        }, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({
            "uuid": stream_uuid,
            "type": "textResponse",
            "textResponse": full_text or f"\n[Lỗi kết nối RAG service: {e}]",
            "close": True,
            "sources": sources,
            "images": images,
        }, ensure_ascii=False)


async def _handle_predict_intent(intent: str, message: str, stream_uuid: str,
                                  connector: dict, workspace_slug: str):
    """Gọi Predict tools và phát kết quả."""
    import re
    base_url = connector.get("base_url") if connector else None
    api_key = connector.get("auth_credentials") if connector else None

    # Trích mã DMA từ message (vd: "78-Vin", "01-BT", "XN01_01")
    dma_match = re.search(
        r"\b(\d{1,3}-[A-Za-z]{1,6}|[A-Za-z]{1,4}\d{1,3}_\d{1,3}|\d{2,3}[A-Za-z]{2,6})\b",
        message
    )
    dma_id = dma_match.group(1) if dma_match else None

    # Trích số lượng tháng (nếu có, ví dụ: "6 tháng")
    month_match = re.search(r"\b(\d{1,2})\s*tháng", message.lower())
    months = int(month_match.group(1)) if month_match else 24

    try:
        if intent == INTENT_DMA_LIST:
            result = await _get_dma_list(base_url=base_url, api_key=api_key)
        elif intent == INTENT_SHORT_FORECAST:
            if dma_id:
                result = await _get_short_term_forecast(dma_id, base_url=base_url, api_key=api_key)
            else:
                result = "Vui lòng cung cấp mã DMA cụ thể (ví dụ: **78-Vin**, **01-BT**). Bạn có thể hỏi 'Danh sách các DMA' để xem toàn bộ mã."
        elif intent == INTENT_LONG_FORECAST:
            if dma_id:
                result = await _get_long_term_forecast(dma_id, base_url=base_url, api_key=api_key)
            else:
                result = "Vui lòng cung cấp mã DMA cụ thể (ví dụ: **78-Vin**, **01-BT**). Bạn có thể hỏi 'Danh sách các DMA' để xem toàn bộ mã."
        elif intent == INTENT_HISTORY:
            if dma_id:
                result = await _get_history(dma_id, months=months, base_url=base_url, api_key=api_key)
            else:
                result = "Vui lòng cung cấp mã DMA cụ thể (ví dụ: **78-Vin**, **01-BT**). Bạn có thể hỏi 'Danh sách các DMA' để xem toàn bộ mã."
        else:
            result = "Tôi là Water AI. Bạn có thể hỏi tôi về tài liệu nội bộ, dữ liệu DMA hoặc dự báo tiêu thụ nước."

        yield json.dumps({
            "uuid": stream_uuid,
            "type": "textResponseChunk",
            "textResponse": str(result),
            "close": False,
            "sources": [],
        }, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({
            "uuid": stream_uuid,
            "type": "textResponseChunk",
            "textResponse": f"\n[Lỗi lấy dữ liệu: {e}]",
            "close": False,
            "sources": [],
        }, ensure_ascii=False)

    yield json.dumps({
        "uuid": stream_uuid,
        "type": "textResponseChunk",
        "textResponse": "",
        "close": True,
        "sources": [],
    }, ensure_ascii=False)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint đồng bộ."""
    intent = await _classify_intent(request.message)
    settings = get_settings()

    if intent == INTENT_RAG:
        url = f"{settings.rag_service_url}/api/v1/chat"
        payload = {
            "workspace": request.workspace_slug or "default",
            "messages": request.message,
            "mode": "mix",
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            data = resp.json()
            return ChatResponse(answer=data.get("response", ""), status="ok")
    else:
        return ChatResponse(answer="Câu hỏi đã được ghi nhận.", status="ok")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming endpoint (SSE).
    Agent chỉ phân loại intent 1 lần → proxy thẳng đến service phù hợp.
    """
    session_id = request.session_id or str(uuid.uuid4())
    stream_uuid = str(uuid.uuid4())
    connector = request.connector or {}
    workspace_slug = request.workspace_slug or "default"

    async def event_generator():
        # Bước 1: Phân loại intent (1 lần LLM duy nhất, không streaming)
        try:
            intent = await _classify_intent(request.message)
        except Exception:
            intent = INTENT_RAG

        # Bước 2: Route đến đúng service
        if intent == INTENT_RAG:
            async for chunk in _proxy_rag_stream(request.message, workspace_slug, stream_uuid):
                yield f"data: {chunk}\n\n"

        elif intent in (INTENT_SHORT_FORECAST, INTENT_LONG_FORECAST,
                        INTENT_HISTORY, INTENT_DMA_LIST):
            async for chunk in _handle_predict_intent(
                intent, request.message, stream_uuid, connector, workspace_slug
            ):
                yield f"data: {chunk}\n\n"

        else:
            # general / unknown
            yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'textResponseChunk', 'textResponse': 'Tôi là Water AI, sẵn sàng hỗ trợ bạn tra cứu tài liệu và dữ liệu DMA.', 'close': False, 'sources': []}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'textResponseChunk', 'textResponse': '', 'close': True, 'sources': []}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
