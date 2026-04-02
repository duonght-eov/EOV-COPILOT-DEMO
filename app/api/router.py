"""
router.py – Hybrid Agentic Router (SSE Proxy for RAG + ReAct for Predict)

Luồng xử lý (Hybrid):
  1. Nhận câu hỏi -> Phân loại Intent (Y định).
  2. RẼ NHÁNH:
     A. Nếu Intent = "rag":
        Proxy SSE trực tiếp từ rag-service về UI (Không qua LLM thứ 2, giữ nguyên citation).
     B. Nếu Intent = "predict", "dma_list", "history":
        Sử dụng AgentExecutor (ReAct) để suy luận câu trả lời đúng trọng tâm.
"""
import json
import uuid
import asyncio
import httpx
import re
from typing import Optional, AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import get_settings
from app.tools.predict_tool import make_predict_tools
from app.tools.document_tool import make_document_tool
from app.agent_core.loops import build_agent_executor
from langchain_core.tools import StructuredTool

router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])

# ── Intent labels ──────────────────────────────────────────────────────────
INTENT_RAG = "rag"
INTENT_PREDICT = "predict" # bao gồm short_forecast, long_forecast, history, dma_list
INTENT_GENERAL = "general"

INTENT_SYSTEM_PROMPT = """Bạn là bộ máy phân loại ý định (Intent Classifier). Phân loại câu hỏi thành đúng MỘT trong các nhãn sau:
- rag: câu hỏi về quy trình, SOP, tài liệu nội bộ, hướng dẫn kỹ thuật, tiêu chuẩn, xây dựng, thi công, cách làm.
- predict: câu hỏi về dự báo (ngắn hạn/dài hạn), danh sách mã DMA, hoặc tra cứu lịch sử tiêu thụ nước của DMA.
- general: câu hỏi chung, chào hỏi, không thuộc nhóm trên.

Chỉ trả về đúng một từ khóa nhãn (rag, predict, general), không giải thích."""

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    workspace_slug: Optional[str] = "default"
    connector: Optional[dict] = None

class ChatResponse(BaseModel):
    answer: str
    status: str = "ok"


# ── Root-level router for AnythingLLM compatibility ───────────────────────
# Create a second router without prefix for /api/* endpoints
root_router = APIRouter(tags=["Root"])

@root_router.post("/workspace/{workspace}/stream-chat")
async def workspace_stream_chat_root(
    workspace: str,
    request: ChatRequest,
):
    """
    AnythingLLM-compatible endpoint at /api/workspace/{workspace}/stream-chat
    This matches what SpeedMaint UI frontend expects.
    """
    # Update request with workspace
    request.workspace_slug = workspace
    # Call the streaming endpoint (defined later in this file)
    return chat_stream(request)


def _classify_intent(message: str) -> str:
    """Phân loại intent bằng keyword matching (không gọi LLM, nhanh hơn)."""
    predict_keywords = [
        "dự báo", "dma", "mã dma", "tiêu thụ", "lịch sử", "forecast",
        "khu vực", "danh sách dma", "có tồn tại", "tồn tại không",
        "short-term", "long-term", "ngắn hạn", "dài hạn", "lượng nước"
    ]
    msg_lower = message.lower()
    if any(k in msg_lower for k in predict_keywords):
        return INTENT_PREDICT
    return INTENT_RAG


def _clean_english_from_response(llm_text: str, tool_outputs: list[str]) -> str:
    """
    Post-process: Loại bỏ text tiếng Anh, giữ nguyên tool output tiếng Việt.
    Tool outputs đã có format tiếng Việt chuẩn, không cần LLM thêm vào.
    """
    if not tool_outputs:
        return llm_text  # Không có tool output, trả nguyên

    # Tìm tool output dài nhất (chứa dữ liệu chính)
    main_tool_output = max(tool_outputs, key=len) if tool_outputs else ""

    # Lọc patterns tiếng Anh thường gặp
    english_patterns = [
        r"Here is (?:the )?(?:complete )?full list of.*?(?=\n\n|\n\[|\Z)",
        r"The data includes.*?(?=\n\n|\n\[|\Z)",
        r"Format: Most codes follow.*?(?=\n\n|\n\[|\Z)",
        r"Special Entry:.*?(?=\n\n|\n\[|\Z)",
        r"Notes?:.*?If you need.*?(?=\n\n|\n\[|\Z)",
        r"Usage:.*?(?=\n\n|\n\[|\Z)",
        r"District Metered Area.*?(?=\n\n|\n\[|\Z)",
    ]

    for pattern in english_patterns:
        llm_text = re.sub(pattern, "", llm_text, flags=re.IGNORECASE | re.DOTALL)

    # Xóa dòng trống thừa
    lines = llm_text.split("\n")
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        is_empty = not line.strip()
        if not (prev_empty and is_empty):  # Không có 2 dòng trống liên tiếp
            cleaned_lines.append(line)
        prev_empty = is_empty

    result = "\n".join(cleaned_lines).strip()

    # Nếu result trống hoặc quá ngắn, dùng tool output
    if len(result) < 50 and main_tool_output:
        return main_tool_output

    return result


async def _proxy_rag_stream(message: str, workspace_slug: str, stream_uuid: str):
    """
    Gói Proxy trực tiếp sang RAG Service.
    Dịch các event ('token', 'done') của RAG thành cấu trúc Frontend/Gateway cần ('textResponseChunk', 'textResponse').
    """
    settings = get_settings()
    url = f"{settings.rag_service_url}/api/v1/chat/stream"
    payload = {
        "workspace": workspace_slug,
        "messages": message,
        "mode": "mix",
        "stream": True,
    }
    
    full_text = ""
    sources = []
    images = []

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
                        yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'textResponseChunk', 'textResponse': token, 'close': False, 'sources': []}, ensure_ascii=False)}\n\n"

                elif etype == "done":
                    sources = event.get("sources", [])
                    images = event.get("images", [])

                elif etype == "error":
                    err_content = event.get("content", "")
                    err_text = "\n[Lỗi RAG: " + err_content + "]"
                    err_payload = {"uuid": stream_uuid, "type": "textResponseChunk", "textResponse": err_text, "close": False, "sources": []}
                    yield "data: " + json.dumps(err_payload, ensure_ascii=False) + "\n\n"

        # Phát event textResponse cuối cùng để chốt sources lên UI
        yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'textResponse', 'textResponse': full_text, 'close': True, 'sources': sources, 'images': images}, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming endpoint (Hybrid SSE).
    - RAG: Proxy trực tiếp (Giữ nguyên citation).
    - Predict/DMA: AgentExecutor ReAct (Có suy luận).
    """
    session_id = request.session_id or str(uuid.uuid4())
    stream_uuid = str(uuid.uuid4())
    connector = request.connector or {}
    workspace_slug = request.workspace_slug or "default"

    # Bước 1: Phân loại Intent
    try:
        intent = _classify_intent(request.message)
    except:
        intent = INTENT_RAG

    # Bước 2: Rẽ nhánh xử lý
    if intent == INTENT_RAG:
        # FLOW A: TRỰC TIẾP SANG RAG (Bypass LLM Agent)
        return StreamingResponse(
            _proxy_rag_stream(request.message, workspace_slug, stream_uuid),
            media_type="text/event-stream"
        )
    
    # FLOW B: RE-ACT AGENT TRẢ LỜI CÓ SUY LUẬN (Dành cho Tools số liệu)
    tools = make_predict_tools(
        base_url=connector.get("base_url"),
        api_key=connector.get("auth_credentials")
    )
    # Vẫn add document_tool đề phòng Agent muốn đối chiếu số liệu với tài liệu
    tools.append(make_document_tool(workspace_slug))
    
    agent = build_agent_executor(tools, session_id)

    async def agent_event_generator():
        config = {"configurable": {"thread_id": session_id}}
        all_sources = []
        all_images = []
        full_text = ""
        predict_tool_detected = False
        
        try:
            raw_tool_output = ""
            async for event in agent.astream_events(
                {"messages": [("user", request.message)]},
                config=config,
                version="v2"
            ):
                kind = event["event"]

                # Streaming token AI sinh ra (AI sẽ tự đọc kết quả tool và trả lời)
                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        # CHẶN GEMINI: Khi kết quả là BẢNG (có dấu |), cấm LLM nhúng tay vào 
                        # vì Gemini không thể gõ chính xác bảng dài.
                        if not predict_tool_detected:
                            full_text += chunk.content
                            yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'textResponseChunk', 'textResponse': chunk.content, 'close': False, 'sources': []}, ensure_ascii=False)}\n\n"

                elif kind == "on_tool_end":
                    output = event["data"].get("output")
                    try:
                        output_content = str(output) if output else ""
                        if hasattr(output, 'content'):
                            output_content = output.content
                        elif isinstance(output, dict) and 'content' in output:
                            output_content = output['content']

                        # Bypass: Nếu tool xuất Bảng SIÊU LỚN (>1500 ký tự) gây tràn ngữ cảnh, ta mới bật cờ chặn
                        is_huge_table = "|" in output_content and ("mã DMA trong hệ thống" in output_content or len(output_content) > 1500)
                        
                        if is_huge_table and not output_content.strip().startswith("{"):
                            # Dọn thẻ hướng dẫn ẩn (nếu có lúc trước nhúng vào)
                            if "<system_instruction>" in output_content:
                                raw_tool_output = output_content.split("</system_instruction>\n\n")[-1]
                            else:
                                raw_tool_output = output_content
                            predict_tool_detected = True

                        # Chỉ bóc sources/images nếu tool có trả về json format
                        if output_content.strip().startswith("{"):
                            data = json.loads(output_content)
                            if "sources" in data: all_sources.extend(data["sources"])
                            if "images" in data: all_images.extend(data["images"])
                    except: pass
                
                # NGẮT LLM SỚM (SHORT-CIRCUIT):
                # Khi đã nhận được bảng kết quả hoàn hảo (predict_tool_detected = True), 
                # Dừng hẳn vòng lặp stream của Agent ở đây để giải phóng LLM, chống treo hệ thống.
                if predict_tool_detected:
                    break

            # Trình giả lập Streaming (Bypass): Trả đúng 100% từng chữ cái trong format bảng của Tool
            if raw_tool_output:
                full_text = raw_tool_output
                chunk_size = 15
                for i in range(0, len(raw_tool_output), chunk_size):
                    piece = raw_tool_output[i:i+chunk_size]
                    yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'textResponseChunk', 'textResponse': piece, 'close': False, 'sources': []}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)

            yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'textResponse', 'textResponse': full_text, 'close': True, 'sources': all_sources, 'images': all_images}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            err_msg = "\n[Hệ thống AI xử lý bị lỗi: " + str(e) + "]"
            err_payload = {"uuid": stream_uuid, "type": "textResponse", "textResponse": err_msg, "close": True, "sources": [], "images": []}
            yield "data: " + json.dumps(err_payload, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        agent_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@router.post("/chat", response_model=ChatResponse)
async def chat_sync(request: ChatRequest):
    """Endpoint đồng bộ."""
    intent = _classify_intent(request.message)  # FIX: không phải async function
    if intent == INTENT_RAG:
        settings = get_settings()
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{settings.rag_service_url}/api/v1/chat", json={
                "workspace": request.workspace_slug or "default",
                "messages": request.message,
                "mode": "mix",
                "stream": False
            })
            data = resp.json()
            return ChatResponse(answer=data.get("response", ""), status="ok")
    
    connector = request.connector or {}
    tools = make_predict_tools(connector.get("base_url"), connector.get("auth_credentials"))
    tools.append(make_document_tool(request.workspace_slug))
    agent = build_agent_executor(tools, request.session_id or "default")
    final_state = await agent.ainvoke({"messages": [("user", request.message)]}, config={"configurable": {"thread_id": request.session_id or "default"}})
    return ChatResponse(answer=final_state["messages"][-1].content, status="ok")
