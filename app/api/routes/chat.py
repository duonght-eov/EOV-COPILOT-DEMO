import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.application.query_pipeline import query_pipeline

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = await query_pipeline.query(
            question=request.messages,
            mode=request.mode,
            workspace=request.workspace
        )

        return ChatResponse(
            response=result.get("answer"),
            sources=result.get("retrieved_chunks", []),
            images=result.get("images", []),
            metadata={"mode": request.mode, "workspace": request.workspace}
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    True streaming endpoint: yields SSE events token by token.
    Event format:
      data: {"type": "token", "content": "..."}\n\n
      data: {"type": "done", "images": [...], "mode": "...", "sources": []}\n\n
      data: {"type": "error", "content": "..."}\n\n
    """
    async def event_generator():
        try:
            async for event in query_pipeline.query_stream(
                question=request.messages,
                mode=request.mode,
                workspace=request.workspace,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )