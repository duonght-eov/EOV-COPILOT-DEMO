import httpx
from langchain_core.tools import Tool
from app.core.config import get_settings


async def _search_documents(query: str, workspace_slug: str = "default") -> str:
    """Gọi RAG Service để tìm kiếm thông tin trong tài liệu nội bộ."""
    settings = get_settings()
    url = f"{settings.rag_service_url}/api/v1/chat"
    payload = {
        "workspace": workspace_slug,
        "messages": query,
        "mode": "mix",
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("response") or ""
            sources = data.get("sources", [])
            
            # Trả về JSON string để Router bắt được RAW Array của sources
            # và Agent LLM đọc được RAW answer có chứa citation marker (vd [1], [2]).
            raw_output = {
                "answer": answer or "Không tìm thấy thông tin liên quan trong tài liệu.",
                "sources": sources
            }
            import json
            return json.dumps(raw_output, ensure_ascii=False)
    except Exception as e:
        import json
        return json.dumps({"answer": f"[Lỗi tra cứu tài liệu: {e}]", "sources": []}, ensure_ascii=False)


def make_document_tool(workspace_slug: str = "default") -> Tool:
    """Factory tạo Tool với workspace_slug được bind sẵn."""
    async def _run(query: str) -> str:
        return await _search_documents(query, workspace_slug)

    return Tool(
        name="search_documents",
        description=(
            "Tìm kiếm quy trình vận hành, SOP, hướng dẫn kỹ thuật, ngưỡng áp suất "
            "trong tài liệu nội bộ. Dùng khi câu hỏi liên quan đến quy định, tiêu chuẩn, "
            "cách xử lý sự cố, hoặc bất kỳ thông tin nào trong tài liệu PDF đã upload. "
            "Input: câu hỏi dạng text tiếng Việt."
        ),
        coroutine=_run,
        func=lambda q: None, 
    )
