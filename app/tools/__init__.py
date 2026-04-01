from langchain_core.tools import Tool
from app.tools.document_tool import make_document_tool
from app.tools.predict_tool import make_predict_tools


def build_tools(
    workspace_slug: str = "default",
    predict_base_url: str | None = None,
    predict_api_key: str | None = None,
    enable_predict: bool = True,
) -> list[Tool]:
    """
    Khởi tạo toàn bộ Tool cho Agent theo cấu hình của Workspace.

    Args:
        workspace_slug:     Slug của Workspace hiện tại (để RAG Tool query đúng namespace).
        predict_base_url:   Base URL của Predict Service connector (lấy từ DB).
        predict_api_key:    API Key xác thực Predict Service (lấy từ DB).
        enable_predict:     Bật/tắt các Tool Predict. False = chỉ có search_documents.
    """
    tools: list[Tool] = [
        make_document_tool(workspace_slug=workspace_slug),
    ]

    if enable_predict:
        predict_tools = make_predict_tools(
            base_url=predict_base_url,
            api_key=predict_api_key,
        )
        tools.extend(predict_tools)

    return tools
