from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import get_settings
from app.agent_core.prompts import load_react_prompt


def _build_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.openai_api_base,
        api_key=settings.openai_api_key,
        model=settings.model_name,
        temperature=0.0,
        streaming=True,
    )


def build_agent_executor(
    tools: list[Tool],
    session_id: str,
    max_iterations: int = 10,
):
    llm = _build_llm()
    prompt = load_react_prompt("react_water_vi")
    memory = MemorySaver()

    # Phiên bản LangGraph mới dùng 'prompt' thay vì 'state_modifier'
    agent_graph = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SystemMessage(content=prompt.template),
        checkpointer=memory,
    )

    return agent_graph
