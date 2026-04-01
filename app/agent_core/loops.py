from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
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
    max_iterations: int = 7,
):
    llm = _build_llm()
    prompt = load_react_prompt("react_water_vi")
    
    tool_names = ", ".join(t.name for t in tools)
    tool_descs = "\n".join(f"{t.name}: {t.description}" for t in tools)
    system_prompt_str = prompt.template.replace("{tools}", tool_descs).replace("{tool_names}", tool_names)

    memory = MemorySaver()
    
    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt_str,
        checkpointer=memory,
    )

    return agent_graph
