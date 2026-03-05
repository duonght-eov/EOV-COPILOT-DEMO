from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    workspace: str = Field("default", description="Workspace Slug")
    messages: str = Field(..., description="User's Question")
    mode: str = Field("mix", description="Rag Query Mode: Mix, Hybrid, Local, Global, Naive, Consensus")
    stream: bool = False

class ChatResponse(BaseModel):
    response: Optional[str] = None
    sources: Optional[List[Any]] = []
    images: Optional[List[Any]] = []
    metadata: Dict[str, Any] = {}