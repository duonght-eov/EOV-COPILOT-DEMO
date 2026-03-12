"""
core/entities/query.py
Domain entity đại diện cho một query từ người dùng.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QueryEntity:
    """Đại diện cho một câu hỏi từ người dùng."""
    question: str
    workspace: str
    mode: str = "hybrid"
    history: List[dict] = field(default_factory=list)
    session_id: Optional[str] = None
