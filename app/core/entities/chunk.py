"""
core/entities/chunk.py
Domain entity đại diện cho một text chunk được retrieve.
Không chứa business logic, chỉ là data container thuần túy.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ChunkEntity:
    """Đại diện cho một đoạn văn bản được truy xuất từ knowledge base."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    rerank_score: Optional[float] = None
    consensus_source: Optional[str] = None  # "gold", "silver", "bronze"

    @property
    def page_idx(self) -> Optional[int]:
        return self.metadata.get("page_idx")

    @property
    def source_document(self) -> Optional[str]:
        return self.metadata.get("source") or self.metadata.get("file_path")
