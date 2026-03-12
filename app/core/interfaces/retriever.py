"""
core/interfaces/retriever.py
Abstract interface cho tất cả retrieval strategies.
Implementation cụ thể nằm ở services/retrieval/.
"""
from abc import ABC, abstractmethod
from typing import Dict, List

from app.core.entities.chunk import ChunkEntity


class IRetriever(ABC):
    """Contract cho retriever. Mọi retriever phải implement interface này."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Retrieve chunk IDs với score.
        Returns: {chunk_id: score}
        """
        ...


class IConsensusRetriever(ABC):
    """Contract cho consensus retriever (orchestrate nhiều IRetriever)."""

    @abstractmethod
    async def consensus_search(
        self,
        query: str,
        top_k_each_method: int = 5,
        final_k: int = 3,
    ) -> List[dict]:
        """Chạy multi-retriever, merge và trả về List chunk data."""
        ...
