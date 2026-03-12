"""
core/interfaces/reranker.py
Abstract interface cho reranking.
"""
from abc import ABC, abstractmethod
from typing import List


class IReranker(ABC):
    """Contract cho reranker."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: List[dict],
    ) -> List[dict]:
        """
        Sắp xếp lại chunks theo độ liên quan với query.
        Returns: chunks đã được sắp xếp lại (và cắt bớt theo top_n)
        """
        ...
