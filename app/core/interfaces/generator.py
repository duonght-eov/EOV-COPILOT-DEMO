"""
core/interfaces/generator.py
Abstract interface cho LLM generation.
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List

from app.core.entities.chunk import ChunkEntity


class IGenerator(ABC):
    """Contract cho generator. Tách biệt logic sinh câu trả lời khỏi retrieval."""

    @abstractmethod
    async def generate(
        self,
        query: str,
        chunks: List[dict],
        **kwargs,
    ) -> dict:
        """
        Sinh câu trả lời từ query + context chunks.
        Returns: {"answer": str, "images": List, "metadata": dict}
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        query: str,
        chunks: List[dict],
        **kwargs,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream câu trả lời token by token.
        Yields: {"type": "token"|"done"|"error", "content": str, ...}
        """
        ...
