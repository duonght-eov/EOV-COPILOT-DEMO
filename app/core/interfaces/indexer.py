"""
core/interfaces/indexer.py
Abstract interface cho document indexing.
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator


class IIndexer(ABC):
    """Contract cho indexer."""

    @abstractmethod
    async def index(
        self,
        file_path: str,
        workspace: str,
        **kwargs,
    ) -> str:
        """
        Index một tài liệu vào knowledge base.
        Returns: document_id sau khi index xong
        """
        ...

    @abstractmethod
    async def index_stream(
        self,
        file_path: str,
        workspace: str,
        **kwargs,
    ) -> AsyncGenerator[dict, None]:
        """
        Index với streaming progress updates.
        Yields: {"status": str, "progress": float, "message": str}
        """
        ...
