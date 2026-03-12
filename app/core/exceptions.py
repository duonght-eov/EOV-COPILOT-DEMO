"""
core/exceptions.py
Định nghĩa tất cả custom exceptions của rag-service.
Không import bất kỳ thứ gì từ infrastructure hoặc services.
"""


class RAGException(Exception):
    """Base exception cho toàn bộ rag-service."""
    pass


class RetrievalError(RAGException):
    """Lỗi trong quá trình retrieval."""
    pass


class GenerationError(RAGException):
    """Lỗi trong quá trình sinh câu trả lời."""
    pass


class IndexingError(RAGException):
    """Lỗi trong quá trình indexing tài liệu."""
    pass


class EmbeddingError(RAGException):
    """Lỗi khi gọi embedding service."""
    pass


class RerankError(RAGException):
    """Lỗi trong quá trình reranking."""
    pass


class StorageError(RAGException):
    """Lỗi khi tương tác với external storage (MinIO, PostgreSQL)."""
    pass


class ConfigurationError(RAGException):
    """Lỗi cấu hình service."""
    pass
