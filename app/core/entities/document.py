"""
core/entities/document.py
Domain entity đại diện cho một tài liệu trong hệ thống.
"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


@dataclass
class DocumentEntity:
    """Đại diện cho một tài liệu được upload và index."""
    id: str
    filename: str
    workspace: str
    status: DocumentStatus = DocumentStatus.PENDING
    file_path: Optional[str] = None
    minio_uri: Optional[str] = None
    error_message: Optional[str] = None
