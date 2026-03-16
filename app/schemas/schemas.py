"""
Pydantic schemas for API Gateway.
"""
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List, Any
from datetime import datetime
from enum import Enum


# ===== Auth Schemas =====
class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token payload data."""
    username: Optional[str] = None
    user_id: Optional[int] = None


class UserRole(str, Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    DEFAULT = "default"


class UserBase(BaseModel):
    """Base user schema."""
    username: str = Field(..., min_length=3, max_length=100)
    email: Optional[EmailStr] = None


class UserCreate(UserBase):
    """User creation schema."""
    password: str = Field(..., min_length=6)
    role: UserRole = UserRole.DEFAULT


class UserUpdate(BaseModel):
    """User update schema with optional fields."""
    username: Optional[str] = Field(None, min_length=3, max_length=100)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=6)
    role: Optional[UserRole] = None
    suspended: Optional[int] = None  # 0 or 1
    dailyMessageLimit: Optional[int] = None


class UserLogin(BaseModel):
    """User login schema."""
    username: str
    password: str


class UserResponse(UserBase):
    """User response schema."""
    id: int
    role: UserRole
    is_active: bool
    suspended: int = 0
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# ===== Workspace Schemas =====
class WorkspaceBase(BaseModel):
    """Base workspace schema."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class WorkspaceCreate(WorkspaceBase):
    """Workspace creation schema."""
    pass


class WorkspaceUpdate(BaseModel):
    """Workspace update schema."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    query_mode: Optional[str] = None
    is_predict_enabled: Optional[bool] = None
    predict_llm_model: Optional[str] = None


class WorkspaceResponse(WorkspaceBase):
    """Workspace response schema."""
    id: int
    slug: str
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    query_mode: str = "consensus"
    is_predict_enabled: bool = False
    predict_llm_model: Optional[str] = None
    created_at: datetime
    document_count: int = 0
    
    model_config = ConfigDict(from_attributes=True)


class WorkspaceConnectorUpdate(BaseModel):
    """Update API Connector Configuration for a workspace."""
    base_url: Optional[str] = None
    auth_type: Optional[str] = "bearer"
    auth_credentials: Optional[str] = None
    custom_headers: Optional[str] = None

class WorkspaceConnectorResponse(BaseModel):
    id: int
    workspace_id: int
    base_url: str
    auth_type: str
    auth_credentials: Optional[str] = None
    custom_headers: Optional[str] = None
    created_at: datetime 
    
    model_config = ConfigDict(from_attributes=True)


# ===== Document Schemas =====
class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentResponse(BaseModel):
    """Document response schema."""
    id: int
    filename: str
    original_filename: str
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    status: DocumentStatus
    error_message: Optional[str] = None
    created_at: datetime
    processed_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


# ===== Chat Schemas =====
class ChatMessage(BaseModel):
    """Chat message schema."""
    message: str
    mode: Optional[str] = None  # local, global, hybrid, mix, naive
    sessionId: Optional[str] = None
    attachments: Optional[List[Any]] = []


class ChatResponse(BaseModel):
    """Chat response schema."""
    id: int
    prompt: str
    response: Optional[str] = None
    sources: Optional[List[Any]] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# ===== System Schemas =====
class SystemInfo(BaseModel):
    """System information."""
    version: str = "1.0.0"
    mode: str = "multi"  # single or multi user
    is_configured: bool = True


class LLMProvider(BaseModel):
    """LLM provider info."""
    name: str
    models: List[str]
