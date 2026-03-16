"""
Database Models for API Gateway
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import enum


class UserRole(str, enum.Enum):
    """User roles."""
    ADMIN = "admin"
    MANAGER = "manager"
    DEFAULT = "default"


class User(Base):
    """User model."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.DEFAULT)
    is_active = Column(Boolean, default=True)
    preferences = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    owned_workspaces = relationship("Workspace", back_populates="owner")
    workspaces = relationship("Workspace", secondary="workspace_users", back_populates="users")
    chats = relationship("Chat", back_populates="user")
    threads = relationship("WorkspaceThread", back_populates="user")


class Workspace(Base):
    """Workspace model - maps to RAGAnything datasets."""
    __tablename__ = "workspaces"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    
    # RAG Configuration
    llm_provider = Column(String(100), nullable=True)
    llm_model = Column(String(100), nullable=True)
    embedding_model = Column(String(100), nullable=True)
    query_mode = Column(String(50), default="consensus")  # naive, local, global, mix, consensus
    
    # Predict/Analytics Configuration
    is_predict_enabled = Column(Boolean, default=True)
    predict_llm_model = Column(String(100), nullable=True)
    
    # Ownership
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="owned_workspaces")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    users = relationship("User", secondary="workspace_users", back_populates="workspaces")
    documents = relationship("Document", back_populates="workspace", cascade="all, delete-orphan")
    chats = relationship("Chat", back_populates="workspace", cascade="all, delete-orphan")
    threads = relationship("WorkspaceThread", back_populates="workspace", cascade="all, delete-orphan")


class WorkspaceUser(Base):
    """Many-to-Many relationship between Users and Workspaces."""
    __tablename__ = "workspace_users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DocumentStatus(str, enum.Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base):
    """Document model."""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer, nullable=True)
    mime_type = Column(String(100), nullable=True)
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.PENDING)
    error_message = Column(Text, nullable=True)
    
    # Workspace
    workspace_id = Column(Integer, ForeignKey("workspaces.id"))
    workspace = relationship("Workspace", back_populates="documents")
    
    # Thread (for attachments)
    thread_id = Column(Integer, ForeignKey("workspace_threads.id", ondelete="CASCADE"), nullable=True)
    is_attachment = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)


class Chat(Base):
    """Chat history model."""
    __tablename__ = "chats"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), index=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    sources = Column(Text, nullable=True)  # JSON string of citations
    metrics = Column(Text, nullable=True)  # JSON string of performance metrics
    
    # Relationships
    workspace_id = Column(Integer, ForeignKey("workspaces.id"))
    workspace = relationship("Workspace", back_populates="chats")
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="chats")
    
    # Thread relationship (optional - chats can exist without a thread)
    thread_id = Column(Integer, ForeignKey("workspace_threads.id", ondelete="CASCADE"), nullable=True)
    thread = relationship("WorkspaceThread", back_populates="chats")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RecoveryCode(Base):
    """Recovery codes for account recovery."""
    __tablename__ = "recovery_codes"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    code_hash = Column(String(255), nullable=False)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User")


class ApiKey(Base):
    """API Key for external access."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=True)
    secret = Column(String(255), unique=True, nullable=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    creator = relationship("User")


class EventLog(Base):
    """Event logging for audit trail."""
    __tablename__ = "event_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(100), nullable=False)
    event_data = Column(Text, nullable=True)  # JSON
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    ip_address = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User")


class SystemSettings(Base):
    """System-wide settings storage."""
    __tablename__ = "system_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(255), unique=True, nullable=False)
    value = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class WorkspaceConnector(Base):
    """Nơi lưu cấu hình CSDL/API của khách hàng thuê nền tảng SaaS."""
    __tablename__ = "workspace_connectors"
    
    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    
    name = Column(String(255), nullable=False)
    connector_type = Column(String(50), default="api", nullable=False)  # api, postgresql, mysql...
    base_url = Column(String(512), nullable=False)
    auth_type = Column(String(50), default="none")  # none, bearer, api_key, basic
    auth_credentials = Column(Text, nullable=True)  # API Key / Token mật
    custom_headers = Column(Text, nullable=True)  # JSON String parameters
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    workspace = relationship("Workspace", backref="connectors")


class ConnectorEndpoint(Base):
    """Bảng Mapping Endpoint API (để Coder không phải sửa fix cứng Code)."""
    __tablename__ = "connector_endpoints"
    
    id = Column(Integer, primary_key=True, index=True)
    connector_id = Column(Integer, ForeignKey("workspace_connectors.id", ondelete="CASCADE"), nullable=False, index=True)
    
    action_code = Column(String(100), nullable=False, index=True)  # GET_DMAS, GET_SHORT_FORECAST...
    endpoint_path = Column(String(512), nullable=False)
    http_method = Column(String(10), default="GET")
    response_mapping = Column(Text, nullable=True)  # JSON config chỉ định path extract data
    
    connector = relationship("WorkspaceConnector", backref="endpoints")
