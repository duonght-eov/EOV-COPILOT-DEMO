"""
Thread model for managing chat threads within workspaces.
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class WorkspaceThread(Base):
    """Thread model for organizing conversations within a workspace."""
    __tablename__ = "workspace_threads"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, default="New Thread")
    slug = Column(String(255), nullable=False, index=True)
    
    # Workspace relationship
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    workspace = relationship("Workspace", back_populates="threads")
    
    # User who created the thread
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="threads")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Chat messages in this thread
    chats = relationship("Chat", back_populates="thread", cascade="all, delete-orphan")
