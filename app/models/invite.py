"""
Invite model for user invitations.
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Invite(Base):
    """Invite model for inviting new users."""
    __tablename__ = "invites"
    
    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), nullable=True)  # Optional target email
    
    # Status
    used = Column(Boolean, default=False)
    used_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Creator
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    creator = relationship("User", foreign_keys=[created_by])
    
    # Validity
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    used_at = Column(DateTime(timezone=True), nullable=True)
