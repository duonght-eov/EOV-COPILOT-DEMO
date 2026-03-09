"""
Invites router - Manage user invitations.
"""
import secrets
from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.database import get_db
from app.models import User
from app.models.invite import Invite
from app.services.auth_service import get_current_user, get_current_active_admin, get_password_hash


router = APIRouter(tags=["Admin - Invites"])


# Schemas
class InviteCreate(BaseModel):
    """Invite creation request."""
    email: Optional[str] = None
    expires_in_days: Optional[int] = 7  # Default 7 days


class InviteResponse(BaseModel):
    """Invite response format."""
    id: int
    code: str
    email: Optional[str]
    used: bool
    expires_at: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


class AcceptInviteRequest(BaseModel):
    """Accept invite request."""
    username: str
    password: str
    email: Optional[str] = None


def generate_invite_code() -> str:
    """Generate a unique invite code."""
    return secrets.token_urlsafe(16)


@router.get("/admin/invites")
async def list_invites(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """List all invites (admin only)."""
    result = await db.execute(
        select(Invite).order_by(Invite.created_at.desc())
    )
    invites = result.scalars().all()
    
    return {
        "invites": [
            {
                "id": invite.id,
                "code": invite.code,
                "email": invite.email,
                "used": invite.used,
                "expires_at": invite.expires_at.isoformat() if invite.expires_at else None,
                "created_at": invite.created_at.isoformat() if invite.created_at else None,
                "created_by": invite.created_by
            }
            for invite in invites
        ]
    }


@router.post("/admin/invite/new")
async def create_invite(
    body: InviteCreate = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Create a new invite (admin only)."""
    code = generate_invite_code()
    
    expires_at = None
    if body and body.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=body.expires_in_days)
    
    invite = Invite(
        code=code,
        email=body.email if body else None,
        created_by=current_user.id,
        expires_at=expires_at
    )
    
    db.add(invite)
    await db.commit()
    await db.refresh(invite)
    
    return {
        "invite": {
            "id": invite.id,
            "code": invite.code,
            "email": invite.email,
            "expires_at": invite.expires_at.isoformat() if invite.expires_at else None,
            "created_at": invite.created_at.isoformat() if invite.created_at else None
        },
        "message": None
    }


@router.delete("/admin/invite/{invite_id}")
async def delete_invite(
    invite_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Delete an invite (admin only)."""
    result = await db.execute(select(Invite).where(Invite.id == invite_id))
    invite = result.scalar_one_or_none()
    
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")
    
    await db.delete(invite)
    await db.commit()
    
    return {"success": True, "message": "Invite deleted"}


@router.get("/invite/{code}")
async def get_invite(
    code: str,
    db: AsyncSession = Depends(get_db)
):
    """Get invite details by code (public)."""
    result = await db.execute(select(Invite).where(Invite.code == code))
    invite = result.scalar_one_or_none()
    
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")
    
    if invite.used:
        raise HTTPException(status_code=400, detail="Invite already used")
    
    if invite.expires_at and invite.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Invite expired")
    
    return {
        "invite": {
            "code": invite.code,
            "email": invite.email,
            "valid": True
        }
    }


@router.post("/invite/{code}/accept")
async def accept_invite(
    code: str,
    body: AcceptInviteRequest,
    db: AsyncSession = Depends(get_db)
):
    """Accept an invite and create a user account (public)."""
    result = await db.execute(select(Invite).where(Invite.code == code))
    invite = result.scalar_one_or_none()
    
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")
    
    if invite.used:
        raise HTTPException(status_code=400, detail="Invite already used")
    
    if invite.expires_at and invite.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Invite expired")
    
    # Check if username exists
    existing = await db.execute(select(User).where(User.username == body.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create user
    from app.models import UserRole
    user = User(
        username=body.username,
        email=body.email or invite.email,
        password_hash=get_password_hash(body.password),
        role=UserRole.DEFAULT
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    # Mark invite as used
    invite.used = True
    invite.used_by = user.id
    invite.used_at = datetime.utcnow()
    await db.commit()
    
    return {
        "success": True,
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role.value
        },
        "message": "Account created successfully"
    }
