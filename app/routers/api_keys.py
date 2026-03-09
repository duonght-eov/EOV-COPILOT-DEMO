"""
API Keys router - Manage API keys for external access.
"""
import secrets
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.database import get_db
from app.models import User, ApiKey
from app.services.auth_service import get_current_user, get_current_active_admin


router = APIRouter(prefix="/system", tags=["System - API Keys"])


# Schemas
class ApiKeyCreate(BaseModel):
    """API Key creation request."""
    name: Optional[str] = None


class ApiKeyResponse(BaseModel):
    """API Key response format."""
    id: int
    name: Optional[str]
    secret: str  # Only shown on creation
    created_at: str
    last_used_at: Optional[str]

    class Config:
        from_attributes = True


def generate_api_key() -> str:
    """Generate a unique API key."""
    return f"sk-{secrets.token_urlsafe(32)}"


@router.get("/api-keys")
async def list_api_keys(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """List all API keys (admin only)."""
    result = await db.execute(
        select(ApiKey).order_by(ApiKey.created_at.desc())
    )
    keys = result.scalars().all()
    
    return {
        "apiKeys": [
            {
                "id": key.id,
                "name": key.name,
                # Only show partial secret for security
                "secret": f"{key.secret[:10]}...{key.secret[-4:]}" if len(key.secret) > 14 else "***",
                "createdBy": key.created_by,
                "createdAt": key.created_at.isoformat() if key.created_at else None,
                "lastUsedAt": key.last_used_at.isoformat() if key.last_used_at else None
            }
            for key in keys
        ]
    }


@router.post("/generate-api-key")
async def generate_new_api_key(
    body: ApiKeyCreate = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Generate a new API key (admin only)."""
    secret = generate_api_key()
    
    api_key = ApiKey(
        name=body.name if body else None,
        secret=secret,
        created_by=current_user.id
    )
    
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)
    
    return {
        "apiKey": {
            "id": api_key.id,
            "name": api_key.name,
            "secret": secret,  # Full secret only shown once
            "createdAt": api_key.created_at.isoformat() if api_key.created_at else None
        },
        "message": "API key created. Save this key, it won't be shown again."
    }


@router.delete("/api-key/{key_id}")
async def delete_api_key(
    key_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Delete an API key (admin only)."""
    result = await db.execute(select(ApiKey).where(ApiKey.id == key_id))
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    await db.delete(api_key)
    await db.commit()
    
    return {"success": True, "message": "API key deleted"}


# API Key validation middleware helper
async def validate_api_key(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> Optional[ApiKey]:
    """Validate API key from request header."""
    auth_header = request.headers.get("Authorization", "")
    
    if not auth_header.startswith("Bearer sk-"):
        return None
    
    api_key_secret = auth_header.replace("Bearer ", "")
    
    result = await db.execute(
        select(ApiKey).where(
            ApiKey.secret == api_key_secret,
            ApiKey.is_active == True
        )
    )
    api_key = result.scalar_one_or_none()
    
    if api_key:
        # Update last used timestamp
        api_key.last_used_at = datetime.utcnow()
        await db.commit()
    
    return api_key
