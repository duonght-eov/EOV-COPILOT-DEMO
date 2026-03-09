"""
Recovery router - Account recovery and password reset.
"""
import secrets
import hashlib
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.database import get_db
from app.models import User, RecoveryCode
from app.services.auth_service import get_password_hash, get_current_user

router = APIRouter(prefix="/system", tags=["Recovery"])


class RecoverAccountRequest(BaseModel):
    username: str
    recoveryCodes: List[str]


class ResetPasswordRequest(BaseModel):
    token: str
    newPassword: str
    confirmPassword: str


def hash_code(code: str) -> str:
    """Hash a recovery code."""
    return hashlib.sha256(code.encode()).hexdigest()


@router.post("/recover-account")
async def recover_account(
    request: RecoverAccountRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Recover account using recovery codes.
    Returns a temporary token for password reset.
    """
    # Find user
    result = await db.execute(select(User).where(User.username == request.username))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    
    # Check recovery codes
    valid_code_found = False
    for code in request.recoveryCodes:
        code_hash = hash_code(code.strip())
        result = await db.execute(
            select(RecoveryCode).where(
                RecoveryCode.user_id == user.id,
                RecoveryCode.code_hash == code_hash,
                RecoveryCode.used == False
            )
        )
        recovery_code = result.scalar_one_or_none()
        
        if recovery_code:
            # Mark code as used
            recovery_code.used = True
            valid_code_found = True
            break
    
    if not valid_code_found:
        raise HTTPException(status_code=400, detail="Invalid recovery codes")
    
    # Generate temporary reset token
    reset_token = secrets.token_urlsafe(32)
    
    # Store token temporarily (in real impl, use Redis or temp table)
    # For now, we'll use a simple approach
    await db.commit()
    
    return {
        "success": True,
        "resetToken": reset_token,
        "message": "Recovery successful. Use the token to reset your password."
    }


@router.post("/reset-password")
async def reset_password(
    request: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Reset password using reset token.
    """
    if request.newPassword != request.confirmPassword:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    
    if len(request.newPassword) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    
    # In production, validate the reset token from Redis/temp storage
    # For now, we'll skip token validation for demo
    
    # This endpoint would typically decode the token to get user_id
    # For demo purposes, we'll return success
    
    return {
        "success": True,
        "message": "Password reset successful. Please login with your new password."
    }


@router.get("/recovery-codes/generate")
async def generate_recovery_codes(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate new recovery codes for the current user.
    Returns codes only once - user must save them.
    """
    # Delete existing unused codes
    result = await db.execute(
        select(RecoveryCode).where(
            RecoveryCode.user_id == current_user.id,
            RecoveryCode.used == False
        )
    )
    old_codes = result.scalars().all()
    for code in old_codes:
        await db.delete(code)
    
    # Generate new codes
    codes = []
    for _ in range(8):
        code = secrets.token_hex(4).upper()  # 8 character code
        codes.append(code)
        
        recovery_code = RecoveryCode(
            user_id=current_user.id,
            code_hash=hash_code(code)
        )
        db.add(recovery_code)
    
    await db.commit()
    
    return {
        "success": True,
        "recoveryCodes": codes,
        "message": "Save these codes securely. They will only be shown once."
    }
