"""
Users router - User management for admins.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import User, UserRole as ModelUserRole
from app.schemas import UserCreate, UserUpdate, UserResponse
from app.services.auth_service import get_current_user, get_current_active_admin, get_password_hash

router = APIRouter(prefix="/admin/users", tags=["Admin - Users"])


@router.get("")
async def list_users(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """List all users (admin only)."""
    result = await db.execute(select(User))
    users_list = result.scalars().all()
    
    formatted_users = []
    for u in users_list:
        formatted_users.append({
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "role": u.role.value,
            "is_active": u.is_active,
            "suspended": 1 if not u.is_active else 0,
            "createdAt": u.created_at.strftime("%Y-%m-%d %H:%M") if u.created_at else None
        })
        
    return {"users": formatted_users}


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Get user by ID (admin only)."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user


@router.post("")
@router.post("/new")
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Create a new user (admin only)."""
    # Check if username exists
    result = await db.execute(select(User).where(User.username == user_data.username))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create user
    user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=get_password_hash(user_data.password),
        role=ModelUserRole(user_data.role.value)
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return {"user": user}


@router.put("/{user_id}")
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Update a user (admin only)."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Handle suspended status
    if user_data.suspended is not None:
        user.is_active = False if user_data.suspended == 1 else True
    
    # Prevent demoting last admin
    if user.role == ModelUserRole.ADMIN and user_data.role and user_data.role.value != "admin":
        admin_count_res = await db.execute(
            select(User).where(User.role == ModelUserRole.ADMIN)
        )
        if len(admin_count_res.scalars().all()) <= 1:
            raise HTTPException(status_code=400, detail="Cannot remove last admin")
    
    if user_data.username:
        user.username = user_data.username
    if user_data.email:
        user.email = user_data.email
    if user_data.password:
        user.password_hash = get_password_hash(user_data.password)
    if user_data.role:
        user.role = ModelUserRole(user_data.role.value)
    
    await db.commit()
    await db.refresh(user)
    
    # Add suspended status to response object
    user.suspended = 1 if not user.is_active else 0
    return {"success": True, "user": user}


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Delete a user (admin only)."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent deleting last admin
    if user.role == ModelUserRole.ADMIN:
        admin_count = await db.execute(
            select(User).where(User.role == ModelUserRole.ADMIN)
        )
        if len(admin_count.scalars().all()) <= 1:
            raise HTTPException(status_code=400, detail="Cannot delete last admin")
    
    await db.delete(user)
    await db.commit()
    
    return {"success": True, "message": "User deleted"}


@router.patch("/{user_id}/toggle-active")
async def toggle_user_active(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Toggle user active status (admin only)."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.is_active = not user.is_active
    await db.commit()
    
    return {"success": True, "is_active": user.is_active}
