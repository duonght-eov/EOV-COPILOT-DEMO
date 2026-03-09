"""
Threads router - Manage chat threads within workspaces.
"""
import re
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.database import get_db
from app.models import User, Workspace, Chat
from app.models.thread import WorkspaceThread
from app.services.auth_service import get_current_user, has_workspace_access


router = APIRouter(tags=["Threads"])


class ThreadFork(BaseModel):
    """Thread fork request."""
    chatId: int
    threadSlug: Optional[str] = None


# ... existing code ...



# Schemas
class ThreadCreate(BaseModel):
    """Thread creation request."""
    name: Optional[str] = None


class ThreadUpdate(BaseModel):
    """Thread update request."""
    name: str


class ThreadResponse(BaseModel):
    """Thread response format."""
    id: int
    name: str
    slug: str
    workspace_id: int
    created_at: str

    class Config:
        from_attributes = True


def generate_thread_slug() -> str:
    """Generate a unique thread slug."""
    return str(uuid.uuid4())[:8]


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    return text[:50]  # Limit length


async def get_workspace_or_404(slug: str, db: AsyncSession, user: User) -> Workspace:
    """Get workspace by slug or raise 404."""
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Check access
    if not await has_workspace_access(db, user, workspace.id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return workspace


@router.get("/workspace/{slug}/threads")
async def list_threads(
    slug: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all threads in a workspace."""
    workspace = await get_workspace_or_404(slug, db, current_user)
    
    query = select(WorkspaceThread).where(WorkspaceThread.workspace_id == workspace.id)
    if current_user.role.value != "admin":
        query = query.where(WorkspaceThread.user_id == current_user.id)
        
    result = await db.execute(query.order_by(WorkspaceThread.created_at.desc()))
    threads = result.scalars().all()
    
    # Format response for AnythingLLM frontend
    return {
        "threads": [
            {
                "id": thread.id,
                "name": thread.name,
                "slug": thread.slug,
                "workspace_id": thread.workspace_id,
                "created_at": thread.created_at.isoformat() if thread.created_at else None
            }
            for thread in threads
        ]
    }


@router.post("/workspace/{slug}/thread/new")
async def create_thread(
    slug: str,
    body: ThreadCreate = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new thread in a workspace."""
    workspace = await get_workspace_or_404(slug, db, current_user)
    
    # Generate unique slug
    thread_slug = generate_thread_slug()
    
    # Use provided name or default
    name = body.name if body and body.name else "New Thread"
    
    thread = WorkspaceThread(
        name=name,
        slug=thread_slug,
        workspace_id=workspace.id,
        user_id=current_user.id
    )
    
    db.add(thread)
    await db.commit()
    await db.refresh(thread)
    
    return {
        "thread": {
            "id": thread.id,
            "name": thread.name,
            "slug": thread.slug,
            "workspace_id": thread.workspace_id,
            "created_at": thread.created_at.isoformat() if thread.created_at else None
        },
        "message": None
    }


@router.get("/workspace/{slug}/thread/{thread_slug}")
async def get_thread(
    slug: str,
    thread_slug: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific thread."""
    workspace = await get_workspace_or_404(slug, db, current_user)
    
    query = select(WorkspaceThread).where(
        WorkspaceThread.slug == thread_slug,
        WorkspaceThread.workspace_id == workspace.id
    )
    if current_user.role.value != "admin":
        query = query.where(WorkspaceThread.user_id == current_user.id)
        
    result = await db.execute(query)
    thread = result.scalar_one_or_none()
    
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    return {
        "thread": {
            "id": thread.id,
            "name": thread.name,
            "slug": thread.slug,
            "workspace_id": thread.workspace_id,
            "created_at": thread.created_at.isoformat() if thread.created_at else None
        }
    }


@router.post("/workspace/{slug}/thread/{thread_slug}/update")
async def update_thread(
    slug: str,
    thread_slug: str,
    body: ThreadUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a thread's name."""
    workspace = await get_workspace_or_404(slug, db, current_user)
    
    query = select(WorkspaceThread).where(
        WorkspaceThread.slug == thread_slug,
        WorkspaceThread.workspace_id == workspace.id
    )
    if current_user.role.value != "admin":
        query = query.where(WorkspaceThread.user_id == current_user.id)
        
    result = await db.execute(query)
    thread = result.scalar_one_or_none()
    
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    thread.name = body.name
    await db.commit()
    await db.refresh(thread)
    
    return {
        "thread": {
            "id": thread.id,
            "name": thread.name,
            "slug": thread.slug,
            "workspace_id": thread.workspace_id,
            "created_at": thread.created_at.isoformat() if thread.created_at else None
        },
        "message": None
    }


@router.delete("/workspace/{slug}/thread/{thread_slug}")
@router.delete("/workspace/{slug}/thread/{thread_slug}/delete")
async def delete_thread(
    slug: str,
    thread_slug: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a thread."""
    workspace = await get_workspace_or_404(slug, db, current_user)
    
    query = select(WorkspaceThread).where(
        WorkspaceThread.slug == thread_slug,
        WorkspaceThread.workspace_id == workspace.id
    )
    if current_user.role.value != "admin":
        query = query.where(WorkspaceThread.user_id == current_user.id)
        
    result = await db.execute(query)
    thread = result.scalar_one_or_none()
    
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    await db.delete(thread)
    await db.commit()
    
    return {"success": True, "message": "Thread deleted"}


@router.delete("/workspace/{slug}/thread-bulk-delete")
@router.post("/workspace/{slug}/threads/delete")
async def bulk_delete_threads(
    slug: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Bulk delete threads by slugs."""
    workspace = await get_workspace_or_404(slug, db, current_user)
    
    slugs = body.get("slugs", [])
    if not slugs:
        return {"success": True, "message": "No threads to delete"}
    
    query = select(WorkspaceThread).where(
        WorkspaceThread.slug.in_(slugs),
        WorkspaceThread.workspace_id == workspace.id
    )
    if current_user.role.value != "admin":
        query = query.where(WorkspaceThread.user_id == current_user.id)
        
    result = await db.execute(query)
    threads = result.scalars().all()
    
    for thread in threads:
        await db.delete(thread)
    
    await db.commit()
    
    return {"success": True, "message": f"Deleted {len(threads)} threads"}


@router.post("/workspace/{slug}/thread/fork")
async def fork_thread(
    slug: str,
    body: ThreadFork,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Fork a chat thread from a specific message."""
    workspace = await get_workspace_or_404(slug, db, current_user)
    
    # Verify chat exists and belongs to workspace
    result = await db.execute(
        select(Chat).where(
            Chat.id == body.chatId,
            Chat.workspace_id == workspace.id
        )
    )
    target_chat = result.scalar_one_or_none()
    
    if not target_chat:
        raise HTTPException(status_code=404, detail="Chat message not found")
        
    # Get all chats up to this point from the same context
    # If it was in a thread, get from that thread
    # If it was in main workspace, get from main workspace (thread_id exists or null)
    
    query = select(Chat).where(
        Chat.workspace_id == workspace.id,
        Chat.created_at <= target_chat.created_at
    )
    
    if target_chat.thread_id:
        query = query.where(Chat.thread_id == target_chat.thread_id)
    else:
        query = query.where(Chat.thread_id.is_(None))
        
    query = query.order_by(Chat.created_at.asc())
    
    result = await db.execute(query)
    source_chats = result.scalars().all()
    
    if not source_chats:
        raise HTTPException(status_code=404, detail="No history found to fork")
        
    # Create new thread
    new_thread_slug = generate_thread_slug()
    new_thread = WorkspaceThread(
        name=f"Fork of {source_chats[-1].prompt[:20]}...",
        slug=new_thread_slug,
        workspace_id=workspace.id,
        user_id=current_user.id
    )
    db.add(new_thread)
    await db.flush() # Get ID
    
    # Duplicate chats
    for chat in source_chats:
        new_chat = Chat(
            workspace_id=workspace.id,
            user_id=chat.user_id,
            thread_id=new_thread.id,
            prompt=chat.prompt,
            response=chat.response,
            sources=chat.sources,
            session_id=f"thread-{new_thread.id}"
        )
        db.add(new_chat)
        
    await db.commit()
    
    return {"newThreadSlug": new_thread_slug}
