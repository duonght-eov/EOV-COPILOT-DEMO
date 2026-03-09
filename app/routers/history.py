"""
History router - Chat history, export, and event logs.
"""
import json
import csv
import io
from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from pydantic import BaseModel

from app.database import get_db
from app.models import User, Workspace, Chat, EventLog
from app.services.auth_service import get_current_user, get_current_active_admin


router = APIRouter(prefix="/system", tags=["System - History"])


# Schemas
class ExportFormat(BaseModel):
    """Export format options."""
    format: str = "json"  # json, csv


# ============= Workspace Chats (Admin) =============

@router.get("/workspace-chats")
async def get_all_workspace_chats(
    page: int = 1,
    limit: int = 20,
    workspace_slug: Optional[str] = None,
    user_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Get all workspace chats with pagination (admin only)."""
    from sqlalchemy.orm import selectinload
    
    offset = (page - 1) * limit
    
    # Build query with relationships loaded
    query = select(Chat).options(
        selectinload(Chat.user),
        selectinload(Chat.workspace)
    )
    count_query = select(func.count(Chat.id))
    
    workspace = None
    if workspace_slug:
        # Filter by workspace
        ws_result = await db.execute(
            select(Workspace).where(Workspace.slug == workspace_slug)
        )
        workspace = ws_result.scalar_one_or_none()
        if workspace:
            query = query.where(Chat.workspace_id == workspace.id)
            count_query = count_query.where(Chat.workspace_id == workspace.id)
    
    if user_id:
        query = query.where(Chat.user_id == user_id)
        count_query = count_query.where(Chat.user_id == user_id)
    
    try:
        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Get chats
        query = query.order_by(Chat.created_at.desc()).offset(offset).limit(limit)
        result = await db.execute(query)
        chats = result.scalars().all()
        
        return {
            "chats": [
                {
                    "id": chat.id,
                    "workspaceId": chat.workspace_id,
                    "userId": chat.user_id,
                    "user": {"username": chat.user.username} if chat.user else None,
                    "workspace": {"name": chat.workspace.name, "slug": chat.workspace.slug} if chat.workspace else None,
                    "prompt": chat.prompt,
                    "response": chat.response,
                    "createdAt": chat.created_at.isoformat() if chat.created_at else None
                }
                for chat in chats
            ],
            "totalChats": total,
            "page": page,
            "limit": limit,
            "hasMore": offset + len(chats) < total
        }
    except Exception as e:
        # Database schema might be outdated, return empty result
        return {
            "chats": [],
            "totalChats": 0,
            "page": page,
            "limit": limit,
            "hasMore": False,
            "error": str(e)
        }


@router.delete("/workspace-chats/{chat_id}")
async def delete_workspace_chat(
    chat_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Delete a specific chat (admin only)."""
    result = await db.execute(select(Chat).where(Chat.id == chat_id))
    chat = result.scalar_one_or_none()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    await db.delete(chat)
    await db.commit()
    
    return {"success": True, "message": "Chat deleted"}


# ============= Export Chats =============

@router.get("/export-chats")
async def export_chats(
    format: str = "json",
    workspace_slug: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Export all chats to JSON or CSV (admin only)."""
    # Build query
    query = select(Chat)
    
    if workspace_slug:
        ws_result = await db.execute(
            select(Workspace).where(Workspace.slug == workspace_slug)
        )
        workspace = ws_result.scalar_one_or_none()
        if workspace:
            query = query.where(Chat.workspace_id == workspace.id)
    
    query = query.order_by(Chat.created_at.desc())
    result = await db.execute(query)
    chats = result.scalars().all()
    
    if format.lower() == "csv":
        # Export as CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "workspace_id", "user_id", "prompt", "response", "created_at"])
        
        for chat in chats:
            writer.writerow([
                chat.id,
                chat.workspace_id,
                chat.user_id,
                chat.prompt,
                chat.response,
                chat.created_at.isoformat() if chat.created_at else ""
            ])
        
        content = output.getvalue()
        return Response(
            content=content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=chats_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )
    else:
        # Export as JSON
        data = {
            "exportedAt": datetime.utcnow().isoformat(),
            "totalChats": len(chats),
            "chats": [
                {
                    "id": chat.id,
                    "workspaceId": chat.workspace_id,
                    "userId": chat.user_id,
                    "prompt": chat.prompt,
                    "response": chat.response,
                    "sources": json.loads(chat.sources) if chat.sources else [],
                    "createdAt": chat.created_at.isoformat() if chat.created_at else None
                }
                for chat in chats
            ]
        }
        
        return Response(
            content=json.dumps(data, indent=2, ensure_ascii=False),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=chats_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            }
        )


# ============= Event Logs =============

@router.get("/event-logs")
async def get_event_logs(
    page: int = 1,
    limit: int = 50,
    event_type: Optional[str] = None,
    user_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Get event logs with pagination (admin only)."""
    offset = (page - 1) * limit
    
    # Build query
    query = select(EventLog)
    
    if event_type:
        query = query.where(EventLog.event_type == event_type)
    if user_id:
        query = query.where(EventLog.user_id == user_id)
    
    # Get total count
    count_query = select(func.count(EventLog.id))
    if event_type:
        count_query = count_query.where(EventLog.event_type == event_type)
    if user_id:
        count_query = count_query.where(EventLog.user_id == user_id)
    
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Get logs
    query = query.order_by(EventLog.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    logs = result.scalars().all()
    
    return {
        "eventLogs": [
            {
                "id": log.id,
                "eventType": log.event_type,
                "eventData": json.loads(log.event_data) if log.event_data else None,
                "userId": log.user_id,
                "ipAddress": log.ip_address,
                "createdAt": log.created_at.isoformat() if log.created_at else None
            }
            for log in logs
        ],
        "total": total,
        "page": page,
        "limit": limit,
        "hasMore": offset + len(logs) < total
    }


# ============= Log Event Helper =============

async def log_event(
    db: AsyncSession,
    event_type: str,
    event_data: dict = None,
    user_id: int = None,
    ip_address: str = None
):
    """Helper to log an event."""
    log = EventLog(
        event_type=event_type,
        event_data=json.dumps(event_data) if event_data else None,
        user_id=user_id,
        ip_address=ip_address
    )
    db.add(log)
    await db.commit()


# ============= Slash Command Presets =============

@router.get("/slash-command-presets")
async def get_slash_command_presets(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all slash command presets."""
    from app.models import SystemSettings
    
    result = await db.execute(
        select(SystemSettings).where(SystemSettings.key == "slash_command_presets")
    )
    setting = result.scalar_one_or_none()
    
    if setting and setting.value:
        try:
            presets = json.loads(setting.value)
        except:
            presets = []
    else:
        presets = []
    
    return {"presets": presets}


@router.post("/slash-command-presets")
async def create_slash_command_preset(
    body: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Create a new slash command preset (admin only)."""
    from app.models import SystemSettings
    import uuid
    
    # Get existing presets
    result = await db.execute(
        select(SystemSettings).where(SystemSettings.key == "slash_command_presets")
    )
    setting = result.scalar_one_or_none()
    
    if setting and setting.value:
        try:
            presets = json.loads(setting.value)
        except:
            presets = []
    else:
        presets = []
    
    # Create new preset
    new_preset = {
        "id": str(uuid.uuid4()),
        "name": body.get("name", ""),
        "command": body.get("command", ""),
        "prompt": body.get("prompt", ""),
        "description": body.get("description", ""),
        "createdAt": datetime.utcnow().isoformat()
    }
    
    presets.append(new_preset)
    
    # Save
    if setting:
        setting.value = json.dumps(presets)
    else:
        setting = SystemSettings(key="slash_command_presets", value=json.dumps(presets))
        db.add(setting)
    
    await db.commit()
    
    return {"preset": new_preset, "message": None}


@router.put("/slash-command-presets/{preset_id}")
async def update_slash_command_preset(
    preset_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Update a slash command preset (admin only)."""
    from app.models import SystemSettings
    
    result = await db.execute(
        select(SystemSettings).where(SystemSettings.key == "slash_command_presets")
    )
    setting = result.scalar_one_or_none()
    
    if not setting or not setting.value:
        raise HTTPException(status_code=404, detail="Preset not found")
    
    try:
        presets = json.loads(setting.value)
    except:
        raise HTTPException(status_code=404, detail="Preset not found")
    
    # Find and update preset
    found = False
    for preset in presets:
        if preset.get("id") == preset_id:
            preset.update({
                "name": body.get("name", preset.get("name")),
                "command": body.get("command", preset.get("command")),
                "prompt": body.get("prompt", preset.get("prompt")),
                "description": body.get("description", preset.get("description")),
            })
            found = True
            break
    
    if not found:
        raise HTTPException(status_code=404, detail="Preset not found")
    
    setting.value = json.dumps(presets)
    await db.commit()
    
    return {"presets": presets, "message": None}


@router.delete("/slash-command-presets/{preset_id}")
async def delete_slash_command_preset(
    preset_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Delete a slash command preset (admin only)."""
    from app.models import SystemSettings
    
    result = await db.execute(
        select(SystemSettings).where(SystemSettings.key == "slash_command_presets")
    )
    setting = result.scalar_one_or_none()
    
    if not setting or not setting.value:
        return {"success": True, "message": "Preset not found"}
    
    try:
        presets = json.loads(setting.value)
    except:
        return {"success": True, "message": "Preset not found"}
    
    # Filter out the preset
    presets = [p for p in presets if p.get("id") != preset_id]
    
    setting.value = json.dumps(presets)
    await db.commit()
    
    return {"success": True, "message": "Preset deleted"}
