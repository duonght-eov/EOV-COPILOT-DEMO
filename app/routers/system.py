"""
System router - System info and configuration.

Note: Response format MUST match what AnythingLLM frontend expects.
"""
from fastapi import APIRouter, Depends, Response
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import os
import json

from app.database import get_db
from app.models import User, UserRole
from app.schemas import SystemInfo
from app.services.auth_service import get_current_user, get_current_active_admin


router = APIRouter(prefix="/system", tags=["System"])


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "speedmaint-api-gateway"}


@router.get("/", response_model=SystemInfo)
async def get_system_info(db: AsyncSession = Depends(get_db)):
    """Get system information."""
    result = await db.execute(
        select(User).where(User.role == UserRole.ADMIN).limit(1)
    )
    admin_exists = result.scalar_one_or_none() is not None
    
    return SystemInfo(
        version="1.0.0",
        mode="multi",
        is_configured=admin_exists
    )


# ============= AnythingLLM Frontend Compatibility Endpoints =============

@router.get("/setup-complete")
async def setup_complete_get(db: AsyncSession = Depends(get_db)):
    """
    Check if initial setup is complete.
    AnythingLLM frontend calls this on startup.
    Expected response: { results: { ... } }
    """
    result = await db.execute(
        select(User).where(User.role == UserRole.ADMIN).limit(1)
    )
    admin_exists = result.scalar_one_or_none() is not None
    
    return {
        "results": {
            "RequiresAuth": True,
            "AuthToken": False,
            "JWTSecret": True,
            "StorageDir": "/app/data",
            "MultiUserMode": True,
            "LLMProvider": "openai",
            "EmbeddingEngine": "openai",
            "VectorDB": "lancedb",
            "HasExistingEmbeddings": True,
            "EmbedderModelPref": "text-embedding-3-small",
            "LLMModelPref": "gpt-4-turbo-preview",
            "AdminExists": admin_exists,
            "DisableViewChatHistory": False,
        }
    }


@router.post("/setup-complete")
async def setup_complete_post(db: AsyncSession = Depends(get_db)):
    """POST version of setup-complete."""
    return await setup_complete_get(db)


@router.get("/multi-user-mode")
async def multi_user_mode():
    """Check if multi-user mode is enabled."""
    return {"multiUserMode": True}


@router.get("/check-token")
async def check_token(current_user: User = Depends(get_current_user)):
    """Validate JWT token."""
    return {"valid": True, "user": current_user.username}


@router.get("/logo")
async def get_logo(theme: str = "light"):
    """Get system logo."""
    # Check for custom logo
    upload_dir = os.path.join(os.getcwd(), "uploads")
    branding_dir = os.path.join(upload_dir, "branding")
    logo_path = os.path.join(branding_dir, "logo.png")
    
    if os.path.exists(logo_path):
        return FileResponse(logo_path)
        
    # Return 204 No Content to use default logo
    return Response(status_code=204)


@router.get("/custom-app-name")
async def get_custom_app_name():
    """Get custom app name."""
    return {"customAppName": "SpeedMaint Intelligence"}


@router.get("/llm-preference")
async def get_llm_preference():
    """Get current LLM configuration."""
    return {
        "LLMProvider": "openai",
        "LLMModelPref": "gpt-4-turbo-preview",
        "EmbedderProvider": "openai",
        "EmbeddingModelPref": "text-embedding-3-small"
    }


@router.get("/embedding-preference")
async def get_embedding_preference():
    """Get current embedding configuration."""
    return {
        "EmbedderProvider": "openai",
        "EmbeddingModelPref": "text-embedding-3-small"
    }


@router.get("/vector-database")
async def get_vector_database():
    """Get vector database configuration."""
    return {
        "VectorDB": "lancedb",
        "configured": True
    }


@router.get("/welcome-messages")
async def get_welcome_messages():
    """Get welcome messages."""
    return {
        "welcomeMessages": [
            {"role": "assistant", "content": "Chào mừng bạn đến với SpeedMaint Intelligence!"},
            {"role": "assistant", "content": "Hãy upload tài liệu hoặc đặt câu hỏi để bắt đầu."}
        ]
    }


@router.get("/footer-data")
async def get_footer_data():
    """Get footer data."""
    return {"footerData": "[]"}


@router.get("/support-email")
async def get_support_email():
    """Get support email."""
    return {"supportEmail": ""}


@router.get("/can-delete-workspaces")
async def can_delete_workspaces(current_user: User = Depends(get_current_user)):
    """Check if user can delete workspaces."""
    return {"canDelete": current_user.role.value in ["admin", "manager"]}


@router.get("/accepted-document-types")
async def accepted_document_types():
    """Get accepted document types."""
    return {
        "types": {
            "application/pdf": [".pdf"],
            "text/plain": [".txt", ".md"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
            "text/csv": [".csv"],
            "image/png": [".png"],
            "image/jpeg": [".jpg", ".jpeg"]
        }
    }


@router.get("/prompt-variables")
async def get_prompt_variables():
    """
    Get all system prompt variables.
    Currently returns an empty list as it's not fully implemented.
    """
    return {"variables": []}


@router.get("/document-processing-status")
async def document_processing_status():
    """Check if document processor is online."""
    return {"online": True}


@router.get("/pfp/{user_id}")
async def get_profile_picture(user_id: int):
    """Get user profile picture from branding directory."""
    pfp_path = os.path.join("./data/branding/pfp", f"{user_id}.png")
    if os.path.exists(pfp_path):
        return FileResponse(pfp_path)
    return Response(status_code=204)


@router.get("/refresh-user")
async def refresh_user(current_user: User = Depends(get_current_user)):
    """Refresh current user session data."""
    preferences = {}
    if current_user.preferences:
        try:
            preferences = json.loads(current_user.preferences)
        except:
            preferences = {}
            
    return {
        "success": True,
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "role": current_user.role.value,
            **preferences
        },
        "message": None
    }


@router.post("/user")
async def update_user_settings(
    updates: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update current user settings (username, theme, lang, etc).
    Payload: {"username": "...", "theme": "...", "userLang": "...", "autoSubmit": "..."}
    """
    new_username = updates.get("username")
    
    # Update username if provided and different
    if new_username and new_username != current_user.username:
        # Check if username exists
        result = await db.execute(select(User).where(User.username == new_username))
        if result.scalar_one_or_none():
            return {"success": False, "error": "Username already exists"}
        current_user.username = new_username
    
    # Update preferences (theme, lang, etc)
    preferences = {}
    if current_user.preferences:
        try:
            preferences = json.loads(current_user.preferences)
        except:
            preferences = {}
            
    # Update with new values, excluding username
    for key, value in updates.items():
        if key != "username":
            preferences[key] = value
            
    current_user.preferences = json.dumps(preferences)
    
    await db.commit()
    await db.refresh(current_user)
    
    # Merge preferences back into user object for response if needed
    # (FastAPI refresh already updated current_user)
    
    return {
        "success": True,
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "role": current_user.role.value,
        },
        "message": "User settings updated"
    }


@router.get("/is-default-logo")
async def is_default_logo():
    """Check if using default logo."""
    return {"isDefaultLogo": True}


# ============= System Configuration Endpoints =============

@router.post("/update-env")
async def update_env(
    updates: dict,
    current_user: User = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Update system environment/settings.
    In AnythingLLM: updates LLM provider, embedding engine, etc.
    For SpeedMaint: stores settings in SystemSettings table.
    """
    from app.models import SystemSettings
    
    changed_settings = {}
    
    for key, value in updates.items():
        result = await db.execute(
            select(SystemSettings).where(SystemSettings.key == key)
        )
        setting = result.scalar_one_or_none()
        
        if setting:
            setting.value = str(value) if value is not None else None
        else:
            new_setting = SystemSettings(key=key, value=str(value) if value is not None else None)
            db.add(new_setting)
        
        changed_settings[key] = value
    
    await db.commit()
    
    return {
        "newValues": changed_settings,
        "error": None
    }


@router.post("/update-password")
async def update_password(
    current_password: str = None,
    new_password: str = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update user password."""
    from app.services.auth_service import verify_password, get_password_hash
    
    # Verify current password
    if not verify_password(current_password, current_user.password_hash):
        return {"success": False, "error": "Current password is incorrect"}
    
    if len(new_password) < 8:
        return {"success": False, "error": "New password must be at least 8 characters"}
    
    # Update password
    current_user.password_hash = get_password_hash(new_password)
    await db.commit()
    
    return {"success": True, "error": None}


@router.get("/local-files")
async def get_local_files(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get local files tree for document browser.
    Returns folder structure with documents.
    """
    import os
    from app.config import get_settings
    from app.models import Workspace, WorkspaceUser
    
    settings = get_settings()
    upload_dir = settings.upload_dir
    
    # Lấy danh sách slug các workspace mà user có quyền truy cập
    if current_user.role.value == "admin":
        result = await db.execute(select(Workspace.slug))
    else:
        query = select(Workspace.slug).join(
            WorkspaceUser, Workspace.id == WorkspaceUser.workspace_id, isouter=True
        ).where(
            (Workspace.owner_id == current_user.id) | (WorkspaceUser.user_id == current_user.id)
        ).distinct()
        result = await db.execute(query)
        
    accessible_slugs = [row[0] for row in result.all()]
    
    def scan_directory(path, rel_path="", is_top_level=True):
        """Recursively scan directory. Filter at top level for security."""
        items = []
        
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            return items
        
        for entry in os.scandir(path):
            # Ở level ngoài cùng (top-level), chỉ cho phép các folder khớp với slug của workspace user được truy cập
            if is_top_level and entry.is_dir() and entry.name not in accessible_slugs:
                continue
                
            item_rel_path = os.path.join(rel_path, entry.name) if rel_path else entry.name
            
            if entry.is_dir():
                items.append({
                    "name": entry.name,
                    "type": "folder",
                    "items": scan_directory(entry.path, item_rel_path, is_top_level=False)
                })
            else:
                stat = entry.stat()
                name_without_ext = os.path.splitext(entry.name)[0]
                items.append({
                    "name": entry.name,
                    "title": name_without_ext,  # Frontend uses title for display
                    "type": "file",
                    "id": item_rel_path,
                    "url": f"/documents/{item_rel_path}",
                    "size": stat.st_size,
                    "cached": False,
                    "pinnedWorkspaces": [],
                    "published": stat.st_mtime,  # For date display
                    "token_count_estimate": int(stat.st_size / 4)  # Rough estimate
                })
        
        return items
    
    return {
        "localFiles": {
            "name": "documents",
            "type": "folder",
            "items": scan_directory(upload_dir, is_top_level=True)
        }
    }


@router.delete("/remove-document")
async def remove_document(
    name: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Remove a single document."""
    import os
    from app.config import get_settings
    from app.models import Document
    from app.services.rag_service import get_rag_client
    from sqlalchemy import select
    
    settings = get_settings()
    file_path = os.path.join(settings.upload_dir, name)
    
    if os.path.exists(file_path):
        parts = name.split('/')
        if len(parts) >= 2:
            slug = parts[0]
            filename = parts[-1]
            try:
                # 1. Notify RAG service
                rag_client = await get_rag_client()
                await rag_client.delete_document_data(slug, filename)
                
                # 2. Clean SQLite DB
                result = await db.execute(select(Document).where(Document.file_path.contains(filename)))
                doc = result.scalar_one_or_none()
                if doc:
                    await db.delete(doc)
                    await db.commit()
            except Exception as e:
                print(f"Warning: Failed to clean up document {name}: {e}")
                
        if os.path.isfile(file_path):
            os.remove(file_path)
            return True
            
    return False


@router.delete("/remove-documents")
async def remove_documents(
    names: list,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Remove multiple documents."""
    import os
    from app.config import get_settings
    from app.models import Document
    from app.services.rag_service import get_rag_client
    from sqlalchemy import select
    
    settings = get_settings()
    removed = 0
    
    for name in names:
        file_path = os.path.join(settings.upload_dir, name)
        if os.path.exists(file_path):
            parts = name.split('/')
            if len(parts) >= 2:
                slug = parts[0]
                filename = parts[-1]
                try:
                    rag_client = await get_rag_client()
                    await rag_client.delete_document_data(slug, filename)
                    
                    result = await db.execute(select(Document).where(Document.file_path.contains(filename)))
                    doc = result.scalar_one_or_none()
                    if doc:
                        await db.delete(doc)
                        await db.commit()
                except Exception as e:
                    print(f"Warning: Failed to clean up document {name}: {e}")
                    
            if os.path.isfile(file_path):
                os.remove(file_path)
                removed += 1
                
    return removed > 0


@router.delete("/remove-folder")
async def remove_folder(
    name: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Remove a folder and its contents."""
    import os
    import shutil
    from app.config import get_settings
    from app.models import Document, Workspace
    from app.services.rag_service import get_rag_client
    from sqlalchemy import select
    
    settings = get_settings()
    folder_path = os.path.join(settings.upload_dir, name)
    
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Notify RAG service to purge the entire workspace data because name is slug
        slug = name.replace("/", "")
        try:
            rag_client = await get_rag_client()
            await rag_client.delete_workspace_data(slug)
            
            # Clean SQLite DB documents matching this workspace
            result = await db.execute(select(Workspace).where(Workspace.slug == slug))
            ws = result.scalar_one_or_none()
            if ws:
                docs_result = await db.execute(select(Document).where(Document.workspace_id == ws.id))
                for doc in docs_result.scalars().all():
                    await db.delete(doc)
                await db.commit()
        except Exception as e:
            print(f"Warning: Failed to clean up folder area {name}: {e}")

        shutil.rmtree(folder_path)
        return True
    
    return False
