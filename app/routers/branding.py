"""
Branding router - Customize system appearance.
"""
import os
import shutil
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.database import get_db
from app.models import User, SystemSettings
from app.services.auth_service import get_current_user, get_current_active_admin
from app.config import get_settings


router = APIRouter(prefix="/system", tags=["System - Branding"])


# Storage paths
BRANDING_DIR = "./data/branding"
LOGO_PATH = os.path.join(BRANDING_DIR, "logo.png")
PFP_DIR = os.path.join(BRANDING_DIR, "pfp")


def ensure_dirs():
    """Ensure branding directories exist."""
    os.makedirs(BRANDING_DIR, exist_ok=True)
    os.makedirs(PFP_DIR, exist_ok=True)


# Schemas
class WelcomeMessagesUpdate(BaseModel):
    """Welcome messages update request."""
    messages: list


class SupportEmailUpdate(BaseModel):
    """Support email update request."""
    email: str


class FooterDataUpdate(BaseModel):
    """Footer data update request."""
    footerData: str


# ============= Logo Endpoints =============

@router.post("/upload-logo")
async def upload_logo(
    logo: UploadFile = File(...),
    current_user: User = Depends(get_current_active_admin)
):
    """Upload a custom logo (admin only)."""
    ensure_dirs()
    
    # Validate file type
    if not logo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save file
    content = await logo.read()
    with open(LOGO_PATH, "wb") as f:
        f.write(content)
    
    return {"success": True, "message": "Logo uploaded"}


@router.delete("/remove-logo")
async def remove_logo(
    current_user: User = Depends(get_current_active_admin)
):
    """Remove custom logo (admin only)."""
    if os.path.exists(LOGO_PATH):
        os.remove(LOGO_PATH)
    
    return {"success": True, "message": "Logo removed"}


@router.get("/has-custom-logo")
async def has_custom_logo():
    """Check if a custom logo exists."""
    return {"hasCustomLogo": os.path.exists(LOGO_PATH)}


# ============= Profile Picture Endpoints =============

@router.post("/upload-pfp")
async def upload_profile_picture(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload user profile picture."""
    ensure_dirs()
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save file
    user_pfp_path = os.path.join(PFP_DIR, f"{current_user.id}.png")
    content = await file.read()
    with open(user_pfp_path, "wb") as f:
        f.write(content)
    
    return {"success": True, "message": "Profile picture uploaded"}


@router.delete("/remove-pfp")
async def remove_profile_picture(
    current_user: User = Depends(get_current_user)
):
    """Remove user profile picture."""
    user_pfp_path = os.path.join(PFP_DIR, f"{current_user.id}.png")
    if os.path.exists(user_pfp_path):
        os.remove(user_pfp_path)
    
    return {"success": True, "message": "Profile picture removed"}


# ============= Welcome Messages =============

@router.get("/welcome-messages")
async def get_welcome_messages(
    db: AsyncSession = Depends(get_db)
):
    """Get welcome messages."""
    result = await db.execute(
        select(SystemSettings).where(SystemSettings.key == "welcome_messages")
    )
    setting = result.scalar_one_or_none()
    
    if setting and setting.value:
        import json
        try:
            messages = json.loads(setting.value)
        except:
            messages = []
    else:
        # Default messages
        messages = [
            {"role": "assistant", "content": "Chào mừng bạn đến với SpeedMaint Intelligence!"},
            {"role": "assistant", "content": "Hãy upload tài liệu hoặc đặt câu hỏi để bắt đầu."}
        ]
    
    return {"welcomeMessages": messages}


@router.post("/set-welcome-messages")
async def set_welcome_messages(
    body: WelcomeMessagesUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Set welcome messages (admin only)."""
    import json
    
    result = await db.execute(
        select(SystemSettings).where(SystemSettings.key == "welcome_messages")
    )
    setting = result.scalar_one_or_none()
    
    if setting:
        setting.value = json.dumps(body.messages)
    else:
        setting = SystemSettings(key="welcome_messages", value=json.dumps(body.messages))
        db.add(setting)
    
    await db.commit()
    
    return {"success": True, "welcomeMessages": body.messages}


# ============= Footer Data =============

@router.get("/footer-data")
async def get_footer_data(
    db: AsyncSession = Depends(get_db)
):
    """Get footer data."""
    result = await db.execute(
        select(SystemSettings).where(SystemSettings.key == "footer_data")
    )
    setting = result.scalar_one_or_none()
    
    return {"footerData": setting.value if setting else "[]"}


@router.post("/set-footer-data")
async def set_footer_data(
    body: FooterDataUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Set footer data (admin only)."""
    result = await db.execute(
        select(SystemSettings).where(SystemSettings.key == "footer_data")
    )
    setting = result.scalar_one_or_none()
    
    if setting:
        setting.value = body.footerData
    else:
        setting = SystemSettings(key="footer_data", value=body.footerData)
        db.add(setting)
    
    await db.commit()
    
    return {"success": True, "footerData": body.footerData}


# ============= Support Email =============

@router.get("/support-email")
async def get_support_email(
    db: AsyncSession = Depends(get_db)
):
    """Get support email."""
    result = await db.execute(
        select(SystemSettings).where(SystemSettings.key == "support_email")
    )
    setting = result.scalar_one_or_none()
    
    return {"supportEmail": setting.value if setting else ""}


@router.post("/set-support-email")
async def set_support_email(
    body: SupportEmailUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Set support email (admin only)."""
    result = await db.execute(
        select(SystemSettings).where(SystemSettings.key == "support_email")
    )
    setting = result.scalar_one_or_none()
    
    if setting:
        setting.value = body.email
    else:
        setting = SystemSettings(key="support_email", value=body.email)
        db.add(setting)
    
    await db.commit()
    
    return {"success": True, "supportEmail": body.email}


# ============= Custom App Name =============

@router.get("/custom-app-name")
async def get_custom_app_name(
    db: AsyncSession = Depends(get_db)
):
    """Get custom app name."""
    result = await db.execute(
        select(SystemSettings).where(SystemSettings.key == "custom_app_name")
    )
    setting = result.scalar_one_or_none()
    
    return {"customAppName": setting.value if setting else "SpeedMaint Intelligence"}


@router.post("/set-custom-app-name")
async def set_custom_app_name(
    body: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """Set custom app name (admin only)."""
    name = body.get("name", "SpeedMaint Intelligence")
    
    result = await db.execute(
        select(SystemSettings).where(SystemSettings.key == "custom_app_name")
    )
    setting = result.scalar_one_or_none()
    
    if setting:
        setting.value = name
    else:
        setting = SystemSettings(key="custom_app_name", value=name)
        db.add(setting)
    
    await db.commit()
    
    return {"success": True, "customAppName": name}
