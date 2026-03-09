"""
SpeedMaint Intelligence API Gateway

API adapter that bridges AnythingLLM frontend with RAGAnything backend.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.config import get_settings
from app.database import init_db
from app.routers import auth, workspaces, system, documents, chat, users, recovery, threads, invites, api_keys, branding, history


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    # Startup
    print("🚀 Starting SpeedMaint API Gateway...")
    
    # Create data directory
    os.makedirs("./data", exist_ok=True)
    os.makedirs(settings.upload_dir, exist_ok=True)
    
    # Initialize database
    await init_db()
    print("✅ Database initialized")
    
    # Run workspace migration
    from app.database import async_session_maker
    from app.models import Workspace, WorkspaceUser
    from sqlalchemy import select
    
    async with async_session_maker() as db:
        result = await db.execute(select(Workspace))
        workspaces = result.scalars().all()
        
        migrated_count = 0
        for ws in workspaces:
            if ws.owner_id:
                existing = await db.execute(
                    select(WorkspaceUser).where(
                        WorkspaceUser.user_id == ws.owner_id,
                        WorkspaceUser.workspace_id == ws.id
                    )
                )
                if not existing.scalar_one_or_none():
                    new_member = WorkspaceUser(
                        user_id=ws.owner_id,
                        workspace_id=ws.id
                    )
                    db.add(new_member)
                    migrated_count += 1
        
        if migrated_count > 0:
            await db.commit()
            print(f"✅ Migrated {migrated_count} workspace ownerships to workspace_users table.")
    
    yield
    
    # Shutdown
    print("👋 Shutting down SpeedMaint API Gateway...")


# Create FastAPI app
app = FastAPI(
    title="SpeedMaint Intelligence API Gateway",
    description="API adapter for SpeedMaint Intelligence RAG system",
    version="1.0.0",
    lifespan=lifespan
)

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request

from fastapi.encoders import jsonable_encoder

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    import json
    from fastapi.encoders import jsonable_encoder
    
    # Sanitize errors: convert bytes to strings for JSON serialization
    def sanitize(obj):
        if isinstance(obj, list):
            return [sanitize(i) for i in obj]
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, bytes):
            return obj.decode(errors='replace')
        return obj
    
    errors = sanitize(exc.errors())
    print(f"❌ Validation Error: {errors}")
    
    try:
        body = await request.json()
        print(f"❌ Request Body: {json.dumps(body, indent=2)}")
    except:
        pass
        
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": errors}),
    )

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(auth.router, prefix="/api")
app.include_router(workspaces.router, prefix="/api")
app.include_router(system.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(users.router, prefix="/api")
app.include_router(recovery.router, prefix="/api")
app.include_router(threads.router, prefix="/api")
app.include_router(invites.router, prefix="/api")
app.include_router(api_keys.router, prefix="/api")
app.include_router(branding.router, prefix="/api")
app.include_router(history.router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "SpeedMaint Intelligence API Gateway",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api/health")
async def health():
    """Health check."""
    return {"status": "ok"}


@app.get("/api/ping")
async def ping():
    """Ping endpoint for frontend health check."""
    return {"online": True}


@app.get("/api/utils/metrics")
async def utils_metrics():
    """Metrics endpoint for frontend version check."""
    return {
        "appVersion": "1.0.0",
        "vectorCount": 0,
        "online": True
    }


@app.get("/api/setup-complete")
async def setup_complete_root():
    """
    Root-level setup-complete for AnythingLLM frontend.
    Returns system configuration status.
    """
    from app.database import async_session_maker
    from app.models import User, UserRole
    from sqlalchemy import select
    
    async with async_session_maker() as db:
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


# AnythingLLM-compatible login endpoint
from fastapi import Request
import json


@app.post("/api/request-token")
async def request_token(request: Request):
    """
    AnythingLLM-compatible login endpoint.
    Frontend sends POST with username/password.
    """
    from app.database import async_session_maker
    from app.services.auth_service import authenticate_user, create_access_token
    from datetime import timedelta
    
    # Parse body - handle both JSON object and string
    try:
        body = await request.json()
    except:
        try:
            raw = await request.body()
            body = json.loads(raw.decode())
        except:
            return {"valid": False, "message": "Invalid request body"}
    
    username = body.get("username", "")
    password = body.get("password", "")
    
    if not username or not password:
        return {"valid": False, "message": "Username and password required"}
    
    async with async_session_maker() as db:
        user = await authenticate_user(db, username, password)
        if not user:
            return {"valid": False, "message": "Invalid credentials"}
        
        if not user.is_active:
            return {"valid": False, "message": "Tài khoản của bạn đã bị khóa. Vui lòng liên hệ quản trị viên."}
        
        access_token = create_access_token(
            data={"sub": user.username, "user_id": user.id},
            expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
        )
        
        return {
            "valid": True,
            "user": {
                "id": user.id,
                "username": user.username,
                "role": user.role.value,
            },
            "token": access_token,
            "message": None
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
