"""
Documents router - Upload and manage documents.
"""
import os
import uuid
import shutil
import asyncio
from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import User, Workspace, Document, DocumentStatus
from app.schemas import DocumentResponse
from app.services.auth_service import get_current_user, has_workspace_access
from app.services.rag_service import get_rag_client, RAGServiceClient
from app.config import get_settings

settings = get_settings()

router = APIRouter(tags=["Documents"])


async def process_document_background(
    document_id: int,
    file_path: str,
    workspace_slug: str,
    db_url: str
):
    """Background task to process document through RAG."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as db:
        # Get document
        result = await db.execute(select(Document).where(Document.id == document_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return
        
        try:
            # Update status to processing
            doc.status = DocumentStatus.PROCESSING
            await db.commit()
            
            # Process through RAG service
            rag_client = RAGServiceClient()
            await rag_client.process_document(
                file_path=file_path,
                workspace_slug=workspace_slug
            )
            
            # Update status to completed
            doc.status = DocumentStatus.COMPLETED
            doc.processed_at = datetime.utcnow()
            await db.commit()
            
        except Exception as e:
            # Update status to failed
            doc.status = DocumentStatus.FAILED
            doc.error_message = str(e)
            await db.commit()


@router.post("/document/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generic document upload (AnythingLLM style).
    Just saves file and returns metadata. Not linked to workspace yet.
    """
    settings = get_settings()
    
    # Ensure upload dir exists
    if not os.path.exists(settings.upload_dir):
        os.makedirs(settings.upload_dir, exist_ok=True)
        
    # Generate filename
    # AnythingLLM keeps original structure usually, but we'll simple save
    file_path = os.path.join(settings.upload_dir, file.filename)
    
    # Handle duplicate
    if os.path.exists(file_path):
        base, ext = os.path.splitext(file.filename)
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(settings.upload_dir, f"{base}-{counter}{ext}")
            counter += 1
            
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Return match AnythingLLM response
    return {
        "success": True,
        "error": None,
        "documents": [
            {
                "id": str(uuid.uuid4()), # Temporary ID
                "location": file_path,
                "name": os.path.basename(file_path),
                "originalName": file.filename,
                "title": os.path.splitext(file.filename)[0],
                "metadata": {}
            }
        ]
    }


@router.post("/workspace/{slug}/upload")
async def upload_document_to_workspace(
    slug: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload document to a specific workspace.
    Matches AnythingLLM-style upload per workspace.
    """
    # Check workspace exists
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Check access
    if not await has_workspace_access(db, current_user, workspace.id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    settings = get_settings()
    
    # Create workspace-specific upload directory
    workspace_upload_dir = os.path.join(settings.upload_dir, slug)
    if not os.path.exists(workspace_upload_dir):
        os.makedirs(workspace_upload_dir, exist_ok=True)
        
    # Generate filename
    file_path = os.path.join(workspace_upload_dir, file.filename)
    
    # Handle duplicate
    if os.path.exists(file_path):
        base, ext = os.path.splitext(file.filename)
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(workspace_upload_dir, f"{base}-{counter}{ext}")
            counter += 1
            
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Background: forward sang rag-service để index
    asyncio.create_task(_index_file_to_rag(file_path, slug))

    return {
        "success": True,
        "error": None,
        "documents": [
            {
                "id": str(uuid.uuid4()),
                "location": file_path,
                "name": os.path.basename(file_path),
                "originalName": file.filename,
                "title": os.path.splitext(file.filename)[0],
                "metadata": {}
            }
        ]
    }


async def _index_file_to_rag(file_path: str, workspace_slug: str):
    """Extract text từ file và gửi sang rag-service để index."""
    import httpx
    from app.config import get_settings
    cfg = get_settings()

    try:
        filename = os.path.basename(file_path)
        rag_url = cfg.rag_service_url
        print(f"[INDEX] Sending {filename} to rag-service for workflow OCR -> RAG index")

        async with httpx.AsyncClient(timeout=12000.0) as client:
            with open(file_path, "rb") as f:
                files = {"file": (filename, f, "application/pdf")}
                data = {"workspace": workspace_slug, "ocr_model": "deepseek"}
                resp = await client.post(f"{rag_url}/api/v1/ingest/upload", files=files, data=data)
            
            print(f"[INDEX] RAG response: {resp.status_code} - {resp.text[:200]}")

    except Exception as e:
        print(f"[INDEX] Error indexing {file_path}: {e}")



@router.get("/workspace/{slug}/documents", response_model=List[DocumentResponse])
async def list_documents(
    slug: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all documents in a workspace."""
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Check access
    if not await has_workspace_access(db, current_user, workspace.id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    result = await db.execute(
        select(Document).where(Document.workspace_id == workspace.id)
    )
    documents = result.scalars().all()
    
    return documents


@router.delete("/workspace/{slug}/documents/{doc_id}")
async def delete_document(
    slug: str,
    doc_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a document from workspace."""
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Check access
    if not await has_workspace_access(db, current_user, workspace.id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    result = await db.execute(
        select(Document).where(
            Document.id == doc_id,
            Document.workspace_id == workspace.id
        )
    )
    doc = result.scalar_one_or_none()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Call RAG service to clean up data
    try:
        rag_client = await get_rag_client()
        await rag_client.delete_document_data(slug, doc.filename)
    except Exception as e:
        print(f"Warning: Failed to clean up RAG data for {doc.filename}: {e}")
    
    # Delete file
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)
    
    # Delete record
    await db.delete(doc)
    await db.commit()
    
    return {"success": True, "message": "Document deleted"}


@router.post("/workspace/{slug}/upload-link")
async def upload_link(
    slug: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload a link for processing (Stub).
    Matches AnythingLLM's link ingestion.
    """
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
        
    link = body.get("link", "")
    if not link:
        raise HTTPException(status_code=400, detail="Link is required")
        
    # In a real implementation, we would call a crawler or the RAG service
    return {"success": True, "error": None}
