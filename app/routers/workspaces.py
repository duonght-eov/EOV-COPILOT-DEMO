"""
Workspaces router - CRUD operations for workspaces.
"""
import re
import os
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database import get_db
from app.models import User, Workspace, Document, DocumentStatus
from app.schemas import WorkspaceCreate, WorkspaceUpdate, WorkspaceResponse
from app.services.auth_service import get_current_user
from app.config import get_settings



router = APIRouter(tags=["Workspaces"])


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    return text


@router.get("/workspaces")
async def list_workspaces(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all workspaces for current user."""
    # Admin sees all, others see only their own (where they are members or owners)
    if current_user.role.value == "admin":
        query = select(Workspace)
    else:
        from app.models import WorkspaceUser
        query = select(Workspace).join(
            WorkspaceUser, Workspace.id == WorkspaceUser.workspace_id
        ).where(
            (Workspace.owner_id == current_user.id) | (WorkspaceUser.user_id == current_user.id)
        ).distinct()
    
    result = await db.execute(query)
    workspaces = result.scalars().all()
    
    # Get document counts
    response = []
    for ws in workspaces:
        doc_count = await db.execute(
            select(func.count(Document.id)).where(Document.workspace_id == ws.id)
        )
        ws_dict = {
            "id": ws.id,
            "name": ws.name,
            "slug": ws.slug,
            "description": ws.description,
            "llm_provider": ws.llm_provider,
            "llm_model": ws.llm_model,
            "query_mode": ws.query_mode,
            "is_predict_enabled": ws.is_predict_enabled,
            "predict_llm_model": ws.predict_llm_model,
            "created_at": ws.created_at.isoformat() if ws.created_at else None,
            "document_count": doc_count.scalar() or 0
        }
        response.append(ws_dict)
    
    # Frontend expects {workspaces: [...]}
    return {"workspaces": response}


@router.post("/workspace/new")
async def create_workspace(
    workspace: WorkspaceCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new workspace."""
    # Only Admin and Manager can create workspaces
    if current_user.role.value not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only Admins and Managers can create workspaces"
        )
    
    slug = slugify(workspace.name)
    
    # Check slug uniqueness
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    existing = result.scalar_one_or_none()
    if existing:
        # Append number to make unique
        counter = 1
        while existing:
            new_slug = f"{slug}-{counter}"
            result = await db.execute(select(Workspace).where(Workspace.slug == new_slug))
            existing = result.scalar_one_or_none()
            if not existing:
                slug = new_slug
                break
            counter += 1
    
    ws = Workspace(
        name=workspace.name,
        slug=slug,
        description=workspace.description,
        owner_id=current_user.id
    )
    db.add(ws)
    await db.commit()
    await db.refresh(ws)
    
    # Also add owner as a member in WorkspaceUser table
    from app.models import WorkspaceUser
    member = WorkspaceUser(user_id=current_user.id, workspace_id=ws.id)
    db.add(member)
    await db.commit()
    
    # Return in format frontend expects: {workspace: {...}, message: null}
    return {
        "workspace": {
            "id": ws.id,
            "name": ws.name,
            "slug": ws.slug,
            "description": ws.description,
            "llm_provider": ws.llm_provider,
            "llm_model": ws.llm_model,
            "query_mode": ws.query_mode,
            "is_predict_enabled": ws.is_predict_enabled,
            "predict_llm_model": ws.predict_llm_model,
            "created_at": ws.created_at.isoformat() if ws.created_at else None,
            "document_count": 0
        },
        "message": None
    }


@router.get("/workspace/{slug}")
async def get_workspace(
    slug: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get workspace by slug."""
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    ws = result.scalar_one_or_none()
    
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Check access: Admin, Owner, or Member
    if current_user.role.value != "admin" and ws.owner_id != current_user.id:
        from app.models import WorkspaceUser
        member_check = await db.execute(
            select(WorkspaceUser).where(
                WorkspaceUser.workspace_id == ws.id,
                WorkspaceUser.user_id == current_user.id
            )
        )
        if not member_check.scalar_one_or_none():
            raise HTTPException(status_code=403, detail="Access denied")
    
    # Get documents for this workspace
    docs_result = await db.execute(
        select(Document).where(Document.workspace_id == ws.id)
    )
    docs = docs_result.scalars().all()
    
    # Build documents array with docpath for frontend
    documents = [
        {
            "id": doc.id,
            "docpath": doc.file_path,  # Frontend expects this for filtering
            "name": doc.filename,
            "status": doc.status.value if doc.status else "pending"
        }
        for doc in docs
    ]
    
    # Frontend expects {workspace: {...}}
    return {
        "workspace": {
            "id": ws.id,
            "name": ws.name,
            "slug": ws.slug,
            "description": ws.description,
            "llm_provider": ws.llm_provider,
            "llm_model": ws.llm_model,
            "query_mode": ws.query_mode,
            "is_predict_enabled": ws.is_predict_enabled,
            "predict_llm_model": ws.predict_llm_model,
            "created_at": ws.created_at.isoformat() if ws.created_at else None,
            "document_count": len(documents),
            "documents": documents  # Add documents array for frontend
        }
    }


@router.put("/workspace/{slug}")
@router.post("/workspace/{slug}/update")
async def update_workspace(
    slug: str,
    update_data: WorkspaceUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update workspace settings."""
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    ws = result.scalar_one_or_none()
    
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Check access
    if current_user.role.value != "admin" and ws.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Update fields
    for field, value in update_data.model_dump(exclude_unset=True).items():
        setattr(ws, field, value)
    
    await db.commit()
    await db.refresh(ws)
    
    doc_count = await db.execute(
        select(func.count(Document.id)).where(Document.workspace_id == ws.id)
    )
    
    # Frontend expects {workspace: {...}, message: null}
    return {
        "workspace": {
            "id": ws.id,
            "name": ws.name,
            "slug": ws.slug,
            "description": ws.description,
            "llm_provider": ws.llm_provider,
            "llm_model": ws.llm_model,
            "query_mode": ws.query_mode,
            "is_predict_enabled": ws.is_predict_enabled,
            "predict_llm_model": ws.predict_llm_model,
            "created_at": ws.created_at.isoformat() if ws.created_at else None,
            "document_count": doc_count.scalar() or 0
        },
        "message": None
    }


@router.get("/workspace/{slug}/connectors")
async def get_workspace_connectors(
    slug: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the active API connector parameters for this workspace."""
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    ws = result.scalar_one_or_none()
    
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
        
    from app.models.models import WorkspaceConnector
    conn_result = await db.execute(select(WorkspaceConnector).where(WorkspaceConnector.workspace_id == ws.id))
    connector = conn_result.scalar_one_or_none()
    
    if not connector:
        # returns empty dict or 404 setup based on frontend, returning empty object is safer
        return {"connector": None}
        
    return {
        "connector": {
            "id": connector.id,
            "base_url": connector.base_url,
            "auth_type": connector.auth_type,
            "auth_credentials": connector.auth_credentials,
            "custom_headers": connector.custom_headers
        }
    }


@router.post("/workspace/{slug}/connectors")
async def update_workspace_connectors(
    slug: str,
    update_data: WorkspaceUpdate, # We can reuse or properly import the connector update schema, let's just make it dict dynamically
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upsert the active API connector parameters for this workspace."""
    body = await request.json()
    
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    ws = result.scalar_one_or_none()
    
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Update predict enabled while we are at it
    if "is_predict_enabled" in body:
        ws.is_predict_enabled = body.get("is_predict_enabled", False)
        
    from app.models.models import WorkspaceConnector
    conn_result = await db.execute(select(WorkspaceConnector).where(WorkspaceConnector.workspace_id == ws.id))
    connector = conn_result.scalar_one_or_none()
    
    if connector:
        # Update
        if "base_url" in body: connector.base_url = body["base_url"]
        if "auth_type" in body: connector.auth_type = body["auth_type"]
        if "auth_credentials" in body: connector.auth_credentials = body["auth_credentials"]
        db.add(connector)
    else:
        # Create
        if "base_url" in body and body["base_url"]:
            new_conn = WorkspaceConnector(
                workspace_id=ws.id,
                base_url=body["base_url"],
                auth_type=body.get("auth_type", "bearer"),
                auth_credentials=body.get("auth_credentials", ""),
                custom_headers="{}"
            )
            db.add(new_conn)
            
    await db.commit()
    return {"success": True, "message": "Connector settings updated"}


@router.post("/workspace/{slug}/update-embeddings")
async def update_embeddings(
    slug: str,
    request: dict,  # {"adds": ["path/to/doc"], "deletes": []}
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        """
        Update workspace embeddings (Add documents).
        AnythingLLM sends { adds: [], deletes: [] }
        """
        result = await db.execute(select(Workspace).where(Workspace.slug == slug))
        ws = result.scalar_one_or_none()
        
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found")
            
        adds = request.get("adds", [])
        deletes = request.get("deletes", [])
        
        # Check access
        if current_user.role.value != "admin" and ws.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
            
        from app.services.rag_service import rag_client as rag_service
        
        # Process updates in RAG service
        settings = get_settings()
        processed_count = 0
        
        for file_path in adds:
            # Normalize relative path
            if file_path.startswith("/"):
                file_path = file_path[1:]
                
            # Resolve full path for RAG service - MUST be absolute path
            # since RAG service runs from a different working directory
            full_path = file_path
            if not os.path.isabs(full_path):
                # Try as is first
                if os.path.exists(full_path):
                    full_path = os.path.abspath(full_path)
                else:
                    # Try under upload_dir
                    potential_path = os.path.join(settings.upload_dir, file_path)
                    if os.path.exists(potential_path):
                        full_path = os.path.abspath(potential_path)
                    else:
                        # Try just basename under upload_dir
                        basename = os.path.basename(file_path)
                        # Extract workspace dir from file_path if present
                        parts = file_path.split("/")
                        for i in range(len(parts)):
                            rebuild_path = os.path.join(settings.upload_dir, *parts[i:])
                            if os.path.exists(rebuild_path):
                                full_path = os.path.abspath(rebuild_path)
                                break
                        else:
                            # Default fallback
                            full_path = os.path.abspath(file_path)
            
            # Check if it exists in DB
            doc_result = await db.execute(select(Document).where(Document.file_path == file_path))
            doc = doc_result.scalar_one_or_none()
            
            # If not in DB, create it
            if not doc and os.path.exists(full_path):
                file_size = os.path.getsize(full_path)
                filename = os.path.basename(full_path)
                
                doc = Document(
                    filename=filename,
                    original_filename=filename,
                    file_path=file_path, # Keep store relative path
                    file_size=file_size,
                    mime_type="application/octet-stream",
                    status=DocumentStatus.PENDING,
                    workspace_id=ws.id
                )
                db.add(doc)
                await db.commit()
                await db.refresh(doc)
            elif doc:
                # Link existing doc
                doc.workspace_id = ws.id
                if doc.status == DocumentStatus.FAILED:
                     doc.status = DocumentStatus.PENDING
                db.add(doc) # Ensure it's in session
            else:
                print(f"File not found: {file_path}")
                continue
                
            # Call RAG Service to process document
            try:
                # Update status
                doc.status = DocumentStatus.PROCESSING
                await db.commit()
                
                await rag_service.process_document(
                    file_path=full_path,
                    workspace_slug=slug
                )
                
                doc.status = DocumentStatus.COMPLETED
                processed_count += 1
            except Exception as e:
                print(f"RAG processing failed for {full_path}: {e}")
                doc.status = DocumentStatus.FAILED
                # doc.error = str(e)
            
            await db.commit()
        
        # Process deletes - unlink documents from workspace
        for file_path in deletes:
            # Normalize path format - match how it's stored
            if file_path.startswith("/"):
                file_path = file_path[1:]
            
            # Find document by file_path or filename pattern
            # Frontend sends format like "folder_name/filename"
            filename = os.path.basename(file_path)
            
            # Try to find by exact file_path first
            doc_result = await db.execute(
                select(Document).where(
                    Document.workspace_id == ws.id,
                    Document.file_path.contains(filename)
                )
            )
            doc = doc_result.scalar_one_or_none()
            
            if not doc:
                # Try by filename
                doc_result = await db.execute(
                    select(Document).where(
                        Document.workspace_id == ws.id,
                        Document.filename == filename
                    )
                )
                doc = doc_result.scalar_one_or_none()
            
            if doc:
                # Notify RAG service to clean up data for this document
                from app.services.rag_service import get_rag_client
                try:
                    rag_client = await get_rag_client()
                    await rag_client.delete_document_data(ws.slug, filename)
                except Exception as e:
                    print(f"Warning: Failed to clean up RAG data for {filename}: {e}")

                # Unlink document from workspace (don't delete the file)
                doc.workspace_id = None
                doc.status = DocumentStatus.PENDING
                await db.commit()
                print(f"Unlinked document {filename} from workspace {slug}")
            else:
                print(f"Document not found for deletion: {file_path}")
                
        # Check if any document failed
        failed_docs = []
        for file_path in adds:
            if file_path.startswith("/"):
                file_path = file_path[1:]
            doc_result = await db.execute(
                select(Document).where(
                    Document.file_path.contains(os.path.basename(file_path)),
                    Document.workspace_id == ws.id,
                    Document.status == DocumentStatus.FAILED
                )
            )
            failed_doc = doc_result.scalar_one_or_none()
            if failed_doc:
                failed_docs.append(os.path.basename(file_path))
        
        if failed_docs:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to process documents: {', '.join(failed_docs)}"
            )
        
        return {"workspace": {"slug": ws.slug}, "message": None}
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing documents: {str(e)}"
        )


@router.delete("/workspace/{slug}/delete-embeddings")
@router.delete("/workspace/{slug}/reset-vector-db")
async def delete_embeddings(
    slug: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Clear all embeddings/documents for a workspace."""
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    ws = result.scalar_one_or_none()
    
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Check access
    if current_user.role.value != "admin" and ws.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
        
    # Unlink/Delete all documents for this workspace
    # In AnythingLLM this also clears the vector database namespace
    from sqlalchemy import update
    await db.execute(
        update(Document)
        .where(Document.workspace_id == ws.id)
        .values(workspace_id=None)
    )
    
    # Optional: Call RAG service to clear namespace
    try:
        from app.services.rag_service import rag_client
        # await rag_client.reset_workspace(slug)
    except:
        pass
        
    await db.commit()
    return {"success": True}


@router.delete("/workspace/{slug}")
async def delete_workspace(
    slug: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a workspace."""
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    ws = result.scalar_one_or_none()
    
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Check access
    if current_user.role.value != "admin" and ws.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    
    # Notify RAG service to purge actual data
    from app.services.rag_service import rag_client
    await rag_client.delete_workspace_data(ws.slug)
    
    # Clean up physical documents directory 
    import os
    import shutil
    from app.config import get_settings
    settings = get_settings()
    workspace_upload_dir = os.path.join(settings.upload_dir, ws.slug)
    if os.path.exists(workspace_upload_dir) and os.path.isdir(workspace_upload_dir):
        try:
            shutil.rmtree(workspace_upload_dir)
        except Exception as e:
            print(f"Warning: Failed to delete workspace directory {workspace_upload_dir}: {e}")
            
    await db.delete(ws)
    await db.commit()
    
    return {"success": True, "message": f"Workspace '{slug}' deleted"}


@router.get("/workspace/{slug}/suggested-messages")
async def get_suggested_messages(
    slug: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get suggested messages for a workspace - stub for frontend compatibility."""
    # Verify workspace exists
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    ws = result.scalar_one_or_none()
    
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    return {"suggestedMessages": []}


@router.get("/workspace/{slug}/pfp")
async def get_workspace_pfp(
    slug: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get workspace profile picture - returns 204 for no custom picture."""
    from fastapi import Response
    
    # Verify workspace exists
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    ws = result.scalar_one_or_none()
    
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # No custom workspace pictures yet
    return Response(status_code=204)


@router.get("/workspace/{slug}/parsed-files")
async def get_parsed_files(
    slug: str,
    thread_slug: str = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get parsed files for a workspace - files ready for chat context."""
    # Verify workspace exists
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    ws = result.scalar_one_or_none()
    
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Filter by is_attachment and optionally thread_id
    query = select(Document).where(
        Document.workspace_id == ws.id,
        Document.status == DocumentStatus.COMPLETED,
        Document.is_attachment == True
    )
    
    if thread_slug:
        from app.models.thread import WorkspaceThread
        thread_result = await db.execute(
            select(WorkspaceThread).where(
                WorkspaceThread.workspace_id == ws.id,
                WorkspaceThread.slug == thread_slug
            )
        )
        thread = thread_result.scalar_one_or_none()
        if thread:
            query = query.where(Document.thread_id == thread.id)
        else:
            # If thread specified but not found, return empty
            return {"files": [], "contextWindow": 128000, "currentContextTokenCount": 0}
    else:
        # If no thread_slug, show only thread-less attachments? 
        # Or all attachments for this workspace? AnythingLLM usually shows thread-specific.
        query = query.where(Document.thread_id == None)
    
    docs_result = await db.execute(query)
    docs = docs_result.scalars().all()
    
    files = [
        {
            "id": doc.id,
            "name": doc.filename,
            "title": doc.filename,
            "tokens": 0,
            "threadId": doc.thread_id
        }
        for doc in docs
    ]
    
    return {
        "files": files,
        "contextWindow": 128000,
        "currentContextTokenCount": 0
    }


@router.post("/workspace/{slug}/parse")
async def parse_file(
    slug: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload and parse file for chat context (attachment).
    Matches AnythingLLM's /parse endpoint.
    """
    from fastapi import UploadFile, File, Form
    from app.config import get_settings
    import shutil
    import os
    
    form = await request.form()
    file = form.get("file")
    thread_slug = form.get("threadSlug")
    
    if not file or not isinstance(file, UploadFile):
         raise HTTPException(status_code=400, detail="File is required")
         
    # Verify workspace
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    ws = result.scalar_one_or_none()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
        
    # Get thread if provided
    thread_id = None
    if thread_slug and thread_slug != "null":
        from app.models.thread import WorkspaceThread
        tr_result = await db.execute(
            select(WorkspaceThread).where(
                WorkspaceThread.workspace_id == ws.id,
                WorkspaceThread.slug == thread_slug
            )
        )
        thread = tr_result.scalar_one_or_none()
        if thread:
            thread_id = thread.id
            
    settings = get_settings()
    # Save to thread-specific or workspace-specific dir
    save_dir = os.path.join(settings.upload_dir, slug, "attachments")
    if thread_slug and thread_slug != "null":
        save_dir = os.path.join(settings.upload_dir, slug, "threads", thread_slug)
        
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file.filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Create Document record
    doc = Document(
        filename=file.filename,
        original_filename=file.filename,
        file_path=os.path.relpath(file_path, settings.upload_dir),
        file_size=os.path.getsize(file_path),
        mime_type=file.content_type,
        status=DocumentStatus.COMPLETED, # For now, assume ready. Ideally process RAG.
        workspace_id=ws.id,
        thread_id=thread_id,
        is_attachment=True
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    
    return {
        "success": True,
        "files": [
            {
                "id": doc.id,
                "name": doc.filename,
                "title": doc.filename,
                "tokenCountEstimate": 0,
                "threadId": doc.thread_id
            }
        ]
    }


@router.delete("/workspace/{slug}/parsed-files")
async def delete_parsed_files(
    slug: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete parsed files from workspace context."""
    from app.config import get_settings
    import os
    
    # Verify workspace exists
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    ws = result.scalar_one_or_none()
    
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    file_ids = body.get("fileIds", body.get("ids", []))
    if not file_ids:
        return {"success": True, "removed": 0}
        
    settings = get_settings()
    removed_count = 0
    
    for fid in file_ids:
        try:
            # Find and verify document is an attachment for this workspace
            doc_result = await db.execute(
                select(Document).where(
                    Document.id == fid,
                    Document.workspace_id == ws.id,
                    Document.is_attachment == True
                )
            )
            doc = doc_result.scalar_one_or_none()
            if not doc:
                continue
                
            # Delete file from disk
            full_path = os.path.join(settings.upload_dir, doc.file_path)
            if os.path.exists(full_path):
                os.remove(full_path)
                
            # Delete DB record
            await db.delete(doc)
            removed_count += 1
        except Exception as e:
            print(f"Error deleting parsed file {fid}: {e}")
            
    await db.commit()
    return {"success": True, "removed": removed_count}


@router.get("/community-hub/settings")
async def get_community_hub_settings():
    """Get community hub settings - stub for frontend compatibility."""
    return {"enabled": False}


# ============= Admin Workspaces =============

@router.get("/admin/workspaces")
async def admin_list_workspaces(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all workspaces (admin view)."""
    from app.services.auth_service import get_current_active_admin
    from sqlalchemy.orm import selectinload
    
    # Check admin
    if current_user.role.value != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Get all workspaces with owner info
    query = select(Workspace).options(selectinload(Workspace.owner))
    result = await db.execute(query)
    workspaces = result.scalars().all()
    
    response = []
    for ws in workspaces:
        # Get document counts and user counts
        from app.models import WorkspaceUser
        doc_count_res = await db.execute(
            select(func.count(Document.id)).where(Document.workspace_id == ws.id)
        )
        
        # Get all member user IDs
        users_res = await db.execute(
            select(WorkspaceUser.user_id).where(WorkspaceUser.workspace_id == ws.id)
        )
        user_ids = [row[0] for row in users_res.fetchall()]
        
        response.append({
            "id": ws.id,
            "name": ws.name,
            "slug": ws.slug,
            "description": ws.description,
            "createdAt": ws.created_at.isoformat() if ws.created_at else None,
            "documentCount": doc_count_res.scalar() or 0,
            "userCount": len(user_ids),
            "userIds": user_ids,
            "owner": {
                "username": ws.owner.username if ws.owner else "Unknown"
            }
        })
    
    return {"workspaces": response}


@router.post("/admin/workspaces/new")
async def admin_create_workspace(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new workspace (admin)."""
    # Check admin
    if current_user.role.value != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Workspace name is required")
    
    slug = slugify(name)
    
    # Check slug uniqueness
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    existing = result.scalar_one_or_none()
    if existing:
        counter = 1
        while existing:
            new_slug = f"{slug}-{counter}"
            result = await db.execute(select(Workspace).where(Workspace.slug == new_slug))
            existing = result.scalar_one_or_none()
            if not existing:
                slug = new_slug
                break
            counter += 1
    
    ws = Workspace(
        name=name,
        slug=slug,
        description=body.get("description"),
        owner_id=current_user.id
    )
    db.add(ws)
    await db.commit()
    await db.refresh(ws)
    
    return {
        "workspace": {
            "id": ws.id,
            "name": ws.name,
            "slug": ws.slug,
            "description": ws.description,
            "createdAt": ws.created_at.isoformat() if ws.created_at else None,
            "documentCount": 0,
            "userCount": 1
        },
        "message": None
    }


@router.delete("/admin/workspaces/{workspace_id}")
async def admin_delete_workspace(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a workspace by ID (admin)."""
    # Check admin
    if current_user.role.value != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    ws = result.scalar_one_or_none()
    
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    
    # Save slug before deleting the record
    workspace_slug = ws.slug
    
    # Notify RAG service to purge actual data
    from app.services.rag_service import rag_client
    await rag_client.delete_workspace_data(workspace_slug)
    
    # Clean up physical documents directory 
    import os
    import shutil
    from app.config import get_settings
    settings = get_settings()
    workspace_upload_dir = os.path.join(settings.upload_dir, workspace_slug)
    if os.path.exists(workspace_upload_dir) and os.path.isdir(workspace_upload_dir):
        try:
            shutil.rmtree(workspace_upload_dir)
        except Exception as e:
            print(f"Warning: Failed to delete workspace directory {workspace_upload_dir}: {e}")
            
    await db.delete(ws)
    await db.commit()

@router.get("/admin/workspaces/{workspace_id}/users")
async def admin_get_workspace_users(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get users in a workspace (admin)."""
    # Check admin
    if current_user.role.value != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    ws = result.scalar_one_or_none()
    
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
        
    # Get all users with access from WorkspaceUser table
    from app.models import WorkspaceUser
    users_query = select(User, WorkspaceUser.created_at).join(WorkspaceUser).where(WorkspaceUser.workspace_id == workspace_id)
    users_result = await db.execute(users_query)
    users_data = users_result.all()  # returns list of (User, created_at)
    
    users = []
    for u, created_at in users_data:
        users.append({
            "id": u.id,
            "username": u.username,
            "role": u.role.value,
            "lastUpdatedAt": created_at.strftime("%Y-%m-%d %H:%M") if created_at else None
        })
        
    return {"users": users}


@router.post("/admin/workspaces/{workspace_id}/update-users")
async def admin_update_workspace_users(
    workspace_id: int,
    body: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update users in a workspace (admin)."""
    # Check admin
    if current_user.role.value != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
        
    user_ids = body.get("userIds", [])
    
    result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    ws = result.scalar_one_or_none()
    
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
        
    from app.models import WorkspaceUser
    from sqlalchemy import delete
    
    # Clear existing non-owner members? Or just all members and rebuild?
    # Usually we want to reflect exactly what's in 'userIds'
    await db.execute(delete(WorkspaceUser).where(WorkspaceUser.workspace_id == workspace_id))
    
    # Add new members
    for uid in user_ids:
        new_member = WorkspaceUser(user_id=uid, workspace_id=workspace_id)
        db.add(new_member)
    
    await db.commit()
            
    return {"success": True, "message": "Users updated"}

