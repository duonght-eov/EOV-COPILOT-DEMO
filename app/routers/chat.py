"""
Chat router - Chat with RAG system using streaming.
"""
import json
import time
import httpx
from typing import Optional, List
from pydantic import BaseModel
class UpdateChatRequest(BaseModel):
    chatId: int
    newText: str

class DeleteEditedChatsRequest(BaseModel):
    startingId: int

class ChatFeedbackRequest(BaseModel):
    feedback: Optional[int] = None

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import User, Workspace, Chat
from app.schemas import ChatMessage, ChatResponse
from app.services.auth_service import get_current_user, has_workspace_access
from app.services.rag_service import get_rag_client, RAGServiceClient

router = APIRouter(prefix="/workspace", tags=["Chat"])


@router.post("/{slug}/stream-chat")
async def chat_with_workspace(
    slug: str,
    payload: ChatMessage = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    rag_client: RAGServiceClient = Depends(get_rag_client)
):
    """
    Chat with a workspace using RAG.
    Returns streaming response for real-time updates.
    """
    # Get workspace
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Check access
    if not await has_workspace_access(db, current_user, workspace.id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Use workspace query mode if not specified
    query_mode = payload.mode or workspace.query_mode or "mix"
    
    # Store response data for saving after stream completes
    response_data = {"full_response": "", "sources": [], "images": []}
    
    # Pre-capture data for background task
    workspace_id = workspace.id
    user_id = current_user.id
    user_message = payload.message
    
    async def generate_stream():
        """Generate streaming response."""
        import uuid as uuid_lib
        stream_uuid = str(uuid_lib.uuid4())
        
        print(f"[STREAM-CHAT] Starting stream for query: {payload.message}")
        
        # [EOV INTEGRATION] Nếu Workspace bật Agent Mode → proxy toàn bộ sang Agentic Service
        from app.config import get_settings
        settings = get_settings()

        if workspace.is_predict_enabled:
            print(f"[AGENT-ROUTING] Proxying query to Agentic Service for workspace {slug}")
            try:
                from app.models.models import WorkspaceConnector
                from app.database import async_session_maker

                connector_data = None
                async with async_session_maker() as session:
                    result_conn = await session.execute(
                        select(WorkspaceConnector).where(WorkspaceConnector.workspace_id == workspace.id)
                    )
                    connector = result_conn.scalar_one_or_none()
                    if connector:
                        connector_data = {
                            "base_url": connector.base_url,
                            "auth_type": connector.auth_type,
                            "auth_credentials": connector.auth_credentials,
                        }

                agent_payload = {
                    "message": payload.message,
                    "session_id": f"{user_id}_{workspace_id}",
                    "workspace_slug": slug,
                    "connector": connector_data,
                }

                # Thu thập toàn bộ stream từ Agent Service
                agent_full_response = ""
                agent_sources = []
                agent_images = []
                start_time = time.time()

                async with httpx.AsyncClient(timeout=180.0) as client:
                    async with client.stream(
                        "POST",
                        f"{settings.agentic_service_url}/api/v1/agent/chat/stream",
                        json=agent_payload,
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if not (line and line.startswith("data: ")):
                                continue
                            raw = line[6:]
                            try:
                                data = json.loads(raw)
                                etype = data.get("type", "")
                                text = data.get("textResponse", "")
                                sources = data.get("sources", [])
                                imgs = data.get("images", [])

                                if etype == "textResponseChunk" and not data.get("close"):
                                    agent_full_response += text
                                    yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'textResponseChunk', 'textResponse': text, 'close': False, 'sources': []})}\n\n"

                                elif etype == "textResponse" or (etype == "textResponseChunk" and data.get("close")):
                                    if sources:
                                        agent_sources = sources
                                    if imgs:
                                        agent_images = imgs

                            except json.JSONDecodeError:
                                pass

                # Lưu chat vào DB
                end_time = time.time()
                duration = end_time - start_time
                metrics = {
                    "duration": duration,
                    "outputTps": (len(agent_full_response) / 4) / max(duration, 0.001),
                    "model": workspace.llm_model or "Agent",
                    "timestamp": start_time,
                }
                chat_id = None
                async with async_session_maker() as session:
                    try:
                        # Mẹo: Gộp images vào sources để lưu db mà không chỉnh schema
                        db_sources = list(agent_sources) if agent_sources else []
                        if agent_images:
                            db_sources.append({"_type": "images", "urls": agent_images})

                        chat = Chat(
                            session_id=f"session_{user_id}_{datetime.utcnow().timestamp()}",
                            prompt=user_message,
                            response=agent_full_response,
                            sources=json.dumps(db_sources) if db_sources else None,
                            metrics=json.dumps(metrics),
                            workspace_id=workspace_id,
                            user_id=user_id,
                        )
                        session.add(chat)
                        await session.commit()
                        await session.refresh(chat)
                        chat_id = chat.id
                        print(f"[AGENT-ROUTING] Chat saved: id={chat_id}, sources={len(agent_sources)}, images={len(agent_images)}")
                    except Exception as db_e:
                        print(f"[AGENT-ROUTING] DB save error: {db_e}")

                # Phát final textResponse kèm images để Frontend render ImageGallery
                yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'textResponse', 'textResponse': agent_full_response, 'close': True, 'sources': agent_sources, 'images': agent_images, 'metrics': metrics, 'chatId': chat_id}, ensure_ascii=False)}\n\n"
                return

            except Exception as e:
                print(f"[AGENT-ROUTING] Agentic Service lỗi ({e}). Fallback về RAG.")
                # Fallback an toàn sang RAG nếu Agentic Service chưa chạy / lỗi


        start_time = time.time()

        
        try:
            # Try streaming from RAG service
            async for component in rag_client.query_stream(
                query=payload.message,
                workspace_slug=slug,
                mode=query_mode
            ):
                if "chunk" in component:
                    chunk = component["chunk"]
                    response_data["full_response"] += chunk
                    yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'textResponseChunk', 'textResponse': chunk, 'close': False, 'sources': []})}\n\n"
                
                if "sources" in component:
                    response_data["sources"] = component["sources"]

                if "images" in component:
                    response_data["images"] = component["images"]
                
                if "error" in component:
                    raise Exception(component["error"])
            
            # Calculate metrics
            end_time = time.time()
            duration = end_time - start_time
            token_count = len(response_data["full_response"]) / 4 # Approx 4 chars per token
            metrics = {
                "duration": duration,
                "outputTps": token_count / duration if duration > 0 else 0,
                "model": workspace.llm_model or "Unknown",
                "timestamp": start_time
            }
            
            print(f"[STREAM-CHAT] Stream complete, saving to DB")
            
            # Mẹo: Gộp images vào sources để lưu db mà không chỉnh schema
            db_sources = list(response_data["sources"]) if response_data["sources"] else []
            if "images" in response_data and response_data["images"]:
                # Attach images specifically as a meta entry
                db_sources.append({"_type": "images", "urls": response_data["images"]})

            # Save to DB synchronously to get ID
            chat_id = None
            from app.database import async_session_maker
            async with async_session_maker() as session:
                try:
                    chat = Chat(
                        session_id=f"session_{user_id}_{datetime.utcnow().timestamp()}",
                        prompt=user_message,
                        response=response_data["full_response"],
                        sources=json.dumps(db_sources) if db_sources else None,
                        metrics=json.dumps(metrics),
                        workspace_id=workspace_id,
                        user_id=user_id
                    )
                    session.add(chat)
                    await session.commit()
                    await session.refresh(chat)
                    chat_id = chat.id
                    print(f"[STREAM-CHAT] Chat saved successfully with ID: {chat_id}")
                except Exception as e:
                    print(f"[STREAM-CHAT] Error saving chat: {e}")

            # Final message: truyền images array gốc thay vì nhúng vào markdown text
            yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'textResponse', 'textResponse': response_data['full_response'], 'close': True, 'sources': response_data['sources'], 'images': response_data.get('images', []), 'metrics': metrics, 'chatId': chat_id}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            print(f"[STREAM-CHAT] Stream error: {e}, trying fallback")
            # Fallback: try non-streaming query
            try:
                result = await rag_client.query(
                    query=payload.message,
                    workspace_slug=slug,
                    mode=query_mode,
                    stream=False
                )
                response_data["full_response"] = result.get("answer", str(e))
                response_data["sources"] = result.get("sources", [])
                
                # Calculate metrics for fallback
                end_time = time.time()
                duration = end_time - start_time
                token_count = len(response_data["full_response"]) / 4
                metrics = {
                    "duration": duration,
                    "outputTps": token_count / duration if duration > 0 else 0,
                    "model": workspace.llm_model or "Unknown",
                    "timestamp": start_time
                }
                
                print(f"[STREAM-CHAT] Fallback success, saving to DB")
                
                # Save to DB synchronously
                chat_id = None
                from app.database import async_session_maker
                async with async_session_maker() as session:
                    try:
                        chat = Chat(
                            session_id=f"session_{user_id}_{datetime.utcnow().timestamp()}",
                            prompt=user_message,
                            response=response_data["full_response"],
                            sources=json.dumps(response_data["sources"]) if response_data["sources"] else None,
                            metrics=json.dumps(metrics),
                            workspace_id=workspace_id,
                            user_id=user_id
                        )
                        session.add(chat)
                        await session.commit()
                        await session.refresh(chat)
                        chat_id = chat.id
                    except Exception as db_e:
                        print(f"[STREAM-CHAT] Error saving chat during fallback: {db_e}")

                # Send full response with type
                yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'textResponse', 'textResponse': response_data['full_response'], 'close': True, 'sources': response_data['sources'], 'images': response_data.get('images', []), 'metrics': metrics, 'chatId': chat_id})}\n\n"
                
            except Exception as inner_e:
                error_msg = f"Error: Unable to process query. {str(inner_e)}"
                print(f"[STREAM-CHAT] Fallback also failed: {inner_e}")
                yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'abort', 'textResponse': error_msg, 'close': True, 'error': True, 'sources': []})}\n\n"
                response_data["full_response"] = error_msg
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream"
        }
    )


@router.get("/image/{object_path:path}")
async def proxy_image(
    object_path: str,
    rag_client: RAGServiceClient = Depends(get_rag_client),
):
    """
    Proxy ảnh từ RAG service (MinIO) về browser.
    URL: /api/workspace/image/{job_id}/images/X.jpg
    """
    from fastapi.responses import StreamingResponse as SR
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{rag_client.base_url}/api/v1/image/{object_path}",
                follow_redirects=True,
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail="Image not found")

            content_type = resp.headers.get("content-type", "image/jpeg")
            return SR(
                content=resp.iter_bytes(),
                media_type=content_type,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def convert_to_chat_history(chats):
    """Convert DB chats to frontend history format."""
    history = []
    for chat in chats:
        # User message
        history.append({
            "role": "user",
            "content": chat.prompt,
            "sentAt": chat.created_at.timestamp(),
            "created_at": chat.created_at.isoformat()
        })
        
        # Assistant message
        # Try to parse response as JSON if it was stored that way (compatibility)
        content = chat.response
        try:
            data = json.loads(chat.response)
            if isinstance(data, dict) and "text" in data:
                content = data["text"]
        except:
            pass
            
        # Extract images from sources if present (using the metadata trick)
        raw_sources = json.loads(chat.sources) if chat.sources else []
        final_sources = []
        images = []
        
        for s in raw_sources:
            if isinstance(s, dict) and s.get("_type") == "images":
                images.extend(s.get("urls", []))
            else:
                final_sources.append(s)

        history.append({
            "role": "assistant",
            "content": content,
            "sources": final_sources,
            "images": images,
            "metrics": json.loads(chat.metrics) if getattr(chat, 'metrics', None) else {},
            "chatId": chat.id,
            "sentAt": chat.created_at.timestamp(),
            "created_at": chat.created_at.isoformat(),
            "feedbackScore": None
        })
    return history


@router.get("/{slug}/chats")
async def get_chat_history(
    slug: str,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get chat history for a workspace."""
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Check access
    if not await has_workspace_access(db, current_user, workspace.id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get chats
    query = select(Chat).where(Chat.workspace_id == workspace.id)
    
    # Non-admin users only see their own chats
    if current_user.role.value != "admin":
        query = query.where(Chat.user_id == current_user.id)
        
    # Exclude thread chats from main workspace history
    query = query.where(Chat.thread_id.is_(None))
    
    # Get latest chats first via DESC, then reverse to show in chronological order
    query = query.order_by(Chat.created_at.desc(), Chat.id.desc()).offset(offset).limit(limit)
    
    result = await db.execute(query)
    chats = result.scalars().all()
    
    # Reverse to get chronological order (Oldest -> Newest)
    chats = chats[::-1]
    
    return {
        "history": convert_to_chat_history(chats)
    }


@router.delete("/{slug}/chats/{chat_id}")
async def delete_chat(
    slug: str,
    chat_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a specific chat."""
    result = await db.execute(select(Chat).where(Chat.id == chat_id))
    chat = result.scalar_one_or_none()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Check access
    if current_user.role.value != "admin" and chat.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    await db.delete(chat)
    await db.commit()
    
    return {"success": True, "message": "Chat deleted"}


class DeleteChatsRequest(BaseModel):
    chatIds: List[int] = []

@router.delete("/{slug}/delete-chats")
async def delete_chats(
    slug: str,
    body: DeleteChatsRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete specific chats in a workspace."""
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Check access
    if not await has_workspace_access(db, current_user, workspace.id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not body.chatIds:
        return {"success": True, "message": "No chats to delete"}

    # Delete chats
    from sqlalchemy import delete
    
    query = delete(Chat).where(
        Chat.workspace_id == workspace.id,
        Chat.id.in_(body.chatIds)
    )
    
    # Non-admin users only delete their own chats
    if current_user.role.value != "admin":
        query = query.where(Chat.user_id == current_user.id)
    
    await db.execute(query)
    await db.commit()
    
    return {"success": True, "message": "Chats deleted"}


@router.post("/{slug}/update-chat")
async def update_chat(
    slug: str,
    body: UpdateChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a specific chat's response text."""
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Check access
    if not await has_workspace_access(db, current_user, workspace.id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get chat
    result = await db.execute(
        select(Chat).where(
            Chat.id == body.chatId,
            Chat.workspace_id == workspace.id,
            Chat.user_id == current_user.id
        )
    )
    chat = result.scalar_one_or_none()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found or access denied")
    
    # Update response
    chat.response = body.newText
    await db.commit()
    
    return {"success": True, "message": "Chat updated"}


@router.delete("/{slug}/delete-edited-chats")
async def delete_edited_chats(
    slug: str,
    body: DeleteEditedChatsRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete chats from a starting ID onwards (inclusive)."""
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Check access
    if not await has_workspace_access(db, current_user, workspace.id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    from sqlalchemy import delete
    query = delete(Chat).where(
        Chat.workspace_id == workspace.id,
        Chat.id >= body.startingId,
        Chat.thread_id.is_(None) # Main workspace chats only
    )
    
    if current_user.role.value != "admin":
        query = query.where(Chat.user_id == current_user.id)
        
    await db.execute(query)
    await db.commit()
    
    return {"success": True, "message": "Edited chats deleted"}


@router.post("/{slug}/chat-feedback/{chat_id}")
async def chat_feedback(
    slug: str,
    chat_id: int,
    body: ChatFeedbackRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Set feedback score for a chat message."""
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
        
    # Get chat
    result = await db.execute(
        select(Chat).where(
            Chat.id == chat_id,
            Chat.workspace_id == workspace.id
        )
    )
    chat = result.scalar_one_or_none()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
        
    # Update feedback (This would need a feedback_score column in Chat model)
    # For now, we'll just return success if the model doesn't have it yet
    # Or we can try to save it if column exists
    try:
        chat.feedback_score = body.feedback
        await db.commit()
    except Exception:
        # If column doesn't exist yet, just ignore but return success for UI compatibility
        pass
        
    return {"success": True}


# ============= Thread Chat Endpoints =============

@router.post("/{slug}/thread/{thread_slug}/stream-chat")
async def stream_chat_in_thread(
    slug: str,
    thread_slug: str,
    payload: ChatMessage = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    rag_client: RAGServiceClient = Depends(get_rag_client)
):
    """
    Chat with a workspace in a specific thread using streaming.
    Messages are linked to the thread for organized conversation history.
    """
    from app.models.thread import WorkspaceThread
    
    # Get workspace
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Check access
    if not await has_workspace_access(db, current_user, workspace.id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get thread
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
    
    # Use workspace query mode if not specified
    query_mode = payload.mode or workspace.query_mode or "mix"
    
    # Store response data for saving after stream completes
    response_data = {"full_response": "", "sources": []}
    
    # Pre-capture data for background task
    workspace_id = workspace.id
    user_id = current_user.id
    thread_id = thread.id
    user_message = payload.message
    
    async def generate_stream():
        """Generate streaming response."""
        import uuid as uuid_lib
        stream_uuid = str(uuid_lib.uuid4())
        
        print(f"[THREAD-STREAM-CHAT] Starting stream for query: {payload.message}")
        
        start_time = time.time()
        
        try:
            # Try streaming from RAG service
            async for component in rag_client.query_stream(
                query=payload.message,
                workspace_slug=slug,
                mode=query_mode
            ):
                if "chunk" in component:
                    chunk = component["chunk"]
                    response_data["full_response"] += chunk
                    # Yield SSE formatted data with type for frontend
                    yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'textResponseChunk', 'textResponse': chunk, 'close': False, 'sources': []})}\n\n"
                
                if "sources" in component:
                    response_data["sources"] = component["sources"]

                if "images" in component:
                    if "images" not in response_data:
                        response_data["images"] = []
                    response_data["images"] = component["images"]

                if "error" in component:
                    raise Exception(component["error"])
            
            # Calculate metrics
            end_time = time.time()
            duration = end_time - start_time
            token_count = len(response_data["full_response"]) / 4 # Approx 4 chars per token
            metrics = {
                "duration": duration,
                "outputTps": token_count / duration if duration > 0 else 0,
                "model": workspace.llm_model or "Unknown",
                "timestamp": start_time
            }
            
            print(f"[THREAD-STREAM-CHAT] Stream complete, saving to DB")
            
            # Save to DB synchronously to get ID
            chat_id = None
            from app.database import async_session_maker
            async with async_session_maker() as session:
                try:
                    # Mẹo: Gộp images vào sources để lưu db mà không chỉnh schema
                    db_sources = list(response_data["sources"]) if response_data["sources"] else []
                    if "images" in response_data and response_data["images"]:
                        db_sources.append({"_type": "images", "urls": response_data["images"]})

                    chat = Chat(
                        session_id=f"thread_{thread_id}_{user_id}_{datetime.utcnow().timestamp()}",
                        prompt=user_message,
                        response=response_data["full_response"],
                        sources=json.dumps(db_sources) if db_sources else None,
                        metrics=json.dumps(metrics),
                        workspace_id=workspace_id,
                        user_id=user_id,
                        thread_id=thread_id
                    )
                    session.add(chat)
                    await session.commit()
                    await session.refresh(chat)
                    chat_id = chat.id
                    print(f"[THREAD-STREAM-CHAT] Chat saved successfully with ID: {chat_id}")
                except Exception as e:
                    print(f"[THREAD-STREAM-CHAT] Error saving chat: {e}")

            # Final message: truyền images array gốc thay vì nhúng vào markdown text
            yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'textResponse', 'textResponse': response_data['full_response'], 'close': True, 'sources': response_data['sources'], 'images': response_data.get('images', []), 'metrics': metrics, 'chatId': chat_id}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            print(f"[THREAD-STREAM-CHAT] Stream error: {e}, trying fallback")
            # Fallback: try non-streaming query
            try:
                result = await rag_client.query(
                    query=payload.message,
                    workspace_slug=slug,
                    mode=query_mode,
                    stream=False
                )
                response_data["full_response"] = result.get("answer", str(e))
                response_data["sources"] = result.get("sources", [])
                
                # Calculate metrics for fallback
                end_time = time.time()
                duration = end_time - start_time
                token_count = len(response_data["full_response"]) / 4
                metrics = {
                    "duration": duration,
                    "outputTps": token_count / duration if duration > 0 else 0,
                    "model": workspace.llm_model or "Unknown",
                    "timestamp": start_time
                }
                
                print(f"[THREAD-STREAM-CHAT] Fallback success, saving to DB")
                
                # Save to DB synchronously
                chat_id = None
                from app.database import async_session_maker
                async with async_session_maker() as session:
                    try:
                        chat = Chat(
                            session_id=f"thread_{thread_id}_{user_id}_{datetime.utcnow().timestamp()}",
                            prompt=user_message,
                            response=response_data["full_response"],
                            sources=json.dumps(response_data["sources"]) if response_data["sources"] else None,
                            metrics=json.dumps(metrics),
                            workspace_id=workspace_id,
                            user_id=user_id,
                            thread_id=thread_id
                        )
                        session.add(chat)
                        await session.commit()
                        await session.refresh(chat)
                        chat_id = chat.id
                    except Exception as db_e:
                        print(f"[THREAD-STREAM-CHAT] Error saving chat during fallback: {db_e}")

                # Send full response with type
                yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'textResponse', 'textResponse': response_data['full_response'], 'close': True, 'sources': response_data['sources'], 'metrics': metrics, 'chatId': chat_id})}\n\n"
                
                
            except Exception as inner_e:
                error_msg = f"Error: Unable to process query. {str(inner_e)}"
                print(f"[THREAD-STREAM-CHAT] Fallback also failed: {inner_e}")
                yield f"data: {json.dumps({'uuid': stream_uuid, 'type': 'abort', 'textResponse': error_msg, 'close': True, 'error': True, 'sources': []})}\n\n"
                response_data["full_response"] = error_msg
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream"
        }
    )
@router.get("/{slug}/thread/{thread_slug}/chats")
async def get_thread_chat_history(
    slug: str,
    thread_slug: str,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get chat history for a specific thread."""
    from app.models.thread import WorkspaceThread
    
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Check access
    if not await has_workspace_access(db, current_user, workspace.id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get thread
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
    
    # Get chats for this thread
    query = select(Chat).where(
        Chat.workspace_id == workspace.id,
        Chat.thread_id == thread.id
    )
    
    # Get latest chats first via DESC, then reverse to show in chronological order
    query = query.order_by(Chat.created_at.desc(), Chat.id.desc()).offset(offset).limit(limit)
    
    result = await db.execute(query)
    chats = result.scalars().all()
    
    # Reverse to get chronological order (Oldest -> Newest)
    chats = chats[::-1]
    
    return {
        "history": convert_to_chat_history(chats)
    }


@router.post("/{slug}/thread/{thread_slug}/update-chat")
async def update_thread_chat(
    slug: str,
    thread_slug: str,
    body: UpdateChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a specific chat's response text within a thread."""
    from app.models.thread import WorkspaceThread
    
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
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
    
    # Get chat
    result = await db.execute(
        select(Chat).where(
            Chat.id == body.chatId,
            Chat.workspace_id == workspace.id,
            Chat.thread_id == thread.id,
            Chat.user_id == current_user.id
        )
    )
    chat = result.scalar_one_or_none()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found or access denied")
    
    # Update response
    chat.response = body.newText
    await db.commit()
    
    return {"success": True, "message": "Chat updated"}


@router.delete("/{slug}/thread/{thread_slug}/delete-edited-chats")
async def delete_thread_edited_chats(
    slug: str,
    thread_slug: str,
    body: DeleteEditedChatsRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete chats from a starting ID onwards within a thread."""
    from app.models.thread import WorkspaceThread
    
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
        
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
    
    from sqlalchemy import delete
    query = delete(Chat).where(
        Chat.workspace_id == workspace.id,
        Chat.thread_id == thread.id,
        Chat.id >= body.startingId
    )
    
    if current_user.role.value != "admin":
        query = query.where(Chat.user_id == current_user.id)
        
    await db.execute(query)
    await db.commit()
    
    return {"success": True, "message": "Edited thread chats deleted"}

