from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body
from typing import Dict, Any, Optional
import uuid
import os
import asyncio
import shutil
import httpx

from app.services.indexing.document_indexer import default_engine
from app.utils.logger import get_logger
from app.config import settings

router = APIRouter()
logger = get_logger("API_INGEST")

OCR_SERVICE_URL = settings.OCR_SERVICE_URL
OCR_POLL_INTERVAL = 5      # giây
OCR_POLL_TIMEOUT = settings.OCR_POLL_TIMEOUT     # giây tối đa chờ OCR

# MinIO endpoint thực tế (dùng để rewrite presigned URL từ OCR service)
# OCR service generate URL với host "localhost" nhưng MinIO thực tế nằm trên máy khác
_MINIO_ENDPOINT = settings.MINIO_ENDPOINT
_MINIO_SECURE = settings.MINIO_SECURE
_MINIO_SCHEME = "https" if _MINIO_SECURE else "http"
_MINIO_PUBLIC_BASE = f"{_MINIO_SCHEME}://{_MINIO_ENDPOINT}"


def _rewrite_minio_url(url: str) -> str:
    """
    Rewrite presigned URL nếu host là localhost/127.0.0.1 → dùng MINIO_ENDPOINT thực tế.
    OCR service chạy trên máy 156, nó tạo presigned URL với host 'localhost:9000'
    nhưng từ máy này phải truy cập qua 10.0.0.156:9000.
    """
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    if parsed.hostname in ("localhost", "127.0.0.1"):
        rewritten = parsed._replace(
            scheme=_MINIO_SCHEME,
            netloc=_MINIO_ENDPOINT
        )
        new_url = urlunparse(rewritten)
        logger.info(f"[OCR] Rewrite MinIO URL: {parsed.netloc} → {_MINIO_ENDPOINT}")
        return new_url
    return url


async def _submit_to_ocr(file_path: str, filename: str, model: str = "auto") -> str:
    """Upload file lên OCR service và trả về job_id."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, "application/pdf")}
            data = {"model": model}
            resp = await client.post(
                f"{OCR_SERVICE_URL}/api/v1/ocr/process",
                files=files,
                data=data,
            )
        resp.raise_for_status()
        result = resp.json()
        ocr_job_id = result.get("job_id")
        if not ocr_job_id:
            raise ValueError(f"OCR service không trả về job_id. Response: {result}")
        logger.info(f"[OCR] Submitted → job_id={ocr_job_id} | model={result.get('message', '')}")
        return ocr_job_id


async def _poll_ocr_status(ocr_job_id: str) -> Dict[str, Any]:
    """Poll OCR service cho đến khi job SUCCESS, trả về status response."""
    elapsed = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        while elapsed < OCR_POLL_TIMEOUT:
            resp = await client.get(f"{OCR_SERVICE_URL}/api/v1/ocr/status/{ocr_job_id}")
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "PENDING").upper()

            logger.info(f"[OCR] Poll {ocr_job_id} → {status} ({elapsed}s)")

            if status == "SUCCESS":
                return data
            if status == "FAILED":
                raise RuntimeError(f"OCR job {ocr_job_id} thất bại: {data.get('error')}")

            await asyncio.sleep(OCR_POLL_INTERVAL)
            elapsed += OCR_POLL_INTERVAL

    raise TimeoutError(f"OCR job {ocr_job_id} timeout sau {OCR_POLL_TIMEOUT}s")


async def _download_ocr_json(ocr_job_id: str) -> Dict[str, Any]:
    """
    Download kết quả OCR JSON từ MinIO bằng SDK (không dùng presigned URL).
    Presigned URL bị 403 khi rewrite host vì chữ ký AWS tính theo header 'host'.
    → Dùng MinIO client với credentials rag-service để get_object trực tiếp.
    """
    import io
    import json as json_lib
    from minio import Minio

    minio_client = Minio(
        _MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=_MINIO_SECURE,
    )

    bucket = settings.MINIO_BUCKET_OCR_RESULTS
    object_path = f"{ocr_job_id}/{ocr_job_id}.json"

    logger.info(f"[OCR] Downloading from MinIO: {bucket}/{object_path}")

    try:
        response = minio_client.get_object(bucket, object_path)
        raw = response.read()
        response.close()
        response.release_conn()
        return json_lib.loads(raw.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"MinIO get_object({bucket}/{object_path}) thất bại: {e}")


@router.post("/ocr-result")
async def ingest_ocr_result(
    payload: Dict[str, Any] = Body(..., description="JSON Output from OCR Service"),
):
    """
    Tiếp nhận kết quả OCR JSON và thực hiện Indexing vào RAG.
    Input: JSON chứa cấu trúc document (content/document key).
    """
    try:
        job_id = payload.get("job_id") or str(uuid.uuid4())
        workspace = payload.get("workspace", "default")
        logger.info(f"Ingest OCR-result. Job={job_id} | Workspace={workspace}")

        if "content" not in payload and "document" not in payload:
            raise HTTPException(
                status_code=400,
                detail="Invalid OCR JSON: thiếu key 'content' hoặc 'document'."
            )

        result = await default_engine.index_document(
            payload, workspace=workspace, job_id=job_id,
            original_filename=payload.get("original_filename", job_id)
        )

        return {
            "message": "Indexing completed successfully",
            "job_id": job_id,
            "workspace": workspace,
            "data": result,
        }

    except Exception as e:
        logger.error(f"Ingest Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_and_index(
    file: UploadFile = File(...),
    workspace: str = Form("default"),
    ocr_model: str = Form("auto"),
):
    """
    Nhận file PDF từ api-gateway:
    1. Lưu file tạm
    2. Submit lên OCR service (10.0.0.156:8001) → nhận job_id
    3. Poll đến khi OCR SUCCESS
    4. Download JSON kết quả từ MinIO
    5. Index vào RAG (LightRAG + Neo4j + PostgreSQL)
    """
    job_id = str(uuid.uuid4())
    logger.info(f"Upload: file={file.filename} | workspace={workspace} | job_id={job_id}")

    # 1. Lưu file tạm
    temp_dir = os.path.join(settings.RAG_WORK_DIR, "temp_uploads", workspace)
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{job_id}_{file.filename}")

    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"File saved: {temp_path}")

        # [DEDUP] Tính mã MD5 của file để chống Double-Click / Retry sinh ra 2 luồng OCR song song
        import hashlib
        def get_file_md5(fp: str) -> str:
            hash_md5 = hashlib.md5()
            with open(fp, "rb") as f2:
                for chunk in iter(lambda: f2.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
            
        file_md5 = get_file_md5(temp_path)
        if not hasattr(router, "_active_md5_jobs"):
            router._active_md5_jobs = set()
            
        if file_md5 in router._active_md5_jobs:
            logger.warning(f"[DEDUP] Bỏ qua file trùng lặp (MD5={file_md5}). File đang được xử lý bởi tiến trình OCR/Indexing trước đó.")
            os.remove(temp_path)
            return {
                "success": True,
                "job_id": job_id,
                "ocr_job_id": "ignored_duplicate",
                "workspace": workspace,
                "filename": file.filename,
                "indexed": False,
                "message": "File đang được xử lý bởi tiến trình khác (Deduplicate).",
            }
            
        router._active_md5_jobs.add(file_md5)
        
        try:
            # 2. Submit sang OCR service
            logger.info(f"[OCR] Submitting to {OCR_SERVICE_URL} ...")
            try:
                ocr_job_id = await _submit_to_ocr(temp_path, file.filename, model=ocr_model)
            except Exception as e:
                logger.error(f"[OCR] Submit failed: {e}")
                return {
                    "success": False,
                    "job_id": job_id,
                    "workspace": workspace,
                    "filename": file.filename,
                    "indexed": False,
                    "message": f"OCR service không phản hồi: {e}",
                }

            # 3. Poll đến khi xong
            try:
                await _poll_ocr_status(ocr_job_id)
            except (RuntimeError, TimeoutError) as e:
                logger.error(f"[OCR] Poll failed: {e}")
                return {
                    "success": False,
                    "job_id": job_id,
                    "ocr_job_id": ocr_job_id,
                    "workspace": workspace,
                    "filename": file.filename,
                    "indexed": False,
                    "message": str(e),
                }

            # 4. Download JSON từ MinIO
            try:
                ocr_json = await _download_ocr_json(ocr_job_id)
                logger.info(f"[OCR] JSON downloaded, keys={list(ocr_json.keys())}")
            except Exception as e:
                logger.error(f"[OCR] JSON download failed: {e}")
                return {
                    "success": False,
                    "job_id": job_id,
                    "ocr_job_id": ocr_job_id,
                    "workspace": workspace,
                    "filename": file.filename,
                    "indexed": False,
                    "message": f"Download OCR JSON thất bại: {e}",
                }

            # 5. Index vào RAG
            ocr_json["workspace"] = workspace
            ocr_json["job_id"] = job_id
            ocr_json["original_filename"] = file.filename
            await default_engine.index_document(
                ocr_json, workspace=workspace, job_id=job_id,
                original_filename=file.filename
            )
            logger.info(f"[RAG] Indexed job={job_id} workspace={workspace}")

            return {
                "success": True,
                "job_id": job_id,
                "ocr_job_id": ocr_job_id,
                "workspace": workspace,
                "filename": file.filename,
                "indexed": True,
                "message": "OCR + Indexing hoàn tất thành công",
            }
        
        finally:
            if file_md5 in getattr(router, "_active_md5_jobs", set()):
                router._active_md5_jobs.remove(file_md5)

    except Exception as e:
        logger.error(f"Upload/Index Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

@router.delete("/workspace/{workspace_slug}/purge_data")
async def purge_workspace_data(workspace_slug: str):
    """
    Xóa toàn bộ dữ liệu Vector, Cache, và Đồ thị (Neo4j) của một workspace cụ thể.
    Được gọi từ API Gateway khi người dùng bấm Delete Workspace.
    """
    from app.services.management.cleanup_service import clean_postgres, clean_neo4j
    logger.info(f"Received request to purge data for workspace: {workspace_slug}")
    try:
        pg_deleted = await clean_postgres(workspace_slug)
        neo_nodes, neo_rels = await clean_neo4j(workspace_slug)
        
        return {
            "success": True,
            "message": f"Dọn dẹp hoàn tất cho workspace '{workspace_slug}'",
            "details": {
                "postgres_deleted_rows": pg_deleted,
                "neo4j_deleted_nodes": neo_nodes,
                "neo4j_deleted_rels": neo_rels
            }
        }
    except Exception as e:
        logger.error(f"Failed to purge data for workspace {workspace_slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi dọn dẹp dữ liệu: {str(e)}")

@router.delete("/workspace/{workspace_slug}/document/{filename}")
async def purge_document_data(workspace_slug: str, filename: str):
    """
    Xóa toàn bộ dữ liệu của một tài liệu cụ thể trong workspace.
    Được gọi từ API Gateway khi người dùng bấm Delete Document.
    """
    from app.services.management.cleanup_service import clean_document
    logger.info(f"Received request to purge data for document: {filename} in workspace: {workspace_slug}")
    try:
        deleted_count = await clean_document(workspace_slug, filename)
        
        return {
            "success": True,
            "message": f"Đã xóa dữ liệu tài liệu '{filename}'",
            "details": {
                "deleted_documents_count": deleted_count
            }
        }
    except Exception as e:
        logger.error(f"Failed to purge document data: {e}")
        raise HTTPException(status_code=500, detail=str(e))