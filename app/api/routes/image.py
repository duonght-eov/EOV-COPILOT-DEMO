from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from minio import Minio
from minio.error import S3Error
from app.config import settings

router = APIRouter()

def _get_minio():
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )

MIME_MAP = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}

@router.get("/image/{object_path:path}")
async def serve_image(object_path: str):
    """
    Proxy ảnh từ MinIO bucket ocr-results.
    object_path ví dụ: {job_id}/images/2_0.jpg
    """
    ext = object_path.rsplit(".", 1)[-1].lower() if "." in object_path else "jpg"
    content_type = MIME_MAP.get(ext, "image/jpeg")

    try:
        client = _get_minio()
        response = client.get_object(settings.MINIO_BUCKET_OCR_RESULTS, object_path)

        def iter_content():
            try:
                for chunk in response.stream(32 * 1024):
                    yield chunk
            finally:
                response.close()
                response.release_conn()

        return StreamingResponse(iter_content(), media_type=content_type)

    except S3Error as e:
        raise HTTPException(status_code=404, detail=f"Image not found: {object_path} ({e})")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
