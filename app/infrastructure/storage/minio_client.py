"""
infrastructure/storage/minio_client.py
Trách nhiệm:
  - Khởi tạo MinIO client
  - Các helper functions: upload, download, get presigned URL
"""
import os
from minio import Minio
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("MINIO_CLIENT")
_minio_client = None

def get_minio_client() -> Minio:
    """Khởi tạo và trả về instance Minio client."""
    global _minio_client
    if _minio_client is None:
        _minio_client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )
        logger.debug(f"MinIO client initialized: {settings.MINIO_ENDPOINT}")
    return _minio_client
    
def download_ocr_image(object_path: str) -> str:
    """Tải ảnh OCR từ Minio lưu vào thư mục /tmp/."""
    client = get_minio_client()
    local_path = f"/tmp/{object_path}"
    
    if os.path.exists(local_path):
        return local_path
        
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        client.fget_object(
            settings.MINIO_BUCKET_OCR_RESULTS,
            object_path,
            local_path
        )
        logger.debug(f"Downloaded image from MinIO: {object_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download image {object_path} from MinIO: {e}")
        return ""