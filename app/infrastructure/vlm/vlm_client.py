"""
infrastructure/vlm/vlm_client.py
Trách nhiệm:
  - vlm_model_func: Gọi Vision Language Model để caption ảnh
  - Quản lý trạng thái _vlm_offline để skip nhanh khi VLM down
  - Encode ảnh sang base64 và build multimodal payload
"""
import os
import time
import base64
import asyncio
import httpx
from app.config import settings
from app.utils.logger import get_logger
from app.infrastructure.llm.llm_func import llm_completion_func

logger = get_logger("VLM_CLIENT")

_vlm_offline: bool = False
_vlm_last_check: float = 0.0
_VLM_RECHECK_INTERVAL = 120.0

async def vlm_model_func(prompt: str, images: list[str] = [], **kwargs) -> str:
    global _vlm_offline, _vlm_last_check

    if not images:
        return await llm_completion_func(prompt, **kwargs)

    if _vlm_offline and (time.time() - _vlm_last_check) < _VLM_RECHECK_INTERVAL:
        return "Hình ảnh minh họa (VLM service không khả dụng)."

    def encode_image(image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception:
            return None

    content = [{"type": "text", "text": prompt}]
    for img_path in images:
        base64_image = encode_image(img_path)
        if base64_image:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

    sys_prompt = kwargs.get("system_prompt", "Bạn là một trợ lý AI chuyên phân tích hình ảnh. LUÔN trả lời bằng Tiếng Việt.")
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": content}]

    try:
        url = f"{settings.VLM_BASE_URL.rstrip('/')}/chat/completions"
        payload = {"model": settings.VLM_MODEL_NAME, "messages": messages, "temperature": 0.1, "max_tokens": 2048}
        headers = {"Authorization": f"Bearer {settings.VLM_API_KEY}", "Content-Type": "application/json"}
        
        timeout = httpx.Timeout(connect=5.0, read=360.0, write=10.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            content_text = response.json()["choices"][0]["message"].get("content", "")
            if not content_text or len(content_text.strip()) < 20:
                retry_payload = payload.copy()
                retry_payload["messages"][0]["content"] = "Mô tả hình ảnh bằng Tiếng Việt."
                retry_response = await client.post(url, json=retry_payload, headers=headers)
                return retry_response.json()["choices"][0]["message"].get("content", "Không thể tạo mô tả cho hình ảnh này.")
            return content_text

    except asyncio.TimeoutError:
        return "Hình ảnh minh họa (VLM timeout — bỏ qua)."

    except Exception as e:
        if "connect" in str(e).lower():
            _vlm_offline = True
            _vlm_last_check = time.time()
        return "Hình ảnh minh họa (VLM service không khả dụng)."
