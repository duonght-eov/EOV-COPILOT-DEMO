"""
services/indexing/document_indexer.py
Trách nhiệm:
  - IndexingEngine class: orchestrate việc index một tài liệu
  - Implements IIndexer interface
  - Gọi VLM, chunker, LightRAG để xử lý document
"""
import os
import asyncio
import ujson as json
from typing import Dict, Any, List
from app.utils.logger import get_logger
from app.services.processing.context_builder import ContextBuilder
from app.services.processing.prompt_loader import get_prompt_config
from app.infrastructure.vlm.vlm_client import vlm_model_func
from app.infrastructure.storage.minio_client import download_ocr_image
from app.infrastructure.graph.lightrag_factory import RAGFactory

logger = get_logger("DOCUMENT INDEXER")

class IndexingEngine:
    def __init__(self, doc_id: str = None):
        self.doc_id = doc_id
        # Khởi tạo utility xử lý khối text
        self.context_builder = ContextBuilder(context_window=2, max_context_chars=500)
        
        # Load trước prompt ra memory
        self.prompt_config = get_prompt_config()
        self.vlm_prompts = self.prompt_config.get("vlm_prompts", {})

    async def preprocess_content_for_chunking(self, full_content_list: List[Dict]) -> List[Dict]:
        """Processes raw content blocks to prepare them for chunking (Enriches images with VLM captions)."""
        text_only_content = []
        self._last_caption_cache: Dict[str, str] = {}
        
        def get_surrounding_text(curr_idx, all_items, window=2):
            prev_text, next_text = [], []
            for k in range(1, window + 1):
                idx = curr_idx - k
                if idx >= 0 and all_items[idx].get("type", "unknown") not in ["image", "table"]:
                    text_val = all_items[idx].get("text", "")
                    if text_val and len(text_val.strip()) > 0: prev_text.insert(0, text_val)
            for k in range(1, window + 1):
                idx = curr_idx + k
                if idx < len(all_items) and all_items[idx].get("type", "unknown") not in ["image", "table"]:
                    text_val = all_items[idx].get("text", "")
                    if text_val and len(text_val.strip()) > 0: next_text.append(text_val)
            return "\n".join(prev_text), "\n".join(next_text)
        logger.info("Phase 1.5: Inline Enrichment (Context-Aware Natural) - Parallelized...")
        semaphore = asyncio.Semaphore(5)

        async def process_item(i, item):
            item_type = item.get("type", "unknown")
            if item_type not in ["image", "table"]:
                return item
            
            elif item_type == "image":
                img_path = item.get("img_path", "")
                caption = item.get("text", "") 
                
                if caption and "[IMAGE_REF:" in caption and "Description:" in caption:
                    return item
                
                if img_path:
                    local_img_path = img_path
                    if not os.path.isabs(img_path) and not os.path.exists(img_path):
                        if img_path.startswith("ocr-results/"):
                            object_path = img_path.replace("ocr-results/", "", 1)
                            # Ủy quyền cho Infrastructure (Storage layer) tải ảnh
                            local_img_path = download_ocr_image(object_path)
                            if not local_img_path: img_path = None
                    
                    if local_img_path and os.path.exists(local_img_path):
                        try:
                            prev_ctx, _ = get_surrounding_text(i, full_content_list)
                            context_str = f"[Văn bản trước đó]:\n{prev_ctx}\n" if prev_ctx else "Không có văn bản ngữ cảnh cụ thể."
                            template = self.vlm_prompts.get("inline_enrichment_narrative")
                            prompt = template.replace("{{ context_str }}", context_str) if template else (
                                f"Ngữ cảnh tài liệu:\n---\n{context_str}\n---\nPhân tích hình ảnh chi tiết dựa vào ngữ cảnh này. Trả lời bằng Tiếng Việt."
                            )
                            
                            vlm_sys_prompt = self.vlm_prompts.get("vlm_system_prompt")
                            vlm_kwargs = {"system_prompt": vlm_sys_prompt.strip()} if vlm_sys_prompt else {}
                            
                            logger.info(f"Phase 1.5 [VLM]: Bắt đầu mô tả ảnh {os.path.basename(img_path)} (Page {item.get('page_idx', 'unknown')})")
                            async with semaphore:
                                caption = await asyncio.wait_for(
                                    vlm_model_func(prompt, images=[local_img_path], **vlm_kwargs),
                                    timeout=480.0
                                )
                            logger.info(f"Phase 1.5 [VLM]: Mô tả thành công: {os.path.basename(img_path)}")
                        except asyncio.TimeoutError:
                            logger.error(f"Phase 1.5 [VLM]: Timeout khi mô tả ảnh {os.path.basename(img_path)}")
                            caption = f"Hình ảnh minh họa: {os.path.basename(img_path)}"
                        except Exception as e:
                            logger.error(f"Phase 1.5 [VLM]: Lỗi '{e}' khi mô tả ảnh {os.path.basename(img_path)}")
                            caption = f"Hình ảnh minh họa: {os.path.basename(img_path)}"
                    else:
                        caption = f"Hình ảnh minh họa: {os.path.basename(img_path) if img_path else 'unknown'}"
                if img_path:
                    self._last_caption_cache[img_path] = caption
                page_label = f"[Page {item.get('page_idx', '')}]" if item.get('page_idx') is not None else ""
                img_ref = f"[IMAGE_REF:{img_path}]" if img_path else ""
                return {"type": "text", "text": f"{page_label}{img_ref} {caption}".strip(), "page_idx": item.get("page_idx")}
            elif item_type == "table":
                return {
                    "type": "table", "text": item.get("table_body") or item.get("text") or item.get("html", ""),
                    "table_caption": "Bảng dữ liệu", "page_idx": item.get("page_idx"), "bbox": item.get("bbox")
                }
            return None
        # Gather
        tasks = [process_item(i, item) for i, item in enumerate(full_content_list)]
        results = await asyncio.gather(*tasks)
        for res in results:
            if res is not None: text_only_content.append(res)
        return text_only_content

    def _extract_multimodal_items(self, full_content_list: List[Dict]) -> List[Dict]:
        """Bóc tách Image/Table từ full_content_list cho RAGAnything."""
        items = []
        try:
            for b in full_content_list:
                if b.get("type") not in ["image", "table"]: continue
                item = b.copy()
                if "img_path" not in item and "image_path" in item: item["img_path"] = item["image_path"]
                img_path = item.get("img_path", "")
                item["_ocr_img_path"] = img_path 
                if img_path:
                    if os.path.isabs(img_path) and os.path.exists(img_path): pass
                    elif img_path.startswith("ocr-results/"):
                        object_path = img_path.replace("ocr-results/", "", 1)
                        # Ủy quyền cho Infrastructure MinIO Client
                        local_path = download_ocr_image(object_path)
                        item["img_path"] = local_path if local_path else None
                    else: item["img_path"] = None
                items.append(item)
        except Exception as e:
            logger.error(f"Multimodal extraction error: {e}")
        return items

    async def index_document(self, ocr_data: Dict[str, Any], workspace: str = "default", job_id: str = "unknown", original_filename: str = ""):
        """Kịch bản Indexing Hợp nhất (Core Workflow Use-Case)"""
        # (1) Lấy instance Database từ Factory thay vì tự khởi tạo
        rag_instance, rag_anything = await RAGFactory.get_or_create_rag(workspace)
        logger.info(f"Processing Ingestion Job: {job_id} | Workspace: {workspace}")
        graph_enhanced = False
        try:
            logger.info("Phase 1: Extracting content and building context map...")
            full_content_list = self.context_builder.extract_full_content_list(ocr_data)
            context_map = self.context_builder.build_context_map(full_content_list)

            # Phase 1.5: VLM enrichment — convert ảnh thành caption, inject vào text
            text_only_content = await self.preprocess_content_for_chunking(full_content_list)

            logger.info("Phase 2: RAG Pipeline (Chunking -> Extract -> Graph Insert)...")
            if len(text_only_content) > 0:
                text_json = json.dumps({"content": text_only_content})
                display_name = original_filename or job_id
                await rag_instance.ainsert(text_json, file_paths=display_name)

            logger.info(f"Ingestion Complete for {job_id} in workspace '{workspace}'")
            return {
                "status": "success", "job_id": job_id, "workspace": workspace,
                "text_chunks": len(text_only_content),
            }
        except Exception as e:
            logger.exception(f"Indexing Failed for {job_id}")
            raise e

default_engine = IndexingEngine()