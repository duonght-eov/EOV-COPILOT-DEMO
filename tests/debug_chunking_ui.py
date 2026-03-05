import sys
import os
import re
import json
import asyncio
import logging
import gradio as gr
import tempfile

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from app.services.text_chunker import CustomChunker
import importlib
import app.services.text_chunker
importlib.reload(app.services.text_chunker)
from app.services.text_chunker import CustomChunker
from app.services.indexing_engine import IndexingEngine, vlm_model_func, embedding_func, llm_completion_func
from app.services.query_engine import query_engine
import httpx
import uuid
from app.utils.logger import logger, RAGLogger
from app.config import settings
from minio import Minio
from openai import AsyncOpenAI

# Initialize Standard RAG Logging
RAGLogger.setup_logging(log_level="INFO")

# Config Logger ra UI
class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_records = []

    def emit(self, record):
        msg = self.format(record)
        self.log_records.append(msg)
        
    def get_logs(self):
        return "\n".join(self.log_records)
        
    def clear(self):
        self.log_records = []

ui_log_handler = ListHandler()

# Sink for UI
def loguru_sink(message):
    ui_log_handler.log_records.append(message.strip())
logger.add(loguru_sink, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

# Sink for FULL LOG FILE (Everything including Chunker Debug)
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logger.add(os.path.join(log_dir, "debug_full.log"), format="{time:HH:mm:ss} | {level: <8} | {name}:{line} | {message}", level="INFO", encoding="utf-8", rotation="50 MB", mode="w")

logger.info("🚀 FULL SYSTEM LOGGING ENABLED -> logs/debug_full.log")

def cleanup_temp():
    """Clean up previous temp directories from /tmp."""
    import glob
    import shutil
    try:
        # Clean Gradio cache
        shutil.rmtree("/tmp/gradio", ignore_errors=True)
        # Clean specific temp patterns (be careful here)
        # We only clean our own named patterns or Python's default if safe
        # For safety, we only clean known Gradio temp paths or trust tempfile to handle its own context.
        # But user requested explicit cleanup. 
        # Let's clean directories created by us if we can track them, 
        # but since we use tempfile.TemporaryDirectory context, they should be gone.
        # The issue is likely Gradio's cache.
        logger.info("Executed cleanup of /tmp/gradio")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

async def process_json(json_input, file_input, save_to_db=False):
    # cleanup_temp() - DO NOT CLEAN HERE, IT DELETES THE UPLOADED FILE!
    ui_log_handler.clear()
    logger.info(">>> START DEBUGGING <<<")
    logger.info(f"--- CONFIGURATION ---")
    logger.info(f"LLM: {settings.LLM_MODEL_NAME} | URL: {settings.LLM_BASE_URL}")
    logger.info(f"VLM: {settings.VLM_MODEL_NAME} | URL: {settings.VLM_BASE_URL}")
    logger.info(f"Embedding: {settings.EMBEDDING_MODEL_NAME} | URL: {settings.EMBEDDING_SERVICE_URL}")
    logger.info(f"MinIO: {settings.MINIO_ENDPOINT} | Bucket: {settings.MINIO_BUCKET_OCR_RESULTS}")
    logger.info(f"---------------------")
    
    try:
        # 1. Load Data
        json_str = ""
        if file_input is not None:
            logger.info(f"Reading uploaded file: {file_input}")
            with open(file_input, 'r', encoding='utf-8') as f:
                json_str = f.read()
        else:
            json_str = json_input

        if not json_str or not json_str.strip():
            return "Error: Empty Input (Please upload file or paste JSON)", [], [], []

        data = json.loads(json_str)
        logger.info("JSON Loaded successfully.")

        # 1.5. INLINE ENRICHMENT (Using Core Indexing Engine Logic)
        logger.info("="*60)
        logger.info("[STEP 0/3] INLINE ENRICHMENT (CORE ENGINE REUSE)")
        logger.info("="*60)
        
        # Parse content list from JSON
        content_list = []
        doc_obj = data.get("document")
        if isinstance(doc_obj, dict) and "content" in doc_obj:
             for page in doc_obj["content"]:
                p_num = page.get("page_number", 0)
                for b in page.get("blocks", []):
                    b["page_number"] = p_num
                    b["page_idx"] = p_num # Normalize for engine
                    content_list.append(b)
        elif "content" in data:
            content_list = data["content"]
            
        # Initialize Engine for Helper Access
        # Note: We use a dummy doc_id since we check storage later
        engine = IndexingEngine(doc_id="debug_session")
        # Ensure prompts are loaded (engine._ensure_initialized is called in __init__)
        
        # Call the Shared Logic
        logger.info(f"🎨 Delegating enrichment of {len(content_list)} items to IndexingEngine...")
        enriched_content_list = await engine.preprocess_content_for_chunking(content_list)
        
        # 2. Text Chunking
        logger.info("="*60)
        logger.info("[STEP 1/3] TEXT CHUNKING (STYLE-DFS STANDARD)")
        logger.info("="*60)
        
        chunker = CustomChunker()
        # Debug: Check structure of enriched data
        if len(enriched_content_list) > 0:
             logger.info(f"🐛 [DEBUG] First Enriched Item: {json.dumps(enriched_content_list[0], default=str)[:500]}")
        
        enriched_data = {"content": enriched_content_list}
        
        chunks = chunker.process(enriched_data, doc_id="debug_doc")
        text_chunks = chunks 
        
        chunk_results = []
        logger.info(f"✅ Generated {len(chunks)} text chunks (Enriched)")
        for i, c in enumerate(chunks):
            chunk_results.append([
                i,
                c.get("type"),
                c.get("page_idx"),
                c.get("content", "")[:200].replace("\n", " ") + "...",
                len(c.get("content", "").split())
            ])
            # Log full content
            preview = c.get("content", "")
            meta_str = json.dumps(c.get("metadata", {}), ensure_ascii=False)
            logger.info(f"[CHUNK #{i}] Page: {c.get('page_idx')} | Meta: {meta_str}\nCONTENT:\n{preview}\n{'-'*40}")
        
        
        # 3. Multimodal Extraction & VLM
        logger.info("="*60)
        logger.info("[STEP 2/3] MULTIMODAL PROCESSING")
        logger.info("="*60)
        mm_items = []
        
        # Safe extraction logic
        blocks = []
        doc_obj = data.get("document")
        if isinstance(doc_obj, dict) and "content" in doc_obj:
             for page in doc_obj["content"]:
                p_num = page.get("page_number", 0)
                for b in page.get("blocks", []):
                    b["page_number"] = p_num
                    blocks.append(b)
        elif "content" in data:
            blocks = data["content"]
            
        for b in blocks:
            if b.get("type") in ["image", "table"]:
                item = b.copy()
                if "img_path" not in item and "image_path" in item:
                    item["img_path"] = item["image_path"]
                mm_items.append(item)
        
        logger.info(f"✅ Found {len(mm_items)} multimodal items")
        
        # Call VLM with MinIO Fetching
        # DISABLE STEP 2 (Focus on Inline Enrichment Only)
        run_step_2 = False
        if mm_items and run_step_2:
            logger.info(">>> Running VLM/LLM for multimodal content...")
            
            # --- MinIO Download Logic ---
            try:
                minio_client = Minio(
                    settings.MINIO_ENDPOINT,
                    access_key=settings.MINIO_ACCESS_KEY,
                    secret_key=settings.MINIO_SECRET_KEY,
                    secure=settings.MINIO_SECURE
                )
                bucket_name = settings.MINIO_BUCKET_OCR_RESULTS
                
                # Create a temp dir for this run
                with tempfile.TemporaryDirectory() as temp_dir:
                    logger.info(f"Created temp dir for images: {temp_dir}")
                    
                    # Load Multimodal Prompts
                    try:
                        from app.services.indexing_engine import load_jinja_prompts
                        mm_prompt_path = os.path.join(settings.RAG_WORK_DIR, "prompts", "processor_prompts.jinja")
                        if not os.path.exists(mm_prompt_path):
                             mm_prompt_path = "/home/datpt/projects/EOVCopilot-Demo/services/rag-service/prompts/processor_prompts.jinja"
                        mm_prompts = load_jinja_prompts(mm_prompt_path)
                        vision_template = mm_prompts.get("vision_prompt", "")
                        logger.info(f"Loaded vision prompt template: {len(vision_template)} chars")
                    except Exception as ex:
                        logger.error(f"Failed to load user prompts: {ex}")
                        vision_template = ""

                    for idx, item in enumerate(mm_items):
                        i_type = item.get("type", "unknown")
                        i_path = item.get("img_path")
                        
                        logger.info(f"[MM-{idx}] Processing {i_type} at {i_path}...")
                        
                        real_path = None
                        if i_path:
                            # 1. Try local check first (if uploaded)
                            if file_input:
                                base_dir = os.path.dirname(file_input)
                                p1 = os.path.join(base_dir, i_path)
                                if os.path.exists(p1):
                                    real_path = p1
                            
                            # 2. Try MinIO download
                            if not real_path:
                                if not bucket_name:
                                    logger.warning("   -> MinIO Bucket Name is not configured!")
                                else:
                                    try:
                                        # Clean path
                                        obj_name = i_path
                                        if obj_name.startswith("./"):
                                            obj_name = obj_name[2:]
                                        if obj_name.startswith("/"):
                                            obj_name = obj_name[1:]
                                            
                                        if obj_name.startswith(f"{bucket_name}/"):
                                             obj_name = obj_name.replace(f"{bucket_name}/", "", 1)
                                        
                                        local_file = os.path.join(temp_dir, os.path.basename(i_path))
                                        logger.info(f"   Downloading from MinIO: {obj_name} -> {local_file}")
                                        
                                        minio_client.fget_object(bucket_name, obj_name, local_file)
                                        real_path = local_file
                                        logger.info("   -> Download Success")
                                    except Exception as me:
                                        logger.warning(f"   -> MinIO Download Failed: {me}")
                        
                        # Handle Missing Image Path
                        else:
                             if i_type == "table":
                                 if "table_body" in item and item["table_body"]:
                                     logger.info(f"[MM-{idx}] Found Table Body (Markdown). Using LLM to summarize...")
                                     table_content = item["table_body"]
                                     
                                     # Use table prompt template
                                     table_template = mm_prompts.get("table_prompt", "")
                                     if table_template:
                                         try:
                                             prompt = table_template.format(
                                                 table_img_path="N/A",
                                                 table_caption="",
                                                 table_body=table_content,
                                                 table_footnote="",
                                                 entity_name="Table"
                                             )
                                         except Exception as e:
                                              logger.warning(f"Table prompt format failed: {e}")
                                              prompt = f"Hãy phân tích và tóm tắt chi tiết nội dung của bảng dữ liệu sau đây bằng Tiếng Việt:\n\n{table_content}"
                                     else:
                                         prompt = f"Hãy phân tích và tóm tắt chi tiết nội dung của bảng dữ liệu sau đây bằng Tiếng Việt:\n\n{table_content}"
                                     try:
                                         # Use LLM for text-only tables
                                         summary = await llm_completion_func(prompt)
                                         logger.info(f"   --> LLM Table Summary:\n{summary}")
                                         item["VLM Description"] = f"[Table Summary (LLM)]:\n{summary}\n\n[Original Content]:\n{table_content[:500]}..."
                                     except Exception as le:
                                         logger.error(f"   --> LLM Summary Failed: {le}")
                                         item["VLM Description"] = f"LLM Error: {le}"
                                     continue
                                 elif "html" in item:
                                     logger.info("   -> Using HTML content as fallback.")
                                     item["VLM Description"] = f"[HTML Source]:\n{item['html'][:500]}..." 
                                     continue
                                 else:
                                     item["VLM Description"] = "Skipped: No Image, Table Body, or HTML"
                                     continue
                             else:
                                 item["VLM Description"] = "Skipped: No Img Path"
                                 continue

                        # 3. Call VLM if we have a file
                        if real_path and os.path.exists(real_path):
                            # Update Prompt to Vietnamese
                            if i_type == "image":
                                if vision_template:
                                    # Use loaded template
                                    try:
                                        prompt = vision_template.format(
                                            image_path=i_path or "",
                                            captions="",
                                            footnotes="",
                                            entity_name=os.path.basename(real_path)
                                        )
                                    except:
                                        # Fallback if format fails
                                        prompt = "Hãy mô tả chi tiết hình ảnh này bằng Tiếng Việt. Tập trung vào các đối tượng chính, hoạt động, và bất kỳ văn bản nào xuất hiện trong ảnh."
                                else:
                                    prompt = "Hãy mô tả chi tiết hình ảnh này bằng Tiếng Việt. Tập trung vào các đối tượng chính, hoạt động, và bất kỳ văn bản nào xuất hiện trong ảnh."
                            else:
                                prompt = "Hãy chuyển đổi nội dung bảng trong hình ảnh này sang định dạng Markdown. Giữ nguyên cấu trúc dòng và cột."
                            
                            try:
                                description = await vlm_model_func(prompt, images=[real_path])
                                logger.info(f"   --> VLM Output:\n{description}")
                                item["VLM Description"] = description
                            except Exception as ve:
                                logger.error(f"   --> VLM Error: {ve}")
                                item["VLM Description"] = f"Error: {ve}"
                        else:
                            if i_type == "table" and "html" in item:
                                # Fallback HTML for table even if download failed
                                item["VLM Description"] = f"[HTML Source (Fallback)]:\n{item['html'][:500]}..."
                            else:
                                logger.warning("   --> Image file not accessible.")
                                item["VLM Description"] = "Image Missing / Download Failed"

            except Exception as e:
                logger.error(f"MinIO/VLM Pipeline Error: {e}")
        
        # 4. ENTITY EXTRACTION (NEW STEP 3)
        entity_results = []
        logger.info("="*60)
        logger.info("[STEP 3/3] ENTITY EXTRACTION")
        logger.info("="*60)
        
        try:
            # Import helper to load prompts
            from app.services.indexing_engine import load_jinja_prompts
            prompt_path = os.path.join(settings.RAG_WORK_DIR, "prompts", "prompt_extractor.jinja")
            if not os.path.exists(prompt_path):
                 # Fallback path if env var not set correctly in test
                 prompt_path = "/home/datpt/projects/EOVCopilot-Demo/services/rag-service/prompts/prompt_extractor.jinja"
            
            custom_prompts = load_jinja_prompts(prompt_path)
            
            if "entity_extraction_system_prompt" not in custom_prompts:
                logger.warning("⚠️ No entity extraction prompts found. Skipping entity extraction.")
            else:
                logger.info(f"✅ Loaded entity extraction prompts")
                
                # Prepare chunks for extraction
                extraction_candidates = []
                
                # Add text chunks
                for idx, chunk in enumerate(text_chunks):
                    # Accept 'text' or 'mixed' (since chunks often contain context headers)
                    c_type = chunk.get("metadata", {}).get("type", chunk.get("type", "text"))
                    if c_type in ["text", "mixed"]:
                        extraction_candidates.append({
                            "source": "text_chunk",
                            "index": idx,
                            "content": chunk.get("content", ""),
                            "page": chunk.get("page_idx", 0)
                        })
                
                # Add multimodal descriptions
                for idx, mm_item in enumerate(mm_items):
                    desc = mm_item.get("VLM Description", "")
                    # Accept all descriptions, even if they have warnings
                    if desc and len(desc.strip()) > 20:  # Minimum content check
                        extraction_candidates.append({
                            "source": "multimodal",
                            "index": idx,
                            "type": mm_item.get("type"),
                            "content": desc,
                            "page": mm_item.get("page_number", 0)
                        })
                
                # Count by source type
                text_count = sum(1 for c in extraction_candidates if c["source"] == "text_chunk")
                mm_count = sum(1 for c in extraction_candidates if c["source"] == "multimodal")
                
                logger.info(f"📋 Entity extraction candidates:")
                logger.info(f"   • Text chunks: {text_count}")
                logger.info(f"   • Multimodal items: {mm_count}")
                logger.info(f"   • Total: {len(extraction_candidates)}")
                
                # Construct Prompt Templates
                sys_prompt = custom_prompts["entity_extraction_system_prompt"]
                user_prompt_template = custom_prompts["entity_extraction_user_prompt"]
                
                entity_types = ["System", "Component", "FailureMode", "Cause", "MaintenanceAction", "Tool", "Material", "Parameter", "ErrorCode", "SafetyHazard", "SafetyMeasure", "Document", "Location", "Role", "Event", "Time", "Concept"]
                
                # Handle Examples
                examples_content = custom_prompts.get("entity_extraction_examples", "")
                if "{examples}" in sys_prompt:
                    sys_prompt = sys_prompt.replace("{examples}", examples_content)
                elif "{{ examples }}" in sys_prompt:
                    sys_prompt = sys_prompt.replace("{{ examples }}", examples_content)

                # Handle Entity Types in system prompt
                if "{entity_types}" in sys_prompt:
                    sys_prompt = sys_prompt.replace("{entity_types}", str(entity_types))
                else:
                    sys_prompt = sys_prompt.replace("{{ entity_types }}", str(entity_types))
                
                # Process all chunks for comprehensive debugging
                total_entities = 0
                total_relations = 0
                max_chunks_to_process = 200  # Increase limit to process all items
                
                # Order: Text chunks then Multimodal chunks
                text_chunks_list = [c for c in extraction_candidates if c["source"] == "text_chunk"]
                mm_chunks_list = [c for c in extraction_candidates if c["source"] == "multimodal"]
                
                chunks_to_process = text_chunks_list + mm_chunks_list
                chunks_to_process = chunks_to_process[:max_chunks_to_process]
                
                logger.info(f"📊 Processing {len(chunks_to_process)} chunks for entity extraction")
                
                for cand in chunks_to_process:
                    content = cand["content"]
                    
                    logger.info(f"\n{'='*50}")
                    if cand["source"] == "multimodal":
                        logger.info(f"Extracting from MULTIMODAL ({cand.get('type', 'unknown').upper()}) [{cand['index']}] (Page {cand['page']})")
                    else:
                        logger.info(f"Extracting from {cand['source'].upper()} [{cand['index']}] (Page {cand['page']})")
                        
                        # DEBUG: List Images in Chunk
                        import re
                        images_in_chunk = re.findall(r"\[\[IMAGE_REF: (.*?)\]\]", content)
                        if images_in_chunk:
                            logger.info(f"🖼️  IMAGES IN THIS CHUNK ({len(images_in_chunk)}):")
                            for img in images_in_chunk:
                                logger.info(f"   -> {os.path.basename(img)}")
                        else:
                            logger.info("🖼️  NO IMAGES IN THIS CHUNK")
                            
                        # DEBUG: Print Text Chunk Content
                        logger.info(f"\n📄 CHUNK CONTENT FULL VIEW:\n{'-'*40}\n{content[:2000]}\n{'-'*40}")
                        if len(content) > 2000:
                            logger.info(f"...And {len(content)-2000} more chars...")
                            
                    logger.info(f"{'='*50}")

                    if len(content) > 3000:  # Limit content size
                        content = content[:3000] + "..."
                    
                    # Prepare user prompt
                    user_prompt = user_prompt_template
                    
                    # Replace entity types
                    if "{entity_types}" in user_prompt:
                        user_prompt = user_prompt.replace("{entity_types}", str(entity_types))
                    else:
                        user_prompt = user_prompt.replace("{{ entity_types }}", str(entity_types))
                    
                    # Replace delimiters
                    user_prompt = user_prompt.replace("{{ tuple_delimiter }}", "<|#|>")
                    user_prompt = user_prompt.replace("{{ completion_delimiter }}", "<|COMPLETE|>")
                    
                    # Replace input text
                    if "{input_text}" in user_prompt:
                        user_prompt = user_prompt.replace("{input_text}", content)
                    else:
                        user_prompt = user_prompt.replace("{{ input_text }}", content)
                    
                    # === RETRY: Based on parse results ===
                    entities = []
                    relations = []
                    max_retries = 3
                    
                    for retry in range(max_retries):
                        if retry > 0:
                            logger.warning(f"🔄 [RETRY] Attempt {retry+1}/{max_retries} (parse failed)...")
                            await asyncio.sleep(1)
                        
                        # Call LLM
                        extraction_result = await llm_completion_func(user_prompt, system_prompt=sys_prompt)
                        
                        # CLEAN THINKING PROCESS (DeepSeek R1/V3)
                        extraction_result = re.sub(r'<think>.*?</think>', '', extraction_result, flags=re.DOTALL).strip()
                        if extraction_result.startswith("Based on the input"):
                             extraction_result = extraction_result.split("\n", 1)[-1]

                        entities = []
                        relations = []
                        
                        for line in extraction_result.split('\n'):
                            line = line.strip().lstrip('*').lstrip('-').strip()
                            # TRUST THE INDEXING ENGINE NORMALIZATION
                            # The engine already converts ####, <|>, etc. to <|#|>
                            curr_delimiter = "<|#|>"
                            
                            if line.startswith(f'entity{curr_delimiter}'):
                                parts = line.split(curr_delimiter)
                                if len(parts) >= 4:
                                    entities.append({"name": parts[1], "type": parts[2], "description": parts[3] if len(parts) > 3 else ""})
                            elif line.startswith(f'relation{curr_delimiter}'):
                                parts = line.split(curr_delimiter)
                                if len(parts) >= 5:
                                    relations.append({"source": parts[1], "target": parts[2], "keywords": parts[3], "description": parts[4] if len(parts) > 4 else ""})
                            elif line and (line.startswith("entity") or line.startswith("relation")):
                                # Fallback: maybe delimiter normalization failed?
                                # Let's try flexible split just in case
                                if "<|>" in line and "<|#|>" not in line:
                                     parts = line.split("<|>")
                                     if line.startswith("entity") and len(parts) >= 3:
                                          entities.append({"name": parts[1], "type": parts[2], "description": parts[3] if len(parts) > 3 else ""})
                                     elif line.startswith("relation") and len(parts) >= 4:
                                          relations.append({"source": parts[1], "target": parts[2], "keywords": parts[3], "description": parts[4] if len(parts) > 4 else ""})
                                else:
                                     logger.warning(f"⚠️ PARSE FAIL: '{line[:100]}...'")
                        
                        # Check success
                        if len(entities) > 0 or len(relations) > 0:
                            if retry > 0:
                                logger.info(f"✅ [PARSE SUCCESS] Retry OK after {retry} attempts")
                            break
                        else:
                            logger.warning(f"⚠️ [PARSE FAILED] 0 entities from {len(extraction_result)} chars")
                            if retry < max_retries - 1:
                                logger.warning(f"   Preview: {extraction_result[:200]}...")
                                logger.warning(f"   🔄 Retrying parse attempt {retry+2}/{max_retries}...")
                                # Fallback: Add strong reminder to prompt
                                user_prompt += "\n\nIMPORTANT: You must parse the content and output entities using 'entity<|#|>' format. Do not return just text."
                            else:
                                logger.error(f"❌ [PARSE FAILED] Gave up after {max_retries} attempts.")
                                logger.error(f"   Full Response Preview:\n{extraction_result[:500]}...")
                    
                    logger.info(f"✅ Extracted {len(entities)} entities, {len(relations)} relations")
                    
                    # Log entities
                    for ent in entities:
                        logger.info(f"  📌 {ent['type']}: {ent['name']}")
                        entity_results.append([
                            f"{cand['source']} ({cand.get('type', 'text')})" if cand['source'] == 'multimodal' else cand['source'],
                            cand['index'],
                            cand['page'],
                            ent['type'],
                            ent['name'],
                            ent['description'][:100] + "..." if len(ent['description']) > 100 else ent['description']
                        ])
                    
                    # Log relations
                    for rel in relations:
                        logger.info(f"  🔗 {rel['source']} --[{rel['keywords']}]--> {rel['target']}")
                    
                    total_entities += len(entities)
                    total_relations += len(relations)
                
                logger.info(f"\n{'='*60}")
                logger.info(f"📊 EXTRACTION SUMMARY:")
                logger.info(f"   Total Entities: {total_entities}")
                logger.info(f"   Total Relations: {total_relations}")
                logger.info(f"   From Text Chunks: {sum(1 for r in entity_results if 'text_chunk' in r[0])}")
                logger.info(f"   From Multimodal: {sum(1 for r in entity_results if 'multimodal' in r[0])}")
                logger.info(f"{'='*60}")

        except Exception as ee:
            logger.error(f"Entity Extraction Failed: {ee}")
            import traceback
            logger.error(traceback.format_exc())
                
        # 4. SAVE TO DB (Optional)
        if save_to_db:
            logger.info("="*60)
            logger.info("💾 [STEP 4/4] DATABASE INSERTION")
            logger.info("="*60)
            try:
                doc_id = f"debug_ui_{uuid.uuid4().hex[:8]}"
                engine = IndexingEngine(doc_id=doc_id)
                await engine._ensure_storages_initialized()
                logger.info(f"Inserting to {settings.STORAGE_TYPE} (Graph: {settings.ENABLE_GRAPH_STORAGE})...")
                
                # Use the engine's full pipeline to ensure consistency
                # This ensures enrichment, proper chunking, and multimodal indexing are applied.
                
                # OPTIMIZATION: Use PRE-ENRICHED data to skip VLM re-processing
                final_data = data.copy()
                if 'enriched_content_list' in locals() and enriched_content_list:
                    final_data["content"] = enriched_content_list
                    logger.info("⚡ Using Pre-Enriched Data for Indexing (Skipping VLM)...")
                
                logger.info(f"Triggering Full Indexing Pipeline for {doc_id}...")
                
                await engine.index_document(ocr_data=final_data, job_id=doc_id)
                logger.info(f"✅ Database Insertion Success! DocID: {doc_id}")
            except Exception as dbe:
                logger.error(f"❌ Database Insertion Failed: {dbe}")
                import traceback
                logger.error(traceback.format_exc())

        logger.info("\n>>> FINISHED <<<")
        
        mm_display = []
        for x in mm_items:
            mm_display.append([
                x.get("type"),
                x.get("page_number"),
                x.get("img_path"),
                x.get("VLM Description", "N/A")
            ])
            
        return ui_log_handler.get_logs(), chunk_results, mm_display, entity_results

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return ui_log_handler.get_logs(), [], [], []

async def check_infrastructure():
    """Kiểm tra kết nối tới các service hạ tầng"""
    logs = []
    logger.info(">>> STARTING INFRASTRUCTURE HEALTH CHECK <<<")
    
    # 1. Check LLM
    try:
        logger.info(f"Checking LLM at {settings.LLM_BASE_URL}...")
        client = AsyncOpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)
        resp = await client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": "Ping"}],
            max_tokens=1
        )
        logs.append(f"✅ LLM Service: Connected ({settings.LLM_MODEL_NAME})")
    except Exception as e:
        logs.append(f"❌ LLM Service: Failed - {str(e)}")
        
    # 2. Check Embedding
    try:
        logger.info(f"Checking Embedding Service...")
        # embedding_func is asyncwrapper or sync? LightRAG wrapper is usually sync or async depending on implementation.
        # In indexing_engine.py: async def embedding_func(texts: list[str]) -> np.ndarray
        res = await embedding_func(["test"])
        if res is not None and len(res) > 0:
            dims = len(res[0])
            logs.append(f"✅ Embedding Service: Connected (Dims: {dims})")
        else:
            logs.append("❌ Embedding Service: Connected but returned empty result")
    except Exception as e:
        logs.append(f"❌ Embedding Service: Failed - {str(e)}")

    # 3. Check VLM
    try:
        logger.info(f"Checking VLM Service at {settings.VLM_BASE_URL}...")
        # We reuse the VLM function but with text only to check connection
        # Or simple OpenAI call if VLM uses same protocol
        client_vlm = AsyncOpenAI(api_key=settings.VLM_API_KEY, base_url=settings.VLM_BASE_URL)
        await client_vlm.chat.completions.create(
            model=settings.VLM_MODEL_NAME,
            messages=[{"role": "user", "content": [{"type": "text", "text": "Ping"}]}],
            max_tokens=1
        )
        logs.append(f"✅ VLM Service: Connected ({settings.VLM_MODEL_NAME})")
    except Exception as e:
        logs.append(f"❌ VLM Service: Failed - {str(e)}")

    # 4. Check MinIO
    try:
        logger.info(f"Checking MinIO at {settings.MINIO_ENDPOINT}...")
        mc = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )
        buckets = mc.list_buckets()
        b_names = [b.name for b in buckets]
        logs.append(f"✅ MinIO Storage: Connected (Buckets: {len(b_names)})")
    except Exception as e:
        logs.append(f"❌ MinIO Storage: Failed - {str(e)}")

    return "\n".join(logs)

# --- UI Interface ---
with gr.Blocks(title="RAG Service Debugger") as demo:
    gr.Markdown("# 🛠️ RAG Service Debugger & Entity Extraction Viewer")
    
    with gr.Tabs():
        with gr.TabItem("🧩 Chunking & Entity Extraction"):
            gr.Markdown("""
            ### Pipeline Steps:
            1. **Text Chunking** - Phân tách text thành chunks với context
            2. **Multimodal Processing** - VLM phân tích images/tables  
            3. **Entity Extraction** - Trích xuất entities từ text + multimodal descriptions
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_file = gr.File(label="Upload JSON File", file_types=[".json"], type="filepath")
                    input_json = gr.Code(language="json", label="Or Paste OCR JSON Result Here", lines=10)
                    chk_save_db = gr.Checkbox(label="💾 Save to Database (Postgres/Neo4j)", value=True)
                    btn_run = gr.Button("🚀 Run Full Pipeline", variant="primary")
                
                with gr.Column(scale=1):
                    logs_out = gr.TextArea(label="Realtime Logs", lines=20, interactive=False)
            
            gr.Markdown("### 📊 Results")
            
            with gr.Tab("Text Chunks"):
                chunk_out = gr.Dataframe(
                    label="Text Chunks Result", 
                    headers=["Index", "Type", "Page", "Content Preview", "Tokens (Approx)"], 
                    wrap=True
                )
            
            with gr.Tab("Multimodal Items"):
                mm_out = gr.Dataframe(
                    label="Multimodal Items & VLM Analysis", 
                    headers=["Type", "Page", "Path", "VLM Output"], 
                    wrap=True
                )
            
            with gr.Tab("🔍 Extracted Entities"):
                entity_out = gr.Dataframe(
                    label="Entity Extraction Results",
                    headers=["Source", "Chunk Index", "Page", "Entity Type", "Entity Name", "Description"],
                    wrap=True
                )

            btn_run.click(
                process_json, 
                inputs=[input_json, input_file, chk_save_db], 
                outputs=[logs_out, chunk_out, mm_out, entity_out]
            )

        with gr.TabItem("🏥 Infrastructure Health"):
            gr.Markdown("Click button below to verify connections to all external services (LLM, Embedding, Storage).")
            btn_health = gr.Button("kiểm tra hạ tầng (Check Infrastructure)", variant="secondary")
            health_out = gr.Code(language="markdown", label="Health Status")
            
            btn_health.click(check_infrastructure, inputs=[], outputs=[health_out])

        with gr.TabItem("💬 Chat / Query"):
            gr.Markdown("### 🤖 Chat with RAG Knowledge Base")
            
            with gr.Row():
                with gr.Column(scale=4):
                    # Chatbot component with specific type for messages
                    chatbot = gr.Chatbot(
                        height=800, 
                        type="messages",
                        show_copy_button=True,
                        avatar_images=(None, "https://api.dicebear.com/9.x/bottts-neutral/svg?seed=RAG")
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Ask a question...", 
                            placeholder="e.g. SpeedMaint là gì? / What is in the picture?", 
                            lines=2,
                            scale=4
                        )
                        btn_send = gr.Button("Send 🚀", variant="primary", scale=1)

                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Settings")
                    mode_dropdown = gr.Dropdown(
                        choices=["naive", "local", "global", "mix", "consensus", "graph_aware"],
                        value="mix",
                        label="Retrieval Mode",
                        info="Select the RAG retrieval strategy."
                    )
                    
                    with gr.Accordion("Mode Details", open=False):
                        gr.Markdown("""
                        - **naive**: Vector search only.
                        - **local**: Entity-based search.
                        - **global**: Community summary search.
                        - **mix**: Hybrid (Local + Global).
                        - **consensus**: High precision (Naive ∩ Local).
                        - **graph_aware**: Multimodal Graph Expansion.
                        """)
                    
                    btn_clear = gr.Button("🗑️ Clear History")

            def _download_images_from_minio(image_refs: list) -> list:
                """Tải ảnh từ MinIO, trả về list base64 data URIs."""
                import base64
                data_uris = []
                try:
                    minio_client = Minio(
                        settings.MINIO_ENDPOINT,
                        access_key=settings.MINIO_ACCESS_KEY,
                        secret_key=settings.MINIO_SECRET_KEY,
                        secure=settings.MINIO_SECURE
                    )
                    tmp_dir = os.path.join(tempfile.gettempdir(), "rag_images")
                    os.makedirs(tmp_dir, exist_ok=True)

                    for obj_key in image_refs:
                        filename = obj_key.replace("/", "_")
                        local_path = os.path.join(tmp_dir, filename)
                        if not os.path.exists(local_path):
                            try:
                                minio_client.fget_object(
                                    settings.MINIO_BUCKET_OCR_RESULTS,
                                    obj_key,
                                    local_path
                                )
                            except Exception as e:
                                logger.warning(f"Cannot download image {obj_key}: {e}")
                                continue
                        try:
                            ext = os.path.splitext(local_path)[1].lower()
                            mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
                            with open(local_path, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode()
                            data_uris.append(f"data:{mime};base64,{b64}")
                        except Exception as e:
                            logger.warning(f"Cannot encode image {local_path}: {e}")
                except Exception as e:
                    logger.error(f"MinIO connection error: {e}")
                return data_uris

            async def user_message(user_msg, history):
                if not user_msg:
                    return history, ""
                if history is None:
                    history = []
                history.append({"role": "user", "content": user_msg})
                return history, ""

            async def bot_response(history, mode):
                if not history:
                    yield history
                    return
                
                user_msg = history[-1]["content"]
                
                try:
                    history.append({"role": "assistant", "content": "⏳ **Đang xử lý...**"})
                    yield history

                    if not query_engine._initialized:
                        await query_engine.initialize()
                    
                    result_dict = await query_engine.query(user_msg, mode=mode)
                    
                    final_answer = result_dict.get("answer", "No response generated.")
                    
                    debug_info = ""

                    image_refs = result_dict.get("images", [])
                    data_uris = []
                    if image_refs:
                        loop = asyncio.get_event_loop()
                        data_uris = await loop.run_in_executor(
                            None, _download_images_from_minio, image_refs
                        )

                    # Build message content với ảnh (nếu có)
                    if data_uris:
                        img_md = "\n\n---\n### 🖼️ Minh họa liên quan:\n"
                        for data_uri in data_uris:
                            img_md += f"![]({data_uri})\n\n"
                        history[-1]["content"] = final_answer + debug_info + img_md
                    else:
                        history[-1]["content"] = final_answer + debug_info

                    yield history
                    
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    history[-1]["content"] = f"❌ **Lỗi xảy ra:**\n```\n{str(e)}\n```\n<details><summary>Traceback</summary>\n\n```python\n{tb}\n```\n</details>"
                    yield history

            # Event Handling
            msg.submit(
                user_message, [msg, chatbot], [chatbot, msg], queue=False
            ).then(
                bot_response, [chatbot, mode_dropdown], chatbot
            )
            
            btn_send.click(
                user_message, [msg, chatbot], [chatbot, msg], queue=False
            ).then(
                bot_response, [chatbot, mode_dropdown], chatbot
            )
            
            btn_clear.click(lambda: [], None, chatbot, queue=False)


if __name__ == "__main__":
    cleanup_temp()
    tmp_image_dir = os.path.join(tempfile.gettempdir(), "rag_images")
    os.makedirs(tmp_image_dir, exist_ok=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[tmp_image_dir])