"""
services/processing/prompt_loader.py
Trách nhiệm:
  - load_jinja_prompts: Parse file .jinja và extract blocks
  - Cache prompt config (singleton)
  - Resolve đường dẫn file prompt theo thứ tự ưu tiên
"""
import os
import re
from pathlib import Path
from typing import Dict
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("PROMPT LOADER")
_prompt_config_cache = None

def load_jinja_prompts(file_path: str) -> Dict[str, str]:
    """Parse file Jinja2 và trích xuất block {% block name %}...{% endblock %}."""
    prompts = {}
    if not os.path.exists(file_path):
        logger.warning(f"Prompt file not found: {file_path}")
        return prompts
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        pattern = re.compile(r"{% block (\w+) %}(.*?){% endblock %}", re.DOTALL)
        matches = pattern.findall(content)

        # Regex nhận diện {placeholder} hợp lệ (word chars only)
        _placeholder_re = re.compile(r"\{(\w+)\}")

        def _escape_non_placeholder_braces(text: str) -> str:
            """Escape { và } mà không phải placeholder {word} để tránh lỗi str.format()."""
            # Bước 1: resolve Jinja escape sequences {{ "{{" }} → { và {{ "}}" }} → }
            text = re.sub(r'\{\{\s*"\{\{"\s*\}\}', '{', text)
            text = re.sub(r'\{\{\s*"\}\}"\s*\}\}', '}', text)
            # Bước 2: bảo vệ các {placeholder} khỏi bị double-escape
            protected = _placeholder_re.sub(lambda m: f"\x00{m.group(1)}\x00", text)
            # Bước 3: escape ngoặc nhọn còn lại thành {{ và }}
            protected = protected.replace("{", "{{").replace("}", "}}")
            # Bước 4: restore placeholder
            result = re.sub(r'\x00(\w+)\x00', r'{\1}', protected)
            return result

        for block_name, block_content in matches:
            cleaned = block_content.strip()
            cleaned = cleaned.replace("{{ tuple_delimiter }}", "<|>")
            cleaned = cleaned.replace("{{ completion_delimiter }}", "<|COMPLETE|>")
            cleaned = re.sub(r"\{\{ '\{(\w+)\}' \}\}", r"{\1}", cleaned)
            cleaned = _escape_non_placeholder_braces(cleaned)
            prompts[block_name] = cleaned
            
        logger.info(f"Loaded {len(prompts)} custom prompts from {file_path}")
        return prompts
    except Exception as e:
        logger.error(f"Failed to load prompts: {e}")
        return {}

def get_prompt_config() -> Dict[str, str]:
    """Load và tính toán (build) danh sách prompts cần thiết, có cơ chế cache singleton."""
    global _prompt_config_cache
    if _prompt_config_cache is not None:
        return _prompt_config_cache
    _prompts_src_dir = Path(settings.PROMPTS_DIR)
    def _resolve_prompt_path(filename: str) -> str:
        """Kiểm tra RAG_WORK_DIR trước, rồi mới xài default prompts."""
        work_dir_path = Path(settings.RAG_WORK_DIR) / "prompts" / filename
        if work_dir_path.exists(): return str(work_dir_path)
        return str(_prompts_src_dir / filename)

    prompt_extractor_path = _resolve_prompt_path("prompt_extractor.jinja")
    custom_prompts = load_jinja_prompts(prompt_extractor_path)
    vlm_prompt_path = _resolve_prompt_path("vlm_prompts.jinja")
    vlm_prompts = load_jinja_prompts(vlm_prompt_path)
    processor_prompt_path = _resolve_prompt_path("processor_prompts.jinja")
    multimodal_prompts = load_jinja_prompts(processor_prompt_path)
    
    if multimodal_prompts:
        try:
            import raganything.prompt
            logger.info(f"Overriding RAGAnything prompts with {len(multimodal_prompts)} templates")
            raganything.prompt.PROMPTS.update(multimodal_prompts)
        except ImportError as e:
            logger.warning(f"Could not update raganything prompts: {e}")

    p_entity_extract = ""
    if "entity_extraction_system_prompt" in custom_prompts:
        p_entity_extract += custom_prompts["entity_extraction_system_prompt"] + "\n"
    if "entity_extraction_user_prompt" in custom_prompts:
        p_entity_extract += custom_prompts["entity_extraction_user_prompt"]
    if "entity_extraction_examples" in custom_prompts and "{examples}" in p_entity_extract:
        p_entity_extract = p_entity_extract.replace("{examples}", custom_prompts["entity_extraction_examples"])
    p_keywords = custom_prompts.get("keywords_extraction", "")
    if "keywords_extraction_examples" in custom_prompts and "{examples}" in p_keywords:
        p_keywords = p_keywords.replace("{examples}", custom_prompts["keywords_extraction_examples"])

    _prompt_config_cache = {
        "vlm_prompts": vlm_prompts,
        "entity_extract": p_entity_extract,
        "entity_summary": custom_prompts.get("summarize_entity_descriptions"),
        "rag_response": custom_prompts.get("rag_response"),
        "naive_rag_response": custom_prompts.get("naive_rag_response"),
        "keywords": p_keywords,
    }
    
    logger.info("Prompt config built and cached successfully.")
    return _prompt_config_cache