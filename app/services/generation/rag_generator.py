"""
services/generation/rag_generator.py
Trách nhiệm:
  - Format list chunks → array các sources (Citations chuẩn UI AnythingLLM)
  - Load Prompt template .jinja
"""
import os
import re
import uuid
from typing import List, Dict, Optional
from app.utils.logger import get_logger

logger = get_logger("RAG_GENERATOR")

PAGE_CITE_PATTERN = re.compile(r'\[Page\s+(\d+)\]', re.IGNORECASE)

def _get_page(chunk: Dict) -> Optional[int]:
    meta = chunk.get("metadata", {})
    if isinstance(meta, dict):
        try:
            return int(meta["page_idx"]) + 1  # 0-indexed → 1-indexed
        except (KeyError, ValueError, TypeError):
            pass
    # Fallback: parse [Page X] from content
    m = PAGE_CITE_PATTERN.search(chunk.get("content", ""))
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None

def _get_doc_name(chunk: Dict) -> str:
    # Ưu tiên file_path → full_doc_id → fallback
    for key in ("file_path", "full_doc_id", "doc_id", "source"):
        val = chunk.get(key, "")
        if not val:
            continue
        # Bỏ path prefix
        name = val.split("/")[-1].split("\\")[-1]
        if "_" in name:
            parts = name.split("_", 1)
            import re as _re
            if _re.match(r'^[a-f0-9]{8,}$', parts[0]):
                name = parts[1]
        if "." in name:
            name = name.rsplit(".", 1)[0]
        return name.replace("_", " ").replace("-", " ")
    return "Tài liệu"

def format_chunks_as_sources(chunks: List[Dict]) -> List[Dict]:
    """
    Convert LightRAG chunks → AnythingLLM Citations format.
    Citations component expects: {id, title, text, chunkSource, score}
    """
    sources = []
    for chunk in chunks:
        page = _get_page(chunk)
        doc_name = _get_doc_name(chunk)
        title = f"{doc_name} — Trang {page}" if page else doc_name
        content = chunk.get("content", chunk.get("content_with_weight", ""))
        
        # Strip [Page X] prefix from display text
        clean_content = PAGE_CITE_PATTERN.sub("", content).strip()

        sources.append({
            "id": chunk.get("id", str(uuid.uuid4())),
            "title": title,
            "text": clean_content,
            "chunkSource": "",
            "score": chunk.get("total_score", chunk.get("score", None)),
        })

    return sources

from app.config import settings

def load_prompt(filename: str) -> str:
    """Load prompt template from prompts directory."""
    try:
        prompt_path = os.path.join(settings.PROMPTS_DIR, filename)
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load prompt '{filename}': {e}")
        return "{context_data}\n\n{question}" # Fallback minimal prompt

RAG_RESPONSE_TEMPLATE = load_prompt("response_system_prompt.jinja")
NAIVE_RAG_RESPONSE_TEMPLATE = RAG_RESPONSE_TEMPLATE
