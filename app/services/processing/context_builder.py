from typing import List, Dict, Any
from app.utils.logger import get_logger

logger = get_logger("CONTEXT BUILDER")

class ContextBuilder:
    def __init__(self, context_window: int=2, max_context_chars: int = 500):
        self.context_window = context_window
        self.max_context_chars = max_context_chars
    
    def extract_full_content_list(self, ocr_data: Dict) -> List[Dict]:
        content_list = []
        doc_obj = ocr_data.get("document")
        if isinstance(doc_obj, dict) and "content" in doc_obj:
            for page in doc_obj["content"]:
                p_num = page.get("page_number", 0)
                for block in page.get("blocks", []):
                    block["page_idx"] = p_num
                    content_list.append(block)

        elif "content" in ocr_data:
            content_list = ocr_data["content"]
            for item in content_list:
                if "page_idx" not in item and "page_number" in item:
                    item["page_idx"] = item["page_number"]

        logger.info(f"Extracted {len(content_list)} content blocks")
        return content_list
    
    def build_context_map(self, full_content_list: List[Dict]) -> Dict[int, List[Dict]]:
        context_map = {}
        for item in full_content_list:
            if item.get("type", "unknown") not in ["image", "table"]:
                page_idx = item.get("page_idx", 0)

                if page_idx not in context_map:
                    context_map[page_idx] = []
                
                context_map[page_idx].append({
                    "text": item.get("text", "").strip(),
                    "page_idx": page_idx,
                    "order": len(context_map[page_idx])
                })

        logger.info(f"Built context map for {len(context_map)} pages")
        return context_map
    
    def get_context_for_item(
        self, 
        mm_item: Dict, 
        context_map: Dict[int, List[Dict]]
    ) -> Dict[str, Any]:
        """
        Get context for a multimodal item.
        
        Returns:
            {
                "context_text": str,  # Combined context string
                "context_chunks": List[Dict],  # Individual context chunks
                "page_range": Tuple[int, int]  # (start_page, end_page)
            }
        """
        page_idx = mm_item.get("page_idx", 0)
        
        # Collect text from current page and adjacent pages
        context_chunks = []
        start_page = max(0, page_idx - 1)
        end_page = page_idx + 1
        
        for p in range(start_page, end_page + 1):
            if p in context_map:
                page_texts = context_map[p]
                
                # Limit number of chunks per page
                for chunk in page_texts[:self.context_window]:
                    text = chunk["text"][:self.max_context_chars]
                    if text:
                        context_chunks.append({
                            "text": text,
                            "page_idx": p,
                            "distance": abs(p - page_idx)  # Distance from mm_item
                        })
        
        # Sort by distance (closer pages first)
        context_chunks.sort(key=lambda x: x["distance"])
        
        # Build combined context string
        context_parts = [
            f"[Page {chunk['page_idx']}] {chunk['text']}"
            for chunk in context_chunks[:self.context_window * 2]
        ]
        context_text = "\n\n".join(context_parts)
        
        return {
            "context_text": context_text,
            "context_chunks": context_chunks,
            "page_range": (start_page, end_page)
        }
    
    def enrich_multimodal_items(
        self,
        mm_items: List[Dict],
        context_map: Dict[int, List[Dict]]
    ) -> List[Dict]:
        """
        Enrich multimodal items with context.
        Injects context into captions and stores metadata.
        """
        enriched_items = []
        
        for mm_item in mm_items:
            context_info = self.get_context_for_item(mm_item, context_map)
            context_text = context_info["context_text"]
            
            # Clone item
            enriched_item = mm_item.copy()
            mm_type = enriched_item.get("type", "unknown")
            
            # Inject context based on type
            if mm_type == "image":
                original_caption = enriched_item.get("image_caption", [])
                if isinstance(original_caption, list):
                    original_caption = ", ".join(original_caption)
                
                # Prepend context
                enriched_caption = f"Context: {context_text}\n\nOriginal: {original_caption}"
                enriched_item["image_caption"] = [enriched_caption]
                
            elif mm_type == "table":
                original_caption = enriched_item.get("table_caption", [])
                if isinstance(original_caption, list):
                    original_caption = ", ".join(original_caption)
                
                enriched_caption = f"Context: {context_text}\n\nOriginal: {original_caption}"
                enriched_item["table_caption"] = [enriched_caption]
            
            # Store metadata for graph enhancement later
            enriched_item["_context_metadata"] = {
                "context_text": context_text,
                "context_page_range": context_info["page_range"],
                "num_context_chunks": len(context_info["context_chunks"])
            }
            
            enriched_items.append(enriched_item)
            
            logger.debug(
                f"Enriched {mm_type} at page {mm_item.get('page_idx')} "
                f"with {len(context_text)} chars context"
            )
        
        return enriched_items

