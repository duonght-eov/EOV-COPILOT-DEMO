"""
services/indexing/lightrag_adapter.py
Trách nhiệm:
  - lightrag_chunking_adapter: Adapter giữa LightRAG chunking API và CustomChunker
  - RAGAnything setup và configuration
"""
from typing import List, Dict, Any
from app.services.processing.text_chunker import CustomChunker, tokenizer

def lightrag_chunking_adapter(*args, **kwargs) -> List[Dict[str, Any]]:
    """Adapter API format 6-tham-số từ lightrag gốc sang CustomChunker của hệ thống."""
    text = args[1] if len(args) >= 2 and isinstance(args[1], str) else kwargs.get("content", args[0] if args else "")
    chunk_token_size = args[5] if len(args) >= 6 else kwargs.get("chunk_token_size", 1200)
    chunk_overlap_token_size = args[4] if len(args) >= 5 else kwargs.get("chunk_overlap_token_size", 100)
    
    chunker = CustomChunker(
        target_chunk_size=chunk_token_size,
        chunk_overlap=chunk_overlap_token_size
    )
    
    chunks = chunker.process(text, doc_id="embedded_doc")
    results = []
    
    for idx, c in enumerate(chunks):
        content = c.get("content", "")
        tokens = tokenizer.count(content)
            
        results.append({
            "content": content,
            "tokens": tokens,
            "chunk_order_index": idx,
            "source_id": f"chunk-{idx}", 
            "metadata": {
                "page_idx": c.get("page_idx"), 
                "type": "text"
            }
        })
    return results
