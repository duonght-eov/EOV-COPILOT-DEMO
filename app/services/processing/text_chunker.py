import json
from typing import List, Dict, Any
import tiktoken
import re
from app.utils.logger import get_logger

logger = get_logger("TEXT_CHUNKER")

class Tokenizer:
    def __init__(self, model_name="cl100k_base"):
        self.encoder = tiktoken.get_encoding(model_name)
    def count(self, text: str) -> int:
        return len(self.encoder.encode(text))

tokenizer = Tokenizer()

class DocumentNode:
    def __init__(self, text: str, level: int, node_type: str, metadata: Dict = None):
        self.text = text
        self.level = level
        self.node_type = node_type # 'heading', 'text', 'image', 'table', 'root'
        self.metadata = metadata or {}
        self.children: List['DocumentNode'] = []
        self.token_count = 0 

    def add_child(self, node: 'DocumentNode'):
        self.children.append(node)

class StyleDFSChunker:
    """
    Style-based Depth-First Search Chunker.
    Respects document structure (Headings) and semantically groups content.
    Strictly follows token limits and binds context.
    """
    def __init__(self, target_chunk_size=1024, chunk_overlap=0): 
        self.target_size = target_chunk_size
        self.overlap = chunk_overlap

    def process(self, text_content: Any, doc_id: str, file_path: str = None) -> List[Dict]:
        try:
            # 1. Normalize Input to Blocks
            blocks = self._normalize_input(text_content)
            if not blocks: return []

            # 2. Build Tree (Heading Hierarchy)
            root = self._build_tree(blocks)
            
            # 3. Traverse and Chunk
            chunks = []
            accum = [] # List[DocumentNode]
            context_stack = [] # List[str] - Heading texts
            
            self._dfs_traverse(root, context_stack, accum, chunks)
            
            # Final flush
            if accum:
                self._finalize_chunk(accum, chunks, context_stack)
            
            logger.info(f"Chunker produced {len(chunks)} chunks for doc {doc_id}")
            return chunks
        except Exception as e:
            logger.error(f"Chunker Error processing doc {doc_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _normalize_input(self, text_content: Any) -> List[Dict]:
        data = text_content
        if isinstance(text_content, str):
            try:
                data = json.loads(text_content)
            except:
                data = {"content": [{"type": "text", "text": text_content}]}

        blocks = []
        # Support various JSON structures
        if isinstance(data, dict):
            if "document" in data and "content" in data["document"]:
                # Mineru/Standard Format
                for page in data["document"]["content"]:
                    page_num = page.get("page_number", 0)
                    page_blocks = page.get("blocks", [])
                    for b in page_blocks:
                        b["page_idx"] = b.get("page_idx", page_num)
                    blocks.extend(page_blocks)
            elif "content" in data and isinstance(data["content"], list):
                 # Flattened format
                 blocks = data["content"]
        elif isinstance(data, list):
             blocks = data
        
        # --- LaTeX Post-Processing ---
        # Detect latent equations in text blocks and mark them essentially atomic
        latex_pattern = re.compile(r"(\$\$[\s\S]*?\$\$)|(\$[^\n$]*\$)|(\\\[[\s\S]*?\\\])|(\\\([\s\S]*?\\\))")
        
        for block in blocks:
            if block.get("type") == "text":
                text = block.get("text", "")
                if latex_pattern.search(text):
                    # Found LaTeX -> Mark as equation to treat atomically
                    block["type"] = "equation"
        
        return blocks

    def _build_tree(self, blocks: List[Dict]) -> DocumentNode:
        root = DocumentNode("ROOT", 0, "root")
        # Stack contains chain of active headings: [Root, H1, H2...]
        stack = [root]

        for block in blocks:
            b_type = block.get("type", "text")
            b_text = block.get("text", "") or ""
            
            # Map specific types to generic categories
            if b_type == "paragraph": b_type = "text"
            if b_type == "caption": b_type = "text" # Treat captions as text flow
            
            # Extract Metadata
            p_idx = block.get("page_idx", block.get("page_number", 0))
            metadata = block.copy() 
            metadata.pop("text", None)
            metadata["page_idx"] = p_idx
            
            # Heading Logic strategy:
            # If Heading -> Find parent in stack -> Add -> Push to stack
            if b_type == "heading":
                level = block.get("level", 1)
                node = DocumentNode(b_text, level, "heading", metadata)
                node.token_count = tokenizer.count(b_text)
                
                # Pop stack until we find a parent with level < node.level
                # Root is level 0. H1 is level 1.
                while len(stack) > 1 and stack[-1].level >= level:
                    stack.pop()
                
                parent = stack[-1]
                parent.add_child(node)
                stack.append(node)
            else:
                # Content Node (Text/Image/Table)
                # Adds to current active heading
                node = DocumentNode(b_text, 999, b_type, metadata)
                if b_type == "image":
                    # For images, calculate tokens based on caption or placeholder
                    caption_text = str(metadata.get("image_caption") or "") 
                    if isinstance(caption_text, list): caption_text = " ".join(caption_text)
                    node.token_count = tokenizer.count(b_text + caption_text) + 100 # +100 base cost for image presence
                elif b_type == "table":
                    # Keep full table body text count
                    node.token_count = tokenizer.count(b_text) + 50
                elif b_type == "equation":
                    # Equations are dense information. Even short ones matter.
                    node.token_count = tokenizer.count(b_text) + 20
                else:
                    node.token_count = tokenizer.count(b_text)
                
                stack[-1].add_child(node)
        
        return root

    def _dfs_traverse(self, node: DocumentNode, context_stack: List[str], accum: List[DocumentNode], chunks: List[Dict]):
        # 1. ENTER NODE
        # If this is a Heading node (and not Root), it signifies a section break.
        if node.node_type == "heading":
            # If we have loose content in 'accum', it belongs to the PREVIOUS section. Finalize it.
            if accum:
                self._finalize_chunk(accum, chunks, context_stack)
                accum.clear()
            
            # Add this heading to context
            context_stack.append(node.text)

        # 2. PROCESS CONTENT for this node
        # In this tree structure, 'node' is the container (Heading).
        # Its children are Sub-Headings OR Content Nodes (Text/Image).
        
        for child in node.children:
            if child.node_type == "heading":
                # Recurse for sub-heading
                self._dfs_traverse(child, context_stack, accum, chunks)
            else:
                # Leaf content (Text, Image, Table)
                self._add_content_to_chunk(child, accum, chunks, context_stack)

        # 3. EXIT NODE
        if node.node_type == "heading":
            # Before leaving this section, finalize any content specific to this section.
            # This ensures content doesn't bleed into the parent's next sibling.
            if accum:
                self._finalize_chunk(accum, chunks, context_stack)
                accum.clear()
            
            # Pop context
            context_stack.pop()

    def _add_content_to_chunk(self, node: DocumentNode, accum: List[DocumentNode], chunks: List[Dict], context_stack: List[str]):
        # PRE-CHECK Size Logic
        current_tokens = sum(n.token_count for n in accum)
        
        if current_tokens + node.token_count > self.target_size:
            # OVERFLOW DETECTED
            
            # Lead-in Strategy: 
            # If current node is Image/Table, and previous node was a short "lead-in" text?
            lead_in_node = None
            if node.node_type in ["image", "table"] and accum:
                last = accum[-1]
                # Look for short text (< 60 tokens) to carry over
                if last.node_type == "text" and last.token_count < 60: 
                    # Verify if moving it to new chunk fits (it should, new chunk is empty)
                    lead_in_node = accum.pop()
            
            # Finalize current accum
            self._finalize_chunk(accum, chunks, context_stack)
            accum.clear()
            
            # Start new chunk
            if lead_in_node:
                accum.append(lead_in_node)
            accum.append(node)
        else:
            # Fits in chunk
            accum.append(node)

    def _finalize_chunk(self, accum: List[DocumentNode], chunks: List[Dict], context_stack: List[str]):
        if not accum: return
        
        content_parts = []
        page_indices = set()
        
        # Add Context Header
        # Exclude ROOT from context path usually
        clean_context = [c for c in context_stack if c != "ROOT"]
        if clean_context:
            header = "Context: " + " > ".join(clean_context)
            content_parts.append(header)
            content_parts.append("-" * 20)

        # Add Body
        for node in accum:
            page_indices.add(node.metadata.get("page_idx", 0))
            
            if node.node_type == "text":
                content_parts.append(node.text)
            elif node.node_type == "image":
                path = node.metadata.get("img_path", "")
                caption = node.metadata.get("image_caption", "") or ""
                if isinstance(caption, list): caption = " ".join(caption)
                
                # Format clearly for RAG
                content_parts.append(f"\n[IMAGE_REF: {path}]\nDescription: {caption}")
                
            elif node.node_type == "table":
                caption = node.metadata.get("table_caption", "") or ""
                if isinstance(caption, list): caption = " ".join(caption)
                content_parts.append(f"\n[TABLE_DATA]\n{node.text}\nCaption: {caption}")
            
            elif node.node_type == "equation":
                # Ensure it stands out
                content_parts.append(f"\n[EQUATION]\n{node.text}\n")

        full_content = "\n\n".join(content_parts)
        
        # Metadata
        min_p = min(page_indices) if page_indices else 0
        max_p = max(page_indices) if page_indices else 0
        page_label = f"[Page {min_p}]" if min_p == max_p else f"[Page {min_p}-{max_p}]"
        
        full_content = f"{page_label}\n{full_content}"
        
        # Generate Metadata for LightRAG/DB
        chunk_meta = {
            "page_idx": min_p, 
            "page_start": min_p,
            "page_end": max_p,
            "context_headings": list(clean_context),
            # Important: extract image path if chunk is primarily image
            "type": "mixed" 
        }
        
        # If meaningful image exists, promote to metadata for retrieval UI
        img_paths = [n.metadata.get("img_path") for n in accum if n.node_type == "image" and n.metadata.get("img_path")]
        if img_paths:
            chunk_meta["img_path"] = img_paths[0] # Take first image as representative
            chunk_meta["has_image"] = True

        chunks.append({
            "content": full_content,
            "tokens": tokenizer.count(full_content),
            "metadata": chunk_meta,
            "source_id": f"chunk-{len(chunks)}", # Temporary ID
        })

CustomChunker = StyleDFSChunker