"""
services/processing/image_resolver.py
Trách nhiệm:
  - extract_image_refs_from_answer logic (Tầng 1 & Tầng 2)
  - Phân tích Text để tìm tag hình ảnh.
"""
import re
from typing import List, Dict
from app.utils.logger import get_logger

logger = get_logger("IMAGE_RESOLVER")

IMAGE_REF_PATTERN = re.compile(r'\[IMAGE_REF:\s*([^\]]+)\]')
PAGE_CITE_PATTERN = re.compile(r'\[Page\s+(\d+)\]', re.IGNORECASE)
_IMG_NGRAM_SIZE = 10 # Giảm ngưỡng để hiển thị ảnh dễ hơn sau khi đã lọc nhiễu VLM
_VISUAL_KEYWORDS = re.compile(
    r'(hình\s*ảnh|sơ\s*đồ|biểu\s*đồ|hình\s*vẽ|ảnh\s*minh\s*họa|minh\s*họa|hình\s*dưới|bảng\s*sau)',
    re.IGNORECASE
)

def _parse_obj_key(raw_path: str) -> str:
    if raw_path.startswith('ocr-results/'):
        return raw_path[len('ocr-results/'):]
    return raw_path

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[\[\]\(\)\{\}"\',.:;!?\-_]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def _extract_img_description(content: str, img_ref_match: re.Match) -> str:
    start = img_ref_match.end()
    desc_raw = content[start:start + 1000].strip()
    stop = re.search(r'\[IMAGE_REF:|\[Page\s+\d+\]', desc_raw)
    if stop:
        desc_raw = desc_raw[:stop.start()].strip()
    return desc_raw

def _image_desc_used_in_answer(description: str, answer: str, ngram_size: int = _IMG_NGRAM_SIZE) -> bool:
    if not description or not answer:
        return False
    norm_desc = _normalize(description)
    norm_ans = _normalize(answer)
    words_desc = norm_desc.split()
    if len(words_desc) < ngram_size:
        return norm_desc in norm_ans

    for i in range(len(words_desc) - ngram_size + 1):
        phrase = ' '.join(words_desc[i:i + ngram_size])
        if phrase in norm_ans:
            return True
    return False

def _answer_visually_references_page(answer: str, page_num: int) -> bool:
    """
    Kiểm tra xem câu trả lời có thực sự nhắc đến hình ảnh ở trang này không.
    Phải có từ khóa thị giác nằm GẦN trích dẫn trang (trong khoảng 100 ký tự).
    """
    norm_ans = answer.lower()
    page_tag = f"[page {page_num}]"
    
    start_search = 0
    while True:
        pos = norm_ans.find(page_tag, start_search)
        if pos == -1:
            break
            
        # Kiểm tra ngữ cảnh xung quanh tag [Page X] (trước 80 ký tự, sau 20 ký tự)
        context_start = max(0, pos - 80)
        context_end = min(len(norm_ans), pos + 25)
        context = norm_ans[context_start:context_end]
        
        if _VISUAL_KEYWORDS.search(context):
            return True
        start_search = pos + 1
        
    return False


def _find_chunk_citation_for_image(content: str, img_start_pos: int) -> str | None:
    """Tìm số citation [n] gần nhất phía trước ảnh để biết ảnh thuộc chunk nào."""
    pre_text = content[:img_start_pos]
    citations = list(re.finditer(r'\[\s*(\d+)\s*\]', pre_text))
    if citations:
        return f"[{citations[-1].group(1)}]"
    return None

def _is_image_visually_relevant(content: str, img_start_pos: int) -> bool:
    """Soi các ký tự xung quanh bức ảnh (trước/sau 200 character) xem có chữ sơ đồ/hình ảnh không."""
    context_start = max(0, img_start_pos - 200)
    context_end = min(len(content), img_start_pos + 200)
    surrounding_text = content[context_start:context_end]
    return bool(_VISUAL_KEYWORDS.search(surrounding_text))

def extract_image_refs_from_answer(chunks: List[Dict], answer: str, context_text: str = "") -> List[str]:
    seen_basenames = set()
    refs = []
    
    # Chỉ duyệt qua context_text vì context_text đã gộp hết các chunks ở query_pipeline rồi
    # Nếu truyền cả chunks và context_text -> trùng lặp thông tin
    sources = [context_text] if context_text else [(chunk.get('content', '') or '') for chunk in chunks]

    for content in sources:
        for img_match in IMAGE_REF_PATTERN.finditer(content):
            obj_key = _parse_obj_key(img_match.group(1).strip())
            if not obj_key:
                continue
                
            basename = obj_key.split('/')[-1]
            if basename in seen_basenames:
                continue

            description = _extract_img_description(content, img_match)

            # Lấy citation [n] của đoạn văn chứa bức ảnh này
            chunk_citation = _find_chunk_citation_for_image(content, img_match.start())
            is_chunk_cited = (chunk_citation and chunk_citation in answer)
            is_visually_relevant = _is_image_visually_relevant(content, img_match.start())

            # Bypass: Nếu ảnh mù dở (Lỗi VLM), ta cần check gắt gao hơn để chống ảnh hiển thị rác.
            # ĐIỀU KIỆN 1: Chunk chứa ảnh này phải được LLM trích dẫn (cite).
            # ĐIỀU KIỆN 2: Văn bản xung quanh bức ảnh (trong tài liệu) phải có nhắc đến các từ khoá thị giác (ví dụ: "sơ đồ dưới đây", "như hình ảnh 1"). Đoạn này giúp lọc sạch 100% logo và viền trang.
            desc_lower = description.lower()
            if "vlm service không khả dụng" in desc_lower or "vlm timeout" in desc_lower or "không thể tạo mô tả cho hình ảnh này" in desc_lower:
                if is_chunk_cited and is_visually_relevant:
                    seen_basenames.add(basename)
                    refs.append(obj_key)
                    logger.info(f"[ImageFilter] [T3-vlm-fail-bypass] {basename} (cited {chunk_citation}, context verified)")
                continue

            if _image_desc_used_in_answer(description, answer):
                seen_basenames.add(basename)
                refs.append(obj_key)
                logger.info(f"[ImageFilter] [T1-ngram] {basename}")
                continue

            pre_text = content[max(0, img_match.start() - 30):img_match.start()]
            page_m = PAGE_CITE_PATTERN.search(pre_text)
            if page_m:
                img_page = int(page_m.group(1))
                if _answer_visually_references_page(answer, img_page):
                    seen_basenames.add(basename)
                    refs.append(obj_key)
                    logger.info(f"[ImageFilter] [T2-visual] {basename} (page {img_page})")
                    continue

            logger.debug(f"[ImageFilter] Bỏ qua: {basename}")

    return refs

