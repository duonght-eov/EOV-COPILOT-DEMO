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
_IMG_NGRAM_SIZE = 10
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
    norm_ans = answer.lower()
    if not _VISUAL_KEYWORDS.search(norm_ans):
        return False
    cited = {int(m.group(1)) for m in PAGE_CITE_PATTERN.finditer(answer)}
    return page_num in cited

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

