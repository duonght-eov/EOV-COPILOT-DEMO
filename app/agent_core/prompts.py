from pathlib import Path
from langchain_core.prompts import PromptTemplate
from app.utils.logger import get_logger

logger = get_logger("Prompts")
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_react_prompt(name: str = "react_water_vi") -> PromptTemplate:
    """
    Tải ReAct prompt template từ file text trong thư mục /prompts/.
    Cho phép chỉnh sửa prompt mà không cần sửa code Python.
    """
    prompt_file = _PROMPTS_DIR / f"{name}.txt"
    logger.debug(f"Loading prompt template: {prompt_file}")
    if not prompt_file.exists():
        logger.error(f"Prompt file not found: {prompt_file}")
        raise FileNotFoundError(f"Không tìm thấy prompt file: {prompt_file}")

    template = prompt_file.read_text(encoding="utf-8")
    logger.debug(f"Loaded prompt template '{name}' ({len(template)} chars)")

    return PromptTemplate.from_template(template)
