from pathlib import Path
from langchain_core.prompts import PromptTemplate

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_react_prompt(name: str = "react_water_vi") -> PromptTemplate:
    """
    Tải ReAct prompt template từ file text trong thư mục /prompts/.
    Cho phép chỉnh sửa prompt mà không cần sửa code Python.
    """
    prompt_file = _PROMPTS_DIR / f"{name}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Không tìm thấy prompt file: {prompt_file}")

    template = prompt_file.read_text(encoding="utf-8")

    return PromptTemplate.from_template(template)
