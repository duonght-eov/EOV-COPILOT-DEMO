from langchain.memory import ConversationBufferWindowMemory

_memory_store: dict[str, ConversationBufferWindowMemory] = {}


def get_memory(session_id: str, k: int = 5) -> ConversationBufferWindowMemory:
    """
    Trả về bộ nhớ hội thoại cho session_id.
    Mỗi session_id (vd: "{user_id}_{workspace_id}") có bộ nhớ riêng biệt.
    k: số lượt hội thoại giữ lại.
    """
    if session_id not in _memory_store:
        _memory_store[session_id] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=k,
            return_messages=True,
        )
    return _memory_store[session_id]


def clear_memory(session_id: str) -> None:
    """Xóa bộ nhớ của một session (dùng khi bắt đầu cuộc chat mới)."""
    _memory_store.pop(session_id, None)
