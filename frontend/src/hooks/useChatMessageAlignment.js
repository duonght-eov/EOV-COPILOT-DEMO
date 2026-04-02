import { useState, useEffect, useCallback } from "react";
const ALIGNMENT_STORAGE_KEY = "anythingllm-chat-message-alignment";

/**
 * Store the message alignment in localStorage as well as provide a function to get the alignment of a message via role.
 * @returns {{msgDirection: 'left'|'left_right', setMsgDirection: (direction: string) => void, getMessageAlignment: (role: string) => string}} - The message direction and the class name for the direction.
 */
export function useChatMessageAlignment() {
  const [msgDirection, setMsgDirection] = useState(() => {
    // Debug: Log khi đọc từ localStorage
    const saved = localStorage.getItem(ALIGNMENT_STORAGE_KEY);
    console.log("[useChatMessageAlignment] Loading from localStorage:", saved);
    return saved ?? "left";
  });

  useEffect(() => {
    // Debug: Log khi lưu vào localStorage
    console.log("[useChatMessageAlignment] Saving to localStorage:", msgDirection);
    if (msgDirection) {
      try {
        localStorage.setItem(ALIGNMENT_STORAGE_KEY, msgDirection);
        // Verify: Đọc lại để xác nhận
        const verified = localStorage.getItem(ALIGNMENT_STORAGE_KEY);
        console.log("[useChatMessageAlignment] Verified saved value:", verified);
      } catch (error) {
        console.error("[useChatMessageAlignment] Failed to save to localStorage:", error);
      }
    }
  }, [msgDirection]);

  const getMessageAlignment = useCallback(
    (role) => {
      const isLeftToRight = role === "user" && msgDirection === "left_right";
      return isLeftToRight ? "flex-row-reverse" : "";
    },
    [msgDirection]
  );

  return {
    msgDirection,
    setMsgDirection,
    getMessageAlignment,
  };
}
