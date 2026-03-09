import { useState } from "react";
import { copyToClipboard } from "@/utils/clipboard";

export default function useCopyText(delay = 2500) {
  const [copied, setCopied] = useState(false);
  const copyText = async (content) => {
    if (!content) return;
    const success = await copyToClipboard(content);
    if (success) {
      setCopied(true);
      setTimeout(() => {
        setCopied(false);
      }, delay);
    } else {
      setCopied(false);
    }
  };

  return { copyText, copied };
}
