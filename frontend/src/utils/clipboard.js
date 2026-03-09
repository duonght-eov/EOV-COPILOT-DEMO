/**
 * Copies text to the clipboard with a fallback for environments where
 * navigate.clipboard is not available or fails (e.g. non-secure contexts).
 *
 * @param {string} text - The text to copy
 * @returns {Promise<boolean>} - True if successful, false otherwise
 */
export async function copyToClipboard(text) {
    if (!text) return false;

    // Try the modern API first
    try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
            return true;
        }
    } catch (error) {
        console.warn("navigator.clipboard.writeText failed, trying fallback.", error);
    }

    // Fallback method using execCommand
    try {
        const textArea = document.createElement("textarea");
        textArea.value = text;

        // Ensure the textarea is not visible but part of the DOM
        textArea.style.position = "fixed";
        textArea.style.left = "-9999px";
        textArea.style.top = "0";
        document.body.appendChild(textArea);

        textArea.focus();
        textArea.select();

        const successful = document.execCommand("copy");
        document.body.removeChild(textArea);
        return successful;
    } catch (fallbackError) {
        console.error("Fallback copy to clipboard failed.", fallbackError);
        return false;
    }
}
