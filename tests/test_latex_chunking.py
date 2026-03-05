import sys
import os
import json

# Add the project root to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.text_chunker import StyleDFSChunker

def test_latex_chunking():
    # 1. Prepare Dummy Data with LaTeX
    dummy_input = {
        "content": [
            {
                "type": "heading",
                "text": "Toán học cơ bản",
                "level": 1,
                "page_idx": 1
            },
            {
                "type": "text",
                "text": "Đây là một phương trình bậc hai đơn giản:",
                "page_idx": 1
            },
            {
                "type": "text",
                "text": "$$ax^2 + bx + c = 0$$", 
                "page_idx": 1
            },
            {
                "type": "text",
                "text": "Nghiệm của phương trình được tính bằng công thức:",
                "page_idx": 1
            },
            {
                "type": "text",
                "text": r"$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$",
                "page_idx": 1
            },
            {
                "type": "text",
                "text": r"Trong đó $\Delta = b^2 - 4ac$ được gọi là biệt thức.",
                "page_idx": 1
            }
        ]
    }

    print("--- Input Data ---")
    print(json.dumps(dummy_input, indent=2, ensure_ascii=False))
    print("\n--- Running Chunker ---")

    # 2. Run Chunker
    chunker = StyleDFSChunker()
    chunks = chunker.process(dummy_input, doc_id="test_doc_latex")

    # 3. Analyze Results
    print(f"\nTotal Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n>>> Chunk {i+1}:")
        print(f"Metadata: {chunk['metadata']}")
        print("-" * 20)
        print(chunk['content'])
        print("-" * 20)
        
        # Validation Logic
        content = chunk['content']
        if "[EQUATION]" in content:
            print("✅ SUCCESS: LaTeX equation detected and wrapped in [EQUATION] tag.")
        else:
            # Note: The last item contains inline latex mixed with text. 
            # The regex searches ANYWHERE in the text. 
            # If the block is purely text containing latex, it should convert to equation type.
            # Let's see if the mixed text block was caught.
            pass

if __name__ == "__main__":
    test_latex_chunking()
