# Cơ Chế Tìm Kiếm Trong LightRAG (Phân Tích Source Code)

Tài liệu này phân tích chi tiết các phương thức tìm kiếm (Retrieval Mode) trong thư viện LightRAG, dựa trên mã nguồn thực tế tại `libs/LightRAG/lightrag/operate.py` và `libs/LightRAG/lightrag/lightrag.py`.

## 1. Tổng Quan

LightRAG hỗ trợ 5 chế độ tìm kiếm chính, được điều phối bởi Class `LightRAG` và các hàm trong `operate.py`:
1.  **Naive Search**: Tìm kiếm vector truyền thống.
2.  **Local Search**: Tìm kiếm dựa trên Thực thể (Entities).
3.  **Global Search**: Tìm kiếm dựa trên Mối quan hệ (Relationships).
4.  **Hybrid Search**: Kết hợp Local + Global.
5.  **Mix Search**: Kết hợp Hybrid (Graph) + Naive (Vector).

Entry point chính cho mọi truy vấn là hàm `aquery` (hoặc `aquery_llm`) trong `lightrag.py`, sau đó gọi đến `naive_query` hoặc `kg_query` trong `operate.py`.

---

## 2. Chi Tiết Các Chế Độ Tìm Kiếm

### 2.1. Naive Search (Vector Search)

Đây là phương pháp RAG cơ bản nhất, không sử dụng Knowledge Graph (KG).

*   **Entry Point**: `operate.py` -> `naive_query`.
*   **Quy trình**:
    1.  **Vector Retrieval**: Gọi hàm `_get_vector_context`.
        *   Sử dụng `chunks_vdb` (Vector Database chứa các đoạn văn bản).
        *   Thực hiện truy vấn vector (`chunks_vdb.query`) với câu hỏi của người dùng.
        *   Trả về top K đoạn văn bản (Chunks) có vector tương đồng nhất.
    2.  **Token Truncation**: Cắt ngắn nội dung nếu vượt quá giới hạn token.
    3.  **Prompt Building**: Ghép các chunks vào prompt (`naive_rag_response`).
    4.  **Generation**: Gửi prompt cho LLM để sinh câu trả lời.
*   **Ưu điểm**: Nhanh, đơn giản, hiệu quả với các câu hỏi trực tiếp có câu trả lời nằm gọn trong một đoạn văn.
*   **Nhược điểm**: Mất ngữ cảnh bao quát, khó trả lời các câu hỏi tổng hợp hoặc đa bước.

### 2.2. Local Search (Entity-Centric Retrieval)

Phương pháp này tập trung vào các chi tiết cụ thể bằng cách tìm kiếm các **Thực thể (Entities)** liên quan.

*   **Entry Point**: `operate.py` -> `kg_query` (với `query_param.mode="local"`).
*   **Quy trình**:
    1.  **Keyword Extraction**: Gọi `get_keywords_from_query` để trích xuất **Low-level Keywords** (bằng LLM). Đây là các danh từ riêng, tên gọi cụ thể.
    2.  **Entity Lookup**: Gọi `_perform_kg_search` -> `_get_node_data`.
        *   Tìm kiếm trong `entities_vdb` (Vector DB chứa tên/mô tả thực thể).
        *   Lấy ra một danh sách các Entity Nodes phù hợp nhất.
    3.  **Graph Expansion (Reverse Indexing)**: Gọi `_find_related_text_unit_from_entities`.
        *   Từ mỗi Entity Node, truy xuất trường metadata `source_id` (được lưu trong Graph Storage).
        *   `source_id` chứa danh sách các `chunk_id` mà thực thể đó xuất hiện.
        *   Hệ thống thu thập tất cả các Text Chunks liên quan đến các Entities tìm được.
    4.  **Edge Retrieval (Optional)**: Tìm thêm các cạnh (Edges) nối giữa các Entities này để bổ sung ngữ cảnh quan hệ.
    5.  **Context Construction**: Tổng hợp Entities Description + Relations + Text Chunks thành context.
*   **Đặc điểm**: "Đi đường vòng" qua Graph để tìm chính xác đoạn văn chứa khái niệm, thay vì đoán mò bằng vector của cả câu. Rất mạnh cho các câu hỏi "Who", "What", "Specific details".

### 2.3. Global Search (Relation-Centric Retrieval)

Phương pháp này tập trung vào bức tranh toàn cảnh bằng cách tìm kiếm các **Mối quan hệ (Relationships)** bao quát.

*   **Entry Point**: `operate.py` -> `kg_query` (với `query_param.mode="global"`).
*   **Quy trình**:
    1.  **Keyword Extraction**: Trích xuất **High-level Keywords** (các chủ đề, khái niệm trừu tượng).
    2.  **Relationship Lookup**: Gọi `_perform_kg_search` -> `_get_edge_data`.
        *   Tìm kiếm trong `relationships_vdb` (Vector DB chứa mô tả của các mối quan hệ/cạnh).
        *   Lấy ra các Edges phù hợp nhất.
    3.  **Node Retrieval**: Từ các Edges tìm được (`src_id`, `tgt_id`), lấy thông tin các Nodes tương ứng.
    4.  **Context Construction**: Chủ yếu sử dụng mô tả của Relationships (Edge Description) làm context chính. Edge Description trong LightRAG thường là tóm tắt quan hệ giữa 2 thực thể.
*   **Đặc điểm**: Trả lời tốt các câu hỏi "How", "Why", "Summary", "Themes" bằng cách tổng hợp thông tin từ nhiều nơi thông qua các mối liên kết.

### 2.4. Hybrid Search

Kết hợp Local và Global để tận dụng ưu điểm của cả hai.

*   **Quy trình**:
    1.  Chạy song song **Local Search** (tìm Entities từ Low-level Keywords) và **Global Search** (tìm Relations từ High-level Keywords).
    2.  **Merge Results**: Hàm `_perform_kg_search` thực hiện hợp nhất kết quả (Entities và Relations) theo cơ chế **Round-Robin** (lấy xen kẽ 1 từ Local, 1 từ Global,...) để đảm bảo sự cân bằng.

### 2.5. Mix Search (Knowledge Graph + Vector)

Chế độ toàn diện nhất, kết hợp sức mạnh của Knowledge Graph (Hybrid) và Vector Search (Naive).

*   **Entry Point**: `operate.py` -> `kg_query` (với `query_param.mode="mix"`).
*   **Quy trình**:
    1.  Thực hiện **Hybrid Search** (như trên) để lấy Entities và Relations từ Graph.
    2.  Thực hiện **Vector Search** (như Naive) qua hàm `_get_vector_context` để lấy thêm các Text Chunks dựa trên semantic similarity thuần túy.
    3.  **Merge All**: Hợp nhất tất cả Text Chunks từ 2 nguồn (Graph-based chunks và Vector-based chunks).
    4.  **Context Dedup**: Loại bỏ trùng lặp và sắp xếp lại theo độ ưu tiên.
*   **Đặc điểm**: Độ phủ cao nhất (High Recall). Đảm bảo không bỏ sót thông tin dù nó nằm trong Graph hay chỉ đơn thuần tương đồng về ngữ nghĩa.

---

## 3. Bảng So Sánh Tóm Tắt

| Mode | Input Keywords | Primary Index | Retrieval Logic | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Naive** | Raw Query | `chunks_vdb` | Chunk Vector Similarity | Direct Q&A, Factoid |
| **Local** | Low-level Keys | `entities_vdb` | Keywords -> Entities -> **Chunks** | Specific Info, Details |
| **Global**| High-level Keys| `relationships_vdb`| Keywords -> **Relations** -> Entities | Summarization, Themes |
| **Hybrid**| Low + High Keys| Both | Merge(Local, Global) | Complex Q&A |
| **Mix** | All | All | Merge(Graph, Vector) | Comprehensive Search |

---

*Tài liệu này được tạo tự động dựa trên phân tích source code ngày 12/02/2026.*
