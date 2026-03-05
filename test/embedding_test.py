import requests
import json
import time
import re
import os
from typing import List

# --- CẤU HÌNH ---
# Đường dẫn file dữ liệu thật của bạn
DATA_FILE_PATH = "/home/datpt/projects/EOVCopilot-Demo/services/embedding-service/test/workflow_result.txt"
# Endpoint của Embedding Service
API_URL = "http://localhost:8003/api/v1/embed/batch"
# Kích thước Batch (nên khớp với cấu hình server để tối ưu)
BATCH_SIZE = 32

class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def parse_workflow_file(file_path: str) -> List[str]:
    """
    Hàm đọc file workflow_result.txt và trích xuất nội dung text của từng chunk.
    Loại bỏ các dòng META và phân cách.
    """
    if not os.path.exists(file_path):
        print(f"{TerminalColors.FAIL}❌ Không tìm thấy file tại: {file_path}{TerminalColors.ENDC}")
        return []

    print(f"{TerminalColors.OKBLUE}📂 Đang đọc file: {file_path}...{TerminalColors.ENDC}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Tách các chunk dựa trên đường kẻ phân cách trong file của bạn
    # Dựa vào mẫu: "------------------------------------------------------------"
    raw_blocks = content.split("-" * 60)
    
    valid_chunks = []
    
    for block in raw_blocks:
        block = block.strip()
        if not block: 
            continue
            
        # Dùng Regex để tìm dòng META và lấy phần nội dung phía sau nó
        # Pattern: Tìm dòng bắt đầu bằng META: {...} và lấy toàn bộ phần text sau đó
        match = re.search(r'META: \{.*?\}\n(.*)', block, re.DOTALL)
        
        if match:
            text_content = match.group(1).strip()
            # Loại bỏ các dòng tiêu đề Chunk nếu còn sót (VD: --- CHUNK #1 ---)
            text_content = re.sub(r'--- CHUNK #\d+ ---', '', text_content).strip()
            
            if len(text_content) > 0:
                valid_chunks.append(text_content)
    
    print(f"{TerminalColors.OKGREEN}✅ Đã trích xuất thành công: {len(valid_chunks)} chunks.{TerminalColors.ENDC}")
    return valid_chunks

def test_embedding_performance(chunks: List[str]):
    """
    Gửi request lên API và đo hiệu năng.
    """
    total_chunks = len(chunks)
    total_time = 0
    total_vectors = 0
    
    print(f"\n{TerminalColors.HEADER}🚀 BẮT ĐẦU TEST HIỆU NĂNG (Batch Size: {BATCH_SIZE}){TerminalColors.ENDC}")
    print("-" * 60)

    # Chia nhỏ thành các batch để gửi
    for i in range(0, total_chunks, BATCH_SIZE):
        batch_texts = chunks[i : i + BATCH_SIZE]
        current_batch_size = len(batch_texts)
        
        payload = {
            "texts": batch_texts
        }
        
        print(f"📡 Đang gửi Batch {i//BATCH_SIZE + 1} ({current_batch_size} chunks)...", end=" ")
        
        start_time = time.time()
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            end_time = time.time()
            duration = end_time - start_time
            total_time += duration
            
            if response.status_code == 200:
                data = response.json()
                vectors = data.get("vectors", [])
                total_vectors += len(vectors)
                
                # Validate nhanh kích thước vector đầu tiên
                dim = len(vectors[0])
                model_name = data.get("model", "unknown")
                
                print(f"{TerminalColors.OKGREEN}OK ({duration:.2f}s){TerminalColors.ENDC}")
                print(f"   ↳ Model: {model_name} | Dims: {dim} | Cached: Check logs")
                
                if dim != 1024:
                     print(f"{TerminalColors.FAIL}   ⚠️ CẢNH BÁO: Số chiều vector là {dim} (Kỳ vọng 1024){TerminalColors.ENDC}")

            else:
                print(f"{TerminalColors.FAIL}FAILED (Status: {response.status_code}){TerminalColors.ENDC}")
                print(response.text)
                
        except Exception as e:
            print(f"{TerminalColors.FAIL}ERROR: {str(e)}{TerminalColors.ENDC}")

    # --- TỔNG KẾT ---
    print("-" * 60)
    print(f"{TerminalColors.HEADER}📊 KẾT QUẢ KIỂM THỬ:{TerminalColors.ENDC}")
    if total_time > 0:
        avg_time_per_chunk = total_time / total_chunks
        tps = total_chunks / total_time
        print(f"   • Tổng số Chunks:      {total_chunks}")
        print(f"   • Tổng số Vectors:     {total_vectors}")
        print(f"   • Tổng thời gian:      {total_time:.2f}s")
        print(f"   • Trung bình/Chunk:    {avg_time_per_chunk*1000:.2f}ms")
        print(f"   • Tốc độ xử lý (TPS):  {TerminalColors.OKBLUE}{tps:.2f} chunks/giây{TerminalColors.ENDC}")
    else:
        print("   Không có dữ liệu thời gian.")

if __name__ == "__main__":
    # 1. Parse dữ liệu
    chunks = parse_workflow_file(DATA_FILE_PATH)
    
    # 2. Chạy test nếu có dữ liệu
    if chunks:
        # In thử nội dung chunk đầu tiên để kiểm tra parser
        print(f"\n📝 [Preview Chunk #0]: {chunks[0][:100]}...")
        test_embedding_performance(chunks)