#!/bin/bash
# Script khởi động RAG Service với pre-load Ollama model

set -e

echo "=== RAG Service Startup Script ==="

# 1. Kiểm tra Ollama có chạy không
echo "[1/4] Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama not running! Please start Ollama first:"
    echo "   export OLLAMA_MODELS=\"/mnt/hdd2tb/ollama_models\""
    echo "   export OLLAMA_KEEP_ALIVE=\"-1\""
    echo "   ollama serve"
    exit 1
fi
echo "✅ Ollama is running"

# 2. Pre-load model vào GPU
echo "[2/4] Pre-loading qwen3:8b-q8_0 model into GPU..."
echo "    (This may take 30-60 seconds for first load)"
curl -s -X POST http://localhost:11434/api/generate \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3:8b-q8_0","prompt":"hi","stream":false}' > /dev/null 2>&1
echo "✅ Model loaded"

# 3. Kiểm tra model đã load chưa
echo "[3/4] Verifying model is loaded..."
MODELS=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' | grep qwen3:8b-q8_0)
if [ -n "$MODELS" ]; then
    echo "✅ Model qwen3:8b-q8_0 is available"
else
    echo "⚠️  Model not found in Ollama"
fi

# 4. Start RAG Service
echo "[4/4] Starting RAG Service..."
cd /home/datpt/projects/EOVCopilot-Demo/services/rag-service

# Activate environment nếu cần
# source /path/to/venv/bin/activate

python -m uvicorn app.main:app --host 0.0.0.0 --port 8006 --reload
