# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder — cài pip deps vào /install để copy sang image cuối
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Chỉ cài những gì pip build cần (gcc cho một số C-extension)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Cài vào thư mục riêng để copy sang runtime image
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime — image nhẹ, không có build tools
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/usr/local/lib/python3.11/site-packages \
    DEBIAN_FRONTEND=noninteractive

# Chỉ cài runtime system deps cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy toàn bộ Python packages từ builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy source code
COPY app ./app
COPY prompts ./prompts

# Workspace dir cho RAG (file JSON index khi dùng json storage)
RUN mkdir -p /app/rag_workspace && chmod 777 /app/rag_workspace

EXPOSE 8006

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8006", "--workers", "1"]
