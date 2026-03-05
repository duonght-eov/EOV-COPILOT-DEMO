
# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# Added 'libreoffice' for RAG-Anything office document support
# Added 'libmagic1' often needed for file type detection
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    libreoffice \
    libmagic1 \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for cache optimization
# Assumes build context is services/rag-service/
COPY requirements.txt .

# Install Python dependencies
# This will now install lightrag-hku and raganything from PyPI
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app /app/app
COPY .env /app/.env

# Create workspace directory for RAG
RUN mkdir -p /app/rag_workspace && chmod 777 /app/rag_workspace

# Expose port 8006
EXPOSE 8006

# Start command with port 8006
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8006"]
