# SpeedMaint Intelligence UI - Setup Guide

## Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- Docker & Docker Compose (optional)

---

## Quick Start

### 1. Clone và Setup

```bash
cd speedmaint-intelligence
```

### 2. API Gateway Setup

```bash
cd api-gateway

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env as needed

# Create data directory
mkdir -p data/uploads

# Run
uvicorn app.main:app --reload --port 8080
```

**Access:** http://localhost:8080/docs

### 3. Create Admin User

```bash
curl -X POST http://localhost:8080/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123","role":"admin"}'
```

### 4. Frontend Setup (Optional)

```bash
cd speedmaint-ui/frontend

# Install deps
yarn install  # or npm install

# Configure API URL
echo "VITE_API_BASE_URL=http://localhost:8080" > .env

# Run
yarn dev  # or npm run dev
```

---

## Docker Deployment

```bash
# Build and run all services
docker-compose up -d

# Services:
# - speedmaint-ui: http://localhost:3000
# - api-gateway: http://localhost:8080
# - rag-service: http://localhost:8000
# - streamlit: http://localhost:8501
```

---

## Configuration

### API Gateway (.env)

```bash
# Server
HOST=0.0.0.0
PORT=8080

# Database
DATABASE_URL=sqlite+aiosqlite:///./data/gateway.db

# JWT
SECRET_KEY=your-secret-key  # Change in production!
ACCESS_TOKEN_EXPIRE_MINUTES=60

# RAG Service
RAG_SERVICE_URL=http://localhost:8000

# CORS
CORS_ORIGINS=["http://localhost:3000"]
```

---

## 🧪 Verification

```bash
# Health check
curl http://localhost:8080/api/health

# Login
curl -X POST http://localhost:8080/api/auth/login \
  -d "username=admin&password=admin123"

# Expected: {"access_token":"...","token_type":"bearer"}
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | Change port in .env or use `--port XXXX` |
| Database error | Delete `data/gateway.db` and restart |
| CORS error | Add frontend URL to CORS_ORIGINS |
| RAG Service unavailable | Ensure rag_service is running on port 8000 |
