# API Gateway for SpeedMaint Intelligence

API adapter service that bridges AnythingLLM frontend with RAGAnything backend.

## Setup

```bash
cd api-gateway
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --port 8080
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```
