# SpeedMaint Intelligence UI

Modern web UI for SpeedMaint Intelligence RAG system, based on [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) (MIT License).

## Quick Start

```bash
# API Gateway
cd ../api-gateway
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080

# Frontend
cd frontend
yarn && yarn dev
```

**Access:** http://localhost:3000

## Documentation

- [Setup Guide](docs/SETUP.md)
- [API Documentation](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## Architecture

```
speedmaint-ui/
├── frontend/          # React app (AnythingLLM fork)
└── docs/              # Documentation
    ├── SETUP.md
    ├── API.md
    └── DEPLOYMENT.md

api-gateway/           # Python FastAPI backend
```

## License

Frontend: MIT (from AnythingLLM)

