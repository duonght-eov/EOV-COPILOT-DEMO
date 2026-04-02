import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import router, root_router

app = FastAPI(
    title="Water AI – Agentic Service",
    description="ReAct Agent điều phối RAG + Predict Service cho hệ thống cấp nước",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include both routers
app.include_router(router)      # /api/v1/agent/*
app.include_router(root_router)  # /api/* (AnythingLLM compatibility)


@app.get("/health")
async def health():
    return {"status": "green", "service": "agentic-service"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8005)),
        reload=True,
    )
