import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import embeddings
from app.core.embedding_model import model_instance 
from app.core.rabbitmq_consumer import RabbitMQConsumer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()
mq_consumer = RabbitMQConsumer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Embedding Service...")
    
    try:
        model_instance.load()
        logger.info(f"Embedding Service initialized (TEI Backend).")
    except Exception as e:
        logger.error(f"Failed to initialize Embedding Service: {e}")
    
    # Chạy RabbitMQ Consumer
    asyncio.create_task(mq_consumer.connect())
    
    yield
    
    logger.info("Shutting down Embedding Service...")
    await mq_consumer.close()
    
app = FastAPI(
    title=settings.APP_NAME,
    description="A microservice for text-to-vector embedding using BGE-M3.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    embeddings.router,
    prefix="/api/v1/embed",
    tags=["Embeddings"]
)

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "model": settings.MODEL_NAME,
        "backend": "TEI (Text Embeddings Inference)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8003, reload=True)