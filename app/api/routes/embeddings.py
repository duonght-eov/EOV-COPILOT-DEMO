from fastapi import APIRouter, HTTPException, status, Depends
from typing import Any
import logging
from app.schemas.embedding_schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    BatchEmbeddingRequest,
    BatchEmbeddingResponse
)
from app.services.embedding_service import EmbeddingService, get_embedding_service

router = APIRouter()
logger = logging.getLogger(__name__)

def get_service():
    return get_embedding_service()

@router.post(
    "/text",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate vector for a text",
    description="Transform a single text to a 1024 dimensions vector."
)
async def create_embedding(
    request: EmbeddingRequest,
    service: EmbeddingService = Depends(get_service)
) -> Any:
    try:
        vector = await service.embed_text(request.text)
        return EmbeddingResponse(
            vector=vector,
            dimensions=len(vector),
            model=service.settings.MODEL_NAME
        )
    except Exception as e:
        logger.error(f"Error processing text embedding: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {str(e)}"
        )
        
@router.post(
    "/batch",
    response_model=BatchEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate vector for list of text (Batch)",
    description="Optimize performance by processing multiple text segments in parallel on the GPU."
)
async def create_batch_embeddings(
    request: BatchEmbeddingRequest,
    service: EmbeddingService = Depends(get_service)
) -> Any:
    try:
        vectors = await service.embed_batch(request.texts)
        return BatchEmbeddingResponse(
            vectors=vectors,
            total_processed=len(vectors),
            model=service.settings.MODEL_NAME
        )
    except Exception as e:
        logger.error(f"Error processing batch embedding: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAl_SERVER_ERROR,
            detail=f"Internal Server Error: {str(e)}"
        )