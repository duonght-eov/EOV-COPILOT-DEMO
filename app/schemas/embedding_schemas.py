from pydantic import BaseModel, Field
from typing import List, Optional

# request schema
class EmbeddingRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="The text to be embedded. Must be a non-empty string."
    )

class BatchEmbeddingRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_items=1,
        description="List of texts to be embedded. Each text must be a non-empty string."
    )
    
# response schema
class EmbeddingResponse(BaseModel):
    vector: List[float] = Field(
        ...,
        description="The floating array representing the embedding vector of the input text."
    )
    dimensions: int = Field(
        ...,
        description="The dimensionality of the embedding vector."
    )
    model: str = Field(
        ...,
        description="The name of the embedding model used."
    )
    
class BatchEmbeddingResponse(BaseModel):
    vectors: List[List[float]] = Field(
        ...,
        description="List of embedding vectors for each input text."
    )
    total_processed: int = Field(
        ...,
        description="Total number of texts processed."
    )
    model: str = Field(
        ...,
        description="The name of the embedding model used."
    )