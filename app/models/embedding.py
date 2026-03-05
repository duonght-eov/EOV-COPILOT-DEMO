from pydantic import BaseModel, Field
from typing import List
from uuid import UUID
from datetime import datetime

class Embedding(BaseModel):
    chunk_id: UUID = Field(..., description="UUID of original chunk text.")
    vector: List[float] = Field(..., description="Floating list representing vector.")
    model_name: str = Field(..., description="Embedding Model name.")
    created_at: datetime = Field(default_factory=datetime.now, description="Embedding generation time.")
    
    class Config:
        populate_by_name = True