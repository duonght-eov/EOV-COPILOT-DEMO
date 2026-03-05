import logging
import httpx
from typing import List, Union
from app.config import get_settings

logger = logging.getLogger(__name__)

class EmbeddingModel:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance.settings = get_settings()
            cls._instance.client = httpx.AsyncClient(timeout=30.0)
            cls._instance.tei_url = cls._instance.settings.TEI_ENDPOINT.rstrip("/") + "/embed"
        return cls._instance
    
    def load(self):
        logger.info(f"Targeting TEI Service at: {self.tei_url}")
        
    async def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            response = await self.client.post(
                self.tei_url,
                json={"inputs": texts, "normalize": True, "truncate": True}
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to encode texts via TEI: {str(e)}")
            raise e

model_instance = EmbeddingModel()