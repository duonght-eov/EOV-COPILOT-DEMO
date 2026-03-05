import json
import logging
import asyncio
import aio_pika
import redis
from typing import List, Dict, Any

from app.config import get_settings
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class RabbitMQConsumer:
    def __init__(self):
        self.settings = get_settings()
        self.connection = None
        self.channel = None
        self.embedding_service = EmbeddingService()
        self.redis_client = redis.from_url(self.settings.REDIS_URL, decode_responses=True)

    async def connect(self):
        retry_delay = 5
        while True:
            try:
                self.connection = await aio_pika.connect_robust(self.settings.RABBITMQ_URL)
                self.channel = await self.connection.channel()

                input_queue_name = self.settings.RABBITMQ_QUEUE_CONSUME
                queue = await self.channel.declare_queue(input_queue_name, durable=True)

                output_queue_name = self.settings.RABBITMQ_QUEUE_PUBLISH
                await self.channel.declare_queue(output_queue_name, durable=True)

                await self.channel.set_qos(prefetch_count=1)
                await queue.consume(self.process_message)
                logger.info(f"RabbitMQ Connected. Listening on '{input_queue_name}' -> Publishing to '{output_queue_name}'")
                break 
            
            except Exception as e:
                logger.warning(f"Failed to connect RabbitMQ. Retrying in {retry_delay}s... Error: {e}")
                await asyncio.sleep(retry_delay)
            
    async def process_message(self, message: aio_pika.IncomingMessage):
        async with message.process():
            try:
                body = message.body.decode()
                signal = json.loads(body)
                
                doc_id = signal.get("document_id")
                redis_key_chunks = signal.get("redis_key")
                batch_idx = signal.get("batch_index", 0)
                
                logger.info(f"Processing Job for Doc ID: {doc_id} | Batch: {batch_idx}")
                if not redis_key_chunks:
                    logger.error("Message missing 'redis_key'. Skipping.")
                    return

                chunks_data_json = self.redis_client.get(redis_key_chunks)
                if not chunks_data_json:
                    logger.warning(f"Redis key {redis_key_chunks} is empty or expired.")
                    return
                
                chunks_data = json.loads(chunks_data_json)
                texts_to_embed = [item.get("text_content", "") for item in chunks_data]
                
                logger.info(f"   -> Embedding {len(texts_to_embed)} chunks...")
                vectors = await self.embedding_service.embed_batch(texts_to_embed)
                
                if len(vectors) != len(chunks_data):
                    logger.error("Mismatch between input chunks and output vectors count!")
                    return

                vectors_payload = []
                for idx, chunk_item in enumerate(chunks_data):
                    vector_item = {
                        "document_id": doc_id,
                        "chunk_id": chunk_item.get("id"),
                        "chunk_index": chunk_item.get("chunk_index"),
                        "text_content": chunk_item.get("text_content"),
                        "embedding": vectors[idx], 
                        "metadata_info": chunk_item.get("metadata_info", {})
                    }
                    vectors_payload.append(vector_item)

                redis_key_vectors = f"job:{doc_id}:batch:{batch_idx}:vectors"
                self.redis_client.setex(
                    name=redis_key_vectors,
                    time=3600, # TTL 1 giờ
                    value=json.dumps(vectors_payload)
                )
                logger.info(f"   -> Saved {len(vectors_payload)} vectors to Redis: {redis_key_vectors}")

                output_queue = self.settings.RABBITMQ_QUEUE_PUBLISH
                
                next_signal = {
                    "document_id": doc_id,
                    "status": "vectors_ready",
                    "redis_key": redis_key_vectors, 
                    "total_vectors": len(vectors_payload)
                }
                
                await self.channel.default_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(next_signal).encode(),
                        content_type="application/json",
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                    ),
                    routing_key=output_queue
                )
                
                logger.info(f"Signal sent to '{output_queue}' for Doc ID: {doc_id}")

            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def close(self):
        if self.connection:
            await self.connection.close()
            logger.info("RabbitMQ connection closed.")