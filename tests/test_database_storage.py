import asyncio
import sys
import os
from lightrag.base import QueryParam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.indexing_engine import IndexingEngine
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("DB_TEST")

async def test_database_storage():
    logger.info("="*60)
    logger.info("🧪 Testing Database Storage")
    logger.info("="*60)
    logger.info(f"Storage Type: {settings.STORAGE_TYPE}")
    logger.info(f"Graph Storage: {settings.ENABLE_GRAPH_STORAGE}")
    logger.info(f"Graph Type: {settings.GRAPH_STORAGE_TYPE}")
    logger.info("="*60)
    
    # Initialize engine
    engine = IndexingEngine(doc_id="test_doc")
    
    # Initialize async storages
    await engine._ensure_storages_initialized()
    
    # Test text
    test_text = """
    SpeedMaint là phần mềm CMMS cloud hàng đầu cho quản lý bảo trì công nghiệp.
    Hệ thống giúp theo dõi thiết bị, lập lịch bảo trì phòng ngừa, và quản lý work orders hiệu quả.
    """
    
    logger.info("📝 Inserting test text...")
    await engine.rag_anything.lightrag.ainsert(test_text)
    logger.info("✅ Text inserted to database")
    
    # Test query
    logger.info("\n🔍 Testing query...")
    result = await engine.rag_anything.lightrag.aquery(
        "SpeedMaint là gì?",
        param=QueryParam(mode="hybrid")
    )
    
    logger.info(f"\n📊 Query Result:\n{result}\n")
    logger.info("="*60)
    logger.info("✅ Database storage test completed!")
    logger.info("="*60)

if __name__ == "__main__":
    asyncio.run(test_database_storage())
