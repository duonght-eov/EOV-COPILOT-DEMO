import asyncpg
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("CLEANUP")

async def clean_postgres(workspace_slug: str) -> int:
    logger.info(f"[*] Đang xoá dữ liệu Postgres cho workspace: {workspace_slug}")
    try:
        conn = await asyncpg.connect(
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            database=settings.POSTGRES_DATABASE,
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT
        )
        
        tables_records = await conn.fetch(f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{settings.POSTGRES_SCHEMA}'")
        tables = [r['table_name'] for r in tables_records]
        
        deleted_total = 0
        for table in tables:
            has_workspace = await conn.fetchval(f"SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_schema = '{settings.POSTGRES_SCHEMA}' AND table_name = '{table}' AND column_name = 'workspace')")
            if has_workspace:
                result = await conn.execute(f"DELETE FROM {table} WHERE workspace = $1", workspace_slug)
                count = int(result.split(" ")[1])
                deleted_total += count
                if count > 0:
                    logger.info(f"    -> Đã xoá {count} bản ghi từ bảng {table}")
                    
        logger.info(f"[+] Dọn dẹp xong {deleted_total} bản ghi trong Postgres cho {workspace_slug}.")
        await conn.close()
        return deleted_total
    except Exception as e:
        logger.error(f"[-] Lỗi khi dọn dẹp Postgres: {e}")
        raise e

async def clean_neo4j(workspace_slug: str):
    if not settings.ENABLE_GRAPH_STORAGE:
        return 0, 0
        
    logger.info(f"[*] Đang xoá dữ liệu Neo4j cho workspace label: {workspace_slug}")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD))
        
        with driver.session() as session:
             query = f"MATCH (n:`{workspace_slug}`) DETACH DELETE n"
             result = session.run(query)
             summary = result.consume()
             deleted_nodes = summary.counters.nodes_deleted
             deleted_rels = summary.counters.relationships_deleted
             
             logger.info(f"    -> Đã xoá {deleted_nodes} nodes (Thực thể) và {deleted_rels} relationships")

        driver.close()
        return deleted_nodes, deleted_rels
    except Exception as e:
        logger.error(f"[-] Lỗi khi dọn dẹp Neo4j: {e}")
        raise e

async def clean_document(workspace_slug: str, filename: str) -> int:
    logger.info(f"[*] Đang tìm và xoá document '{filename}' cho workspace: {workspace_slug}")
    try:
        # Load indexing engine and get rag instance
        from app.services.indexing_engine import default_engine
        rag = await default_engine._get_or_create_rag(workspace=workspace_slug)
        
        conn = await asyncpg.connect(
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            database=settings.POSTGRES_DATABASE,
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT
        )
        
        schema = settings.POSTGRES_SCHEMA if hasattr(settings, 'POSTGRES_SCHEMA') else 'public'
        query = f"SELECT id FROM {schema}.lightrag_doc_full WHERE workspace = $1 AND doc_name LIKE $2"
        doc_ids = await conn.fetch(query, workspace_slug, f"%{filename}%")
        
        deleted_count = 0
        for record in doc_ids:
            doc_id = record['id']
            logger.info(f"    -> Đang xoá doc_id={doc_id} từ LightRAG cho file {filename}")
            try:
                await rag.adelete_by_doc_id(doc_id)
                deleted_count += 1
            except Exception as inner_e:
                logger.error(f"Lỗi khi xóa doc_id={doc_id}: {inner_e}")
                
        await conn.close()
        logger.info(f"[+] Dọn dẹp xong document '{filename}'. Đã xoá {deleted_count} bản ghi gốc.")
        return deleted_count
    except Exception as e:
        logger.error(f"[-] Lỗi khi dọn dẹp document: {e}")
        raise e
