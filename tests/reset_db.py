import asyncio
import os
from dotenv import load_dotenv
import asyncpg
from neo4j import GraphDatabase

load_dotenv()

POSTGRES_Config = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", 5432),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "password"),
    "database": os.getenv("POSTGRES_DATABASE", "rag_db"),
}

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

async def reset_postgres():
    print(f"🗑️ Clearing PostgreSQL Database: {POSTGRES_Config['database']}...")
    try:
        conn = await asyncpg.connect(**POSTGRES_Config)
        
        # Get all tables starting with lightrag_
        tables = await conn.fetch("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public' AND tablename LIKE 'lightrag_%'
        """)
        
        for table in tables:
            t_name = table['tablename']
            print(f"   Dropping table: {t_name}")
            await conn.execute(f"DROP TABLE IF EXISTS {t_name} CASCADE")
        
        await conn.close()
        print("✅ PostgreSQL Cleared.")
    except Exception as e:
        print(f"❌ PostgreSQL Error: {e}")

def reset_neo4j():
    print(f"🗑️ Clearing Neo4j Database: {NEO4J_URI}...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()
        print("✅ Neo4j Cleared.")
    except Exception as e:
        print(f"❌ Neo4j Error: {e}")

async def main():
    await reset_postgres()
    reset_neo4j()

if __name__ == "__main__":
    asyncio.run(main())
