import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def main():
    engine = create_async_engine("postgresql+asyncpg://ocr_cuong:ocr_cuong@localhost:5432/api_gateway_db")
    async with engine.begin() as conn:
        await conn.execute(text("UPDATE workspaces SET query_mode = 'mix' WHERE query_mode = 'consensus'"))
    print("Database updated successfully")

asyncio.run(main())
