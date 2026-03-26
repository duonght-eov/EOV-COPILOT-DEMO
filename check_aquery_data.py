import asyncio
from app.infrastructure.graph.lightrag_factory import QueryRAGFactory
from lightrag import QueryParam

async def check():
    rag = await QueryRAGFactory.get_or_create_rag("qtxd")
    param = QueryParam(mode="mix", top_k=5)
    data = await rag.aquery_data("nhiệt độ tối đa là bao nhiêu", param=param)
    print("Keys:", data.keys())
    if "chunks" in data:
        print("Chunks keys:", [c.keys() for c in data["chunks"]][:1])
    print(data)

asyncio.run(check())
