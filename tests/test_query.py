import asyncio
from app.services.query_engine import query_engine
import json

async def main():
    print("Testing query...")
    try:
        res = await query_engine.query('Các bước bảo trì dự phòng bao gồm những bước nào', mode='consensus')
        print('ANSWER:\n', res.get('answer'))
        print('IMAGES:\n', res.get('images', []))
        #print('RETRIEVED:\n', len(res.get('retrieved_chunks', [])))
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
