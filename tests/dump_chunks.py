import sys
import os
import asyncio
import asyncpg
import json
from datetime import datetime

# Setup paths to import app modules
# Script is in tests/, so project root is ../
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

try:
    from app.config import settings
except ImportError:
    print("Could not import app.config. Please ensure you are running from the correct environment.")
    sys.exit(1)

async def dump_all_chunks():
    print(f"Connecting to PostgreSQL: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DATABASE}")
    
    conn = None
    try:
        conn = await asyncpg.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            database=settings.POSTGRES_DATABASE
        )
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # Dynamic table detection
    print("🔍 Listing available tables...")
    try:
        tables = await conn.fetch("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public';")
        table_names = [t['tablename'] for t in tables]
        target_table = next((t for t in table_names if "chunk" in t.lower() and "doc" in t.lower()), None)
        
        if not target_table:
            # Fallback to any chunk table if doc_chunk specific not found
            target_table = next((t for t in table_names if "chunk" in t.lower()), None)

        if not target_table:
            print(f"❌ Could not find chunks table. Available: {table_names}")
            await conn.close()
            return
            
        print(f"📋 Dumping data from table: {target_table}")
        rows = await conn.fetch(f'SELECT * FROM "{target_table}"')
        
    except Exception as e:
        print(f"❌ Error during fetch: {e}")
        if conn: await conn.close()
        return

    # Process Data and Extract Content
    print(f"✅ Fetched {len(rows)} rows. Extracting content...")

    output_content = []
    
    for row in rows:
        item = dict(row)
        chunk_id = str(item.get('id', 'Unknown'))
        raw_content = item.get('content', '')
        
        # Determine Page Index (if available in metadata)
        page_str = "?"
        try:
            if 'metadata' in item and item['metadata']:
                meta = item['metadata']
                if isinstance(meta, str) and meta.startswith('{'):
                    meta = json.loads(meta)
                if isinstance(meta, dict):
                    page_str = str(meta.get('page_idx', '?'))
        except:
            pass

        # Parse Content
        final_text = ""
        
        # Case 1: Content is a simple string
        if not isinstance(raw_content, str) or not raw_content.strip().startswith(('{', '[')):
             final_text = str(raw_content)
             
        # Case 2: Content is JSON string (common in our RAG)
        else:
            try:
                parsed = json.loads(raw_content)
                
                # Handle Mineru/Custom format where content is wrapped in a dict
                if isinstance(parsed, dict) and 'content' in parsed and isinstance(parsed['content'], list):
                    parsed = parsed['content']

                # Helper to extract text from an item
                def extract_item_text(p):
                    if isinstance(p, dict):
                        # Construct page prefix
                        page_pfx = ""
                        if 'page_idx' in p:
                            page_pfx = f"[Page {p['page_idx']}] "
                        
                        # Extract content based on type
                        if 'table_body' in p:
                            return f"{page_pfx}[TABLE]\n{p['table_body']}"
                        elif 'image_caption' in p:
                            caption = p['image_caption']
                            if isinstance(caption, list): caption = ", ".join(caption)
                            path = p.get('img_path', '')
                            return f"{page_pfx}[IMAGE]\nCaption: {caption}\nPath: {path}"
                        elif 'text' in p:
                            type_pfx = ""
                            if p.get('type') == 'heading':
                                type_pfx = "# " * p.get('level', 1)
                            return f"{page_pfx}{type_pfx}{p['text']}"
                    return str(p)

                # Process List or Dict
                if isinstance(parsed, list):
                    final_text = "\n\n".join([extract_item_text(x) for x in parsed])
                elif isinstance(parsed, dict):
                    final_text = extract_item_text(parsed)
                else:
                    final_text = str(parsed)
                
            except json.JSONDecodeError:
                final_text = raw_content # Parsing failed, use raw

        # Append to output list with nice formatting
        separator = "=" * 80
        chunk_block = (
            f"{separator}\n"
            f"CHUNK ID: {chunk_id} | PAGE: {page_str}\n"
            f"{separator}\n"
            f"{final_text.strip()}\n\n"
        )
        output_content.append(chunk_block)

    # Save to .txt file in all_results folder
    output_dir = os.path.join(current_dir, "all_results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "all_chunks_content.txt")
    
    print(f"💾 Saving readable content to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(output_content)

    print("🎉 Done! Open 'tests/all_results/all_chunks_content.txt' to view.")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(dump_all_chunks())
