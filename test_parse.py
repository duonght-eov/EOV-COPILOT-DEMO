import re

text = """-----Entities-----
entity details...
-----Relationships-----
relation details...
-----Sources-----
[1] content page 12 bla bla
[2] content page 15 bla bla
"""

def parse(ctx):
    sources = []
    if "-----Sources-----" in ctx:
        src_text = ctx.split("-----Sources-----")[-1].strip()
        chunks = re.split(r'\[\d+\]\s*', src_text)
        for i, c in enumerate(chunks):
            if c.strip():
                sources.append({
                    "content": c.strip(),
                    "doc_id": "Tài liệu hệ thống",
                })
    return sources

print(parse(text))
