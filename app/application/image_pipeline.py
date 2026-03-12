"""
application/image_pipeline.py

USE CASE: Detect và resolve image references trong câu trả lời.

Trách nhiệm:
  - Phát hiện [IMAGE_REF:...] trong LLM response
  - Resolve MinIO object path → proxied URL
  - Inject image URLs vào response

Thay thế: image detection logic trong app/services/query_engine.py

TODO (Phase 3): Tách từ query_engine.py vào đây.
"""
# placeholder — sẽ implement ở Phase 3
