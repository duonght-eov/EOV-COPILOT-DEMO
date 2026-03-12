"""
application/indexing_pipeline.py

USE CASE: Ingest tài liệu end-to-end.

Trách nhiệm:
  - Orchestrate: parse → chunk → embed → store vào graph/vector DB
  - KHÔNG chứa logic cụ thể (delegate xuống services/indexing/)

Thay thế: phần indexing trong app/services/indexing_engine.py

TODO (Phase 2): Di chuyển IndexingEngine logic vào đây.
"""
# placeholder — sẽ implement ở Phase 2
