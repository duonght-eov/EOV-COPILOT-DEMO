import time
import asyncio
import hashlib
from typing import List, Dict, Set, Any, Optional
from collections import defaultdict
import logging

from app.config import settings

logger = logging.getLogger("ConsensusRetriever")

# In-memory keyword cache: {md5(query): (keywords_str, expire_timestamp)}
_keyword_cache: Dict[str, tuple] = {}
_KEYWORD_CACHE_TTL = 3600  # 60 phút


def _get_cached_keywords(query: str) -> Optional[str]:
    key = hashlib.md5(query.encode()).hexdigest()
    if key in _keyword_cache:
        kw, expire = _keyword_cache[key]
        if time.time() < expire:
            return kw
        del _keyword_cache[key]
    return None


def _set_cached_keywords(query: str, keywords: str):
    key = hashlib.md5(query.encode()).hexdigest()
    _keyword_cache[key] = (keywords, time.time() + _KEYWORD_CACHE_TTL)


def _normalize_map(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    min_val = min(score_map.values())
    max_val = max(score_map.values())
    if max_val == min_val:
        return {k: 1.0 for k in score_map.keys()}
    return {k: (v - min_val) / (max_val - min_val) for k, v in score_map.items()}


class ConsensusRetriever:
    def __init__(self, rag_instance):
        self.rag = rag_instance

    async def _extract_keywords(self, query: str) -> str:
        """
        Bước độc lập: trích xuất keywords từ query bằng LLM.
        - Kiểm tra cache trước để tránh gọi LLM nhiều lần.
        - Trả về query gốc nếu cache miss và LLM thất bại.
        """
        cached = _get_cached_keywords(query)
        if cached:
            logger.info(f"[Consensus] Keyword cache HIT: '{cached[:60]}'")
            return cached

        if not hasattr(self.rag, 'llm_model_func'):
            return query

        keywords_template = getattr(self.rag, 'keywords_extract_template', None)
        if not keywords_template:
            keywords_template = """Bạn là một công cụ trích xuất thực thể. Nhiệm vụ duy nhất của bạn là xuất ra danh sách các danh từ/thực thể/chủ đề cốt lõi từ câu hỏi.
                                    Yêu cầu:
                                    - Chỉ output các từ khóa, mỗi từ khóa cách nhau bằng dấu phẩy.
                                    - KHÔNG giải thích. KHÔNG viết lại câu hỏi hoặc lập luận.
                                    Câu hỏi: '{query}'
                                    Từ khóa:"""

        try:
            t0 = time.perf_counter()
            prompt = keywords_template.format(query=query)
            keyword_str = await self.rag.llm_model_func(prompt, max_tokens=50)
            ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[Consensus][TIMING] keyword_extraction={ms:.0f}ms")

            if keyword_str and ":" in keyword_str and "{" not in keyword_str:
                keyword_str = keyword_str.split(":")[-1].strip()

            if keyword_str and len(keyword_str.strip()) > 0:
                logger.info(f"[Consensus] Extracted keywords: {keyword_str[:80]}")
                _set_cached_keywords(query, keyword_str)
                return keyword_str
        except Exception as ke:
            logger.warning(f"[Consensus] Keyword extraction failed, using raw query. Error: {ke}")

        return query

    async def _get_naive_chunk_ids(self, query: str, top_k: int = 5) -> Dict[str, float]:
        """Naive Search: Vector search trực tiếp trên chunks. Trả về {ChunkID: Score}"""
        if not self.rag.chunks_vdb:
            return {}
        try:
            t0 = time.perf_counter()
            results = await self.rag.chunks_vdb.query(query, top_k=top_k)
            ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[Consensus][TIMING] naive_vector_search={ms:.0f}ms, found={len(results)}")
            total = len(results)
            return {
                res['id']: float(res.get('score') or (total - idx))
                for idx, res in enumerate(results)
            }
        except Exception:
            return {}

    async def _get_local_chunk_ids(
        self,
        keywords: str,
        top_k_entities: int = 5
    ) -> Dict[str, float]:
        """
        Local Search: Dùng keywords đã extract (truyền từ ngoài vào) để tìm Entity.
        Trả về {ChunkID: Score}.
        """
        if not self.rag.entities_vdb:
            return {}
        try:
            t0 = time.perf_counter()
            entities = await self.rag.entities_vdb.query(keywords, top_k=top_k_entities * 2)
            ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[Consensus][TIMING] entity_vector_search={ms:.0f}ms, found={len(entities)}")

            chunk_scores = defaultdict(float)
            t0 = time.perf_counter()
            count_mapped = 0
            total_entities = len(entities)
            delimiter = getattr(self.rag, 'tuple_delimiter', "<|#|>")

            for idx, entity in enumerate(entities):
                entity_key = entity.get('entity_name') or entity.get('id')
                entity_score = float(entity.get('score') or (total_entities - idx))
                node_data = await self.rag.chunk_entity_relation_graph.get_node(entity_key)
                if node_data and 'source_id' in node_data:
                    chunk_ids = node_data['source_id'].split(delimiter)
                    if chunk_ids:
                        count_mapped += 1
                    for cid in chunk_ids:
                        cid = cid.strip()
                        if cid:
                            chunk_scores[cid] += entity_score

            ms = (time.perf_counter() - t0) * 1000
            logger.info(
                f"[Consensus][TIMING] entity_to_chunk_mapping={ms:.0f}ms "
                f"(Neo4j x{len(entities)} nodes, mapped {count_mapped} entities → {len(chunk_scores)} chunks)"
            )
            return dict(chunk_scores)
        except Exception as e:
            logger.error(f"[Consensus][Local] Error: {e}")
            return {}

    async def _get_relation_chunk_ids(
        self,
        keywords: str,
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Relationship Search: Dùng keywords đã extract để tìm Relationship trong relations_vdb.
        Trả về {ChunkID: Score} từ cả relation.source_id và 2 đầu entity của mỗi relation.
        """
        if not settings.CONSENSUS_ENABLE_RELATION_SEARCH:
            return {}

        relations_vdb = getattr(self.rag, 'relationships_vdb', None)
        if not relations_vdb:
            logger.debug("[Consensus][Relation] relations_vdb not available, skipping.")
            return {}

        try:
            t0 = time.perf_counter()
            relations = await relations_vdb.query(keywords, top_k=top_k)
            ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[Consensus][TIMING] relation_vector_search={ms:.0f}ms, found={len(relations)}")

            if not relations:
                return {}

            delimiter = getattr(self.rag, 'tuple_delimiter', "<|#|>")
            entity_keys_to_resolve: Dict[str, float] = {}
            relation_source_chunks: Dict[str, float] = {}
            total_relations = len(relations)

            for idx, rel in enumerate(relations):
                rel_score = float(rel.get('score') or (total_relations - idx))

                # Chunk trực tiếp từ relation (nếu có source_id)
                direct_src = rel.get('source_id') or rel.get('src_id')
                if direct_src and isinstance(direct_src, str) and len(direct_src) < 500:
                    for cid in direct_src.split(delimiter):
                        cid = cid.strip()
                        if cid:
                            relation_source_chunks[cid] = max(
                                relation_source_chunks.get(cid, 0.0), rel_score
                            )

                # Entity đầu nguồn
                src_entity = rel.get('src_id') or rel.get('source') or rel.get('src')
                if src_entity and isinstance(src_entity, str) and len(src_entity) < 200:
                    entity_keys_to_resolve[src_entity] = max(
                        entity_keys_to_resolve.get(src_entity, 0.0), rel_score * 0.8
                    )

                # Entity đầu đích
                tgt_entity = rel.get('tgt_id') or rel.get('target') or rel.get('tgt')
                if tgt_entity and isinstance(tgt_entity, str) and len(tgt_entity) < 200:
                    entity_keys_to_resolve[tgt_entity] = max(
                        entity_keys_to_resolve.get(tgt_entity, 0.0), rel_score * 0.8
                    )

            # Resolve entity → chunk (Neo4j)
            chunk_scores = defaultdict(float, relation_source_chunks)
            t0 = time.perf_counter()
            count_mapped = 0
            for entity_key, entity_score in entity_keys_to_resolve.items():
                try:
                    node_data = await self.rag.chunk_entity_relation_graph.get_node(entity_key)
                    if node_data and 'source_id' in node_data:
                        for cid in node_data['source_id'].split(delimiter):
                            cid = cid.strip()
                            if cid:
                                chunk_scores[cid] += entity_score
                                count_mapped += 1
                except Exception:
                    pass

            ms = (time.perf_counter() - t0) * 1000
            logger.info(
                f"[Consensus][TIMING] relation_entity_mapping={ms:.0f}ms "
                f"(Neo4j x{len(entity_keys_to_resolve)} entities, mapped {count_mapped} chunks)"
            )
            return dict(chunk_scores)

        except Exception as e:
            logger.error(f"[Consensus][Relation] Error: {e}")
            return {}

    async def consensus_search(
        self,
        query: str,
        top_k_each_method: int = 5,
        final_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Pipeline 3 bước:
          Bước 0: Extract keywords (1 lần, dùng chung cho Local + Relation)
          Bước 1: Naive + Local + Relation chạy song song
          Bước 2: Gold/Silver/Bronze scoring → fetch chunks → sort by page
        """
        t_start = time.perf_counter()

        # ─── Bước 0: Extract keywords ───────────────────────────────────
        # Phải chạy TRƯỚC gather để tránh race condition và LLM gọi 2 lần
        t0 = time.perf_counter()
        keywords = await self._extract_keywords(query)
        logger.info(
            f"[Consensus][TIMING] keyword_step={1000*(time.perf_counter()-t0):.0f}ms "
            f"| keywords='{keywords[:60]}'"
        )

        # ─── Bước 1: 3 nhánh song song, dùng chung keywords ─────────────
        naive_map, local_map, relation_map = await asyncio.gather(
            self._get_naive_chunk_ids(query, top_k=top_k_each_method),
            self._get_local_chunk_ids(keywords, top_k_entities=top_k_each_method),
            self._get_relation_chunk_ids(keywords, top_k=settings.CONSENSUS_RELATION_TOP_K),
        )

        enable_relation = settings.CONSENSUS_ENABLE_RELATION_SEARCH and bool(relation_map)
        logger.info(
            f"[Consensus] Naive={len(naive_map)}, Local={len(local_map)}, "
            f"Relation={len(relation_map)} chunks (relation_enabled={enable_relation})"
        )

        # ─── Bước 2: Normalize + Weighted Scoring ───────────────────────
        naive_norm = _normalize_map(naive_map)
        local_norm = _normalize_map(local_map)
        relation_norm = _normalize_map(relation_map) if enable_relation else {}

        W_NAIVE    = settings.CONSENSUS_WEIGHT_NAIVE    if enable_relation else 0.60
        W_LOCAL    = settings.CONSENSUS_WEIGHT_LOCAL    if enable_relation else 0.40
        W_RELATION = settings.CONSENSUS_WEIGHT_RELATION if enable_relation else 0.0

        all_ids = set(naive_map) | set(local_map) | set(relation_map)
        chunk_scores: Dict[str, float] = {}
        gold_ids: Set[str] = set()
        silver_ids: Set[str] = set()

        for cid in all_ids:
            score = (
                naive_norm.get(cid, 0.0) * W_NAIVE
                + local_norm.get(cid, 0.0) * W_LOCAL
                + relation_norm.get(cid, 0.0) * W_RELATION
            )
            sources_count = sum([
                cid in naive_map,
                cid in local_map,
                cid in relation_map and enable_relation,
            ])
            if sources_count == 3:
                score += 0.2  # Gold bonus
                gold_ids.add(cid)
            elif sources_count == 2:
                silver_ids.add(cid)
            chunk_scores[cid] = score

        # ─── Bước 3: Chọn lọc Gold → Silver → Bronze ────────────────────
        bronze_ids = all_ids - gold_ids - silver_ids

        final_selected_ids: List[str] = []
        for tier in (gold_ids, silver_ids, bronze_ids):
            for cid in sorted(tier, key=lambda x: chunk_scores[x], reverse=True):
                if cid not in final_selected_ids:
                    final_selected_ids.append(cid)
                if len(final_selected_ids) >= final_k:
                    break
            if len(final_selected_ids) >= final_k:
                break

        logger.info(
            f"[Consensus] Gold={len(gold_ids)}, Silver={len(silver_ids)}, "
            f"Bronze={len(bronze_ids)} → Selected {len(final_selected_ids)}"
        )

        # ─── Bước 4: Fetch Chunk Content (PostgreSQL) ───────────────────
        t0 = time.perf_counter()
        final_results = []
        for cid in final_selected_ids:
            chunk_data = await self.rag.text_chunks.get_by_id(cid)
            if chunk_data:
                tier_label = (
                    "gold" if cid in gold_ids
                    else "silver" if cid in silver_ids
                    else "bronze"
                )
                chunk_data['consensus_source'] = tier_label
                chunk_data['total_score'] = chunk_scores[cid]
                final_results.append(chunk_data)
        ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[Consensus][TIMING] chunk_fetch={ms:.0f}ms ({len(final_selected_ids)} chunks)")

        # ─── Bước 5: Sort theo Page Index ───────────────────────────────
        def get_page_idx(chunk):
            meta = chunk.get('metadata', {})
            if isinstance(meta, dict):
                try:
                    return int(meta['page_idx'])
                except (KeyError, ValueError, TypeError):
                    pass
            return 999999

        final_results.sort(key=get_page_idx)

        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(f"[Consensus][TIMING] total={total_ms:.0f}ms")
        logger.info(f"[Consensus] Returning {len(final_results)} chunks to Reranker/LLM.")
        for i, res in enumerate(final_results[:3]):
            snippet = res.get('content', '')[:200].replace('\n', ' ')
            logger.info(f"   - Chunk {i+1} ({res.get('consensus_source')}): {snippet}...")

        return final_results