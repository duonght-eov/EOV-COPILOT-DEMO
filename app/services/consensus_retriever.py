import time
import asyncio
import hashlib
from typing import List, Dict, Set, Any
from collections import defaultdict
import logging

from app.config import settings

logger = logging.getLogger("ConsensusRetriever")

# In-memory keyword cache: {md5(query): (keywords_str, expire_timestamp)}
_keyword_cache: Dict[str, tuple] = {}
_KEYWORD_CACHE_TTL = 3600  # 60 phút


def _get_cached_keywords(query: str) -> str | None:
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

    async def _get_naive_chunk_ids(self, query: str, top_k: int = 5) -> Dict[str, float]:
        """Naive Search: Trả về {ChunkID: Score}"""
        if not self.rag.chunks_vdb:
            return {}
        try:
            t0 = time.perf_counter()
            results = await self.rag.chunks_vdb.query(query, top_k=top_k)
            ms = (time.perf_counter() - t0) * 1000
            if results:
                logger.info(f"Naive Raw first item: {results[0]}")
            logger.info(f"[Consensus][TIMING] naive_vector_search={ms:.0f}ms, found={len(results)}")

            total = len(results)
            return {
                res['id']: float(res.get('score') or (total - idx))
                for idx, res in enumerate(results)
            }
        except Exception:
            return {}

    async def _get_local_chunk_ids(self, query: str, top_k_entities: int = 5) -> Dict[str, float]:
        """Local Search: Trả về {ChunkID: Score}"""
        if not self.rag.entities_vdb:
            return {}
        try:
            # 1. Keyword Extraction (with cache)
            search_query = query
            keywords_template = getattr(self.rag, 'keywords_extract_template', None)
            if not keywords_template:
                keywords_template = """
                    Dựa vào câu hỏi của người dùng, hãy trích xuất các từ khóa quan trọng (Entities, Concepts) để tìm kiếm trong cơ sở dữ liệu.
                    Chỉ trả về danh sách từ khóa, ngăn cách bằng dấu phẩy. TUYỆT ĐỐI KHÔNG DỊCH sang tiếng Anh.

                    Câu hỏi: {query}
                    Từ khóa (Tiếng Việt):
                """

            if hasattr(self.rag, 'llm_model_func'):
                cached = _get_cached_keywords(query)
                if cached:
                    logger.info(f"[Consensus] Keyword cache HIT: '{cached[:50]}'")
                    search_query = cached
                else:
                    try:
                        t0 = time.perf_counter()
                        prompt = keywords_template.format(query=query)
                        keyword_str = await self.rag.llm_model_func(prompt)
                        ms = (time.perf_counter() - t0) * 1000
                        logger.info(f"[Consensus][TIMING] keyword_extraction={ms:.0f}ms")

                        if keyword_str and ":" in keyword_str and "{" not in keyword_str:
                            keyword_str = keyword_str.split(":")[-1].strip()

                        if keyword_str and len(keyword_str.strip()) > 0:
                            logger.info(f"Consensus: Extracted keywords: {keyword_str}")
                            _set_cached_keywords(query, keyword_str)
                            search_query = keyword_str
                    except Exception as ke:
                        logger.warning(f"Consensus: Keyword extraction failed, using raw query. Error: {ke}")

            # 2. Entity Vector Search
            t0 = time.perf_counter()
            entities = await self.rag.entities_vdb.query(search_query, top_k=top_k_entities * 2)
            ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[Consensus][TIMING] entity_vector_search={ms:.0f}ms, found={len(entities)}")
            logger.info(f"Local Entity Search found: {len(entities)} entities for query '{search_query[:20]}...'")

            # 3. Entity → Chunk Mapping (Neo4j/Graph)
            chunk_scores = defaultdict(float)
            t0 = time.perf_counter()
            count_mapped = 0
            total_entities = len(entities)
            for idx, entity in enumerate(entities):
                entity_key = entity.get('entity_name') or entity.get('id')
                entity_score = float(entity.get('score') or (total_entities - idx))
                node_data = await self.rag.chunk_entity_relation_graph.get_node(entity_key)
                if node_data and 'source_id' in node_data:
                    source_ids_str = node_data['source_id']
                    delimiter = getattr(self.rag, 'tuple_delimiter', "<|#|>")
                    chunk_ids = source_ids_str.split(delimiter)
                    if chunk_ids:
                        count_mapped += 1
                    for cid in chunk_ids:
                        cid = cid.strip()
                        if cid:
                            chunk_scores[cid] += entity_score
            ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[Consensus][TIMING] entity_to_chunk_mapping={ms:.0f}ms (Neo4j x{len(entities)} nodes)")
            logger.info(f"Local: Mapped {count_mapped} entities to {len(chunk_scores)} unique chunks.")
            return dict(chunk_scores)
        except Exception as e:
            logger.error(f"Error in Local Search: {e}")
            return {}

    async def _get_relation_chunk_ids(self, query: str, top_k: int = 5) -> Dict[str, float]:
        """
        Relationship Search (mới): Trả về {ChunkID: Score}.
        1. Tìm Top-K Relationship từ relations_vdb gần với query nhất.
        2. Mỗi Relationship có source_entity + target_entity.
        3. Tra Graph để lấy source_id của từng entity đầu/cuối.
        4. Trả về chunk_ids từ cả 3 nguồn: relation.source_id, source_entity chunks, target_entity chunks.
        """
        if not settings.CONSENSUS_ENABLE_RELATION_SEARCH:
            return {}

        relations_vdb = getattr(self.rag, 'relationships_vdb', None)
        if not relations_vdb:
            logger.debug("[Consensus][Relation] relations_vdb not available, skipping.")
            return {}

        try:
            t0 = time.perf_counter()
            # 1. Lấy từ khóa đã cache (nếu có) để tăng chất lượng vector search
            search_query = _get_cached_keywords(query) or query
            relations = await relations_vdb.query(search_query, top_k=top_k)
            ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[Consensus][TIMING] relation_vector_search={ms:.0f}ms, found={len(relations)}")

            if not relations:
                return {}

            # 2. Thu thập entity keys từ 2 đầu Relationship
            entity_keys_to_resolve: Dict[str, float] = {}
            relation_source_chunks: Dict[str, float] = {}
            delimiter = getattr(self.rag, 'tuple_delimiter', "<|#|>")
            total_relations = len(relations)

            for idx, rel in enumerate(relations):
                rel_score = float(rel.get('score') or (total_relations - idx))

                # Một số LightRAG version lưu src_id trực tiếp trong relation record
                direct_src = rel.get('source_id') or rel.get('src_id')
                if direct_src:
                    for cid in direct_src.split(delimiter):
                        cid = cid.strip()
                        if cid:
                            relation_source_chunks[cid] = max(relation_source_chunks.get(cid, 0.0), rel_score)

                # Đầu nguồn
                src_entity = rel.get('src_id') or rel.get('source') or rel.get('src')
                if src_entity and isinstance(src_entity, str) and len(src_entity) < 200:
                    entity_keys_to_resolve[src_entity] = max(
                        entity_keys_to_resolve.get(src_entity, 0.0), rel_score * 0.8
                    )

                # Đầu đích
                tgt_entity = rel.get('tgt_id') or rel.get('target') or rel.get('tgt')
                if tgt_entity and isinstance(tgt_entity, str) and len(tgt_entity) < 200:
                    entity_keys_to_resolve[tgt_entity] = max(
                        entity_keys_to_resolve.get(tgt_entity, 0.0), rel_score * 0.8
                    )

            # 3. Resolve Entity → Chunk (Neo4j) — batch style
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
            logger.error(f"[Consensus][Relation] Error in Relationship Search: {e}")
            return {}

    async def consensus_search(self, query: str, top_k_each_method: int = 5, final_k: int = 3) -> List[Dict[str, Any]]:
        """
        Chiến lược 3 nguồn:
        1. Lấy top_k_each_method từ Naive, Local, và Relationship Search.
        2. Phân loại chunk theo 3 mức:
           - Gold  : Có ở cả 3 nguồn (bonus +0.2)
           - Silver: Có ở 2 trong 3 nguồn
           - Bronze: Có ở 1 nguồn (chỉ làm filler)
        3. Ưu tiên Gold > Silver > Bronze cho đến đủ final_k.
        """
        t_start = time.perf_counter()

        # 1. Chạy song song Naive + Local + Relation
        task_naive = self._get_naive_chunk_ids(query, top_k=top_k_each_method)
        task_local = self._get_local_chunk_ids(query, top_k_entities=top_k_each_method)
        task_relation = self._get_relation_chunk_ids(query, top_k=settings.CONSENSUS_RELATION_TOP_K)
        naive_map, local_map, relation_map = await asyncio.gather(task_naive, task_local, task_relation)

        enable_relation = settings.CONSENSUS_ENABLE_RELATION_SEARCH and bool(relation_map)
        logger.info(
            f"Consensus DEBUG: Naive={len(naive_map)}, Local={len(local_map)}, "
            f"Relation={len(relation_map)} chunks (enabled={enable_relation})"
        )

        # 2. Normalize từng nguồn
        naive_norm = _normalize_map(naive_map)
        local_norm = _normalize_map(local_map)
        relation_norm = _normalize_map(relation_map) if enable_relation else {}

        W_NAIVE = settings.CONSENSUS_WEIGHT_NAIVE if enable_relation else 0.60
        W_LOCAL = settings.CONSENSUS_WEIGHT_LOCAL if enable_relation else 0.40
        W_RELATION = settings.CONSENSUS_WEIGHT_RELATION if enable_relation else 0.0

        # 3. Tính tổng điểm và phân loại
        all_ids = set(naive_map.keys()) | set(local_map.keys()) | set(relation_map.keys())
        chunk_scores: Dict[str, float] = {}
        gold_ids: Set[str] = set()
        silver_ids: Set[str] = set()

        for cid in all_ids:
            n_score = naive_norm.get(cid, 0.0) * W_NAIVE
            l_score = local_norm.get(cid, 0.0) * W_LOCAL
            r_score = relation_norm.get(cid, 0.0) * W_RELATION
            score = n_score + l_score + r_score

            # Đếm số nguồn có chunk này
            sources_count = sum([
                cid in naive_map,
                cid in local_map,
                cid in relation_map and enable_relation
            ])

            if sources_count == 3:
                score += 0.2  # Gold bonus
                gold_ids.add(cid)
            elif sources_count == 2:
                silver_ids.add(cid)

            chunk_scores[cid] = score

        # 4. Chọn lọc theo thứ tự Gold → Silver → Bronze
        gold_sorted = sorted(gold_ids, key=lambda x: chunk_scores[x], reverse=True)
        silver_sorted = sorted(silver_ids, key=lambda x: chunk_scores[x], reverse=True)
        bronze_sorted = sorted(
            all_ids - gold_ids - silver_ids,
            key=lambda x: chunk_scores[x], reverse=True
        )

        final_selected_ids: List[str] = []
        for candidates in [gold_sorted, silver_sorted, bronze_sorted]:
            for cid in candidates:
                if cid not in final_selected_ids:
                    final_selected_ids.append(cid)
                if len(final_selected_ids) >= final_k:
                    break
            if len(final_selected_ids) >= final_k:
                break

        logger.info(
            f"Consensus: Gold={len(gold_ids)}, Silver={len(silver_ids)}, "
            f"Bronze={len(all_ids - gold_ids - silver_ids)} → Selected {len(final_selected_ids)}"
        )

        # 5. Fetch Chunk Content (PostgreSQL)
        t0 = time.perf_counter()
        final_results = []
        for cid in final_selected_ids:
            chunk_data = await self.rag.text_chunks.get_by_id(cid)
            if chunk_data:
                tier = "gold" if cid in gold_ids else ("silver" if cid in silver_ids else "bronze")
                chunk_data['consensus_source'] = tier
                chunk_data['total_score'] = chunk_scores[cid]
                final_results.append(chunk_data)
        ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[Consensus][TIMING] chunk_fetch={ms:.0f}ms (PostgreSQL x{len(final_selected_ids)} chunks)")

        # 6. Re-rank by Page Index
        def get_page_idx(chunk):
            meta = chunk.get('metadata', {})
            if isinstance(meta, dict) and 'page_idx' in meta:
                try:
                    return int(meta['page_idx'])
                except Exception:
                    pass
            return 999999

        final_results.sort(key=get_page_idx)

        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(f"[Consensus][TIMING] total={total_ms:.0f}ms")
        logger.info(f"Consensus: Returning {len(final_results)} chunks to Reranker/LLM.")
        for i, res in enumerate(final_results[:3]):
            content_snippet = res.get('content', '')[:200].replace('\n', ' ')
            logger.info(f"   - Chunk {i+1} ({res.get('consensus_source')}): {content_snippet}...")

        return final_results