import time
import asyncio
from typing import List, Dict, Set, Any
from collections import defaultdict
import logging

logger = logging.getLogger("ConsensusRetriever")


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
            # 1. Keyword Extraction
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
                        search_query = keyword_str
                except Exception as ke:
                    logger.warning(f"Consensus: Keyword extraction failed, using raw query. Error: {ke}")

            # 2. Entity Vector Search
            t0 = time.perf_counter()
            entities = await self.rag.entities_vdb.query(search_query, top_k=top_k_entities * 2)
            ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[Consensus][TIMING] entity_vector_search={ms:.0f}ms, found={len(entities)}")
            logger.info(f"Local Entity Search found: {len(entities)} entities for query '{search_query[:20]}...'")
            if entities:
                logger.info(f"First Entity Raw: {entities[0]}")

            # 3. Entity → Chunk Mapping (Neo4j)
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

    async def consensus_search(self, query: str, top_k_each_method: int = 5, final_k: int = 3) -> List[Dict[str, Any]]:
        """
        Chiến lược:
        1. Lấy top_k_each_method (mặc định 5) từ Naive và Local.
        2. Tìm các Chunk XUẤT HIỆN Ở CẢ 2 bên (Giao thoa). -> LẤY HẾT nhóm này.
        3. Nếu số lượng nhóm giao thoa < final_k (mặc định 3), lấy thêm các chunk có điểm cao nhất còn lại.
        """
        t_start = time.perf_counter()

        # 1. Chạy song song Naive + Local
        task_naive = self._get_naive_chunk_ids(query, top_k=top_k_each_method)
        task_local = self._get_local_chunk_ids(query, top_k_entities=top_k_each_method)
        naive_map, local_map = await asyncio.gather(task_naive, task_local)

        logger.info(f"Consensus DEBUG: Naive found {len(naive_map)} chunks.")
        for cid, sc in list(naive_map.items())[:5]:
            logger.info(f"   - Naive: {cid[:20]}... | Score: {sc:.4f}")
        logger.info(f"Consensus DEBUG: Local found {len(local_map)} chunks.")
        for cid, sc in list(local_map.items())[:5]:
            logger.info(f"   - Local: {cid[:20]}... | Score: {sc:.4f}")

        # 2. Phân loại Chunk
        intersection_ids: Set[str] = set()
        chunk_data_map: Dict[str, float] = {}

        all_ids = set(naive_map.keys()) | set(local_map.keys())
        for cid in all_ids:
            score = naive_map.get(cid, 0.0) + local_map.get(cid, 0.0)
            chunk_data_map[cid] = score
            if cid in naive_map and cid in local_map:
                intersection_ids.add(cid)

        # 3. Chọn lọc kết quả
        intersect_list = sorted(intersection_ids, key=lambda x: chunk_data_map[x], reverse=True)
        final_selected_ids = list(intersect_list)
        logger.info(f"Consensus: Found {len(intersect_list)} intersection chunks.")

        if len(final_selected_ids) < final_k:
            needed = final_k - len(final_selected_ids)
            ordered_unique = []
            for cid in naive_map.keys():
                if cid not in intersection_ids:
                    ordered_unique.append(cid)
            for cid in local_map.keys():
                if cid not in intersection_ids and cid not in ordered_unique:
                    ordered_unique.append(cid)
            ordered_unique.sort(key=lambda x: chunk_data_map[x], reverse=True)
            fillers = ordered_unique[:needed]
            final_selected_ids.extend(fillers)

            filler_sources = []
            for fid in fillers:
                src_list = []
                if fid in naive_map: src_list.append("NAIVE")
                if fid in local_map: src_list.append("LOCAL")
                filler_sources.append(f"{fid[:8]}...({'+'.join(src_list)})")
            logger.info(f"Consensus: Added {len(fillers)} fillers: {', '.join(filler_sources)}")

        # 4. Fetch Chunk Content (PostgreSQL)
        t0 = time.perf_counter()
        final_results = []
        for cid in final_selected_ids:
            chunk_data = await self.rag.text_chunks.get_by_id(cid)
            if chunk_data:
                chunk_data['consensus_source'] = "intersection" if cid in intersection_ids else "unique"
                chunk_data['total_score'] = chunk_data_map[cid]
                final_results.append(chunk_data)
        ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[Consensus][TIMING] chunk_fetch={ms:.0f}ms (PostgreSQL x{len(final_selected_ids)} chunks)")

        # 5. Re-rank by Page Index
        def get_page_idx(chunk):
            meta = chunk.get('metadata', {})
            if isinstance(meta, dict) and 'page_idx' in meta:
                try:
                    return int(meta['page_idx'])
                except Exception:
                    pass
            return 999999

        final_results.sort(key=get_page_idx)
        logger.info(f"Consensus: Re-ranked {len(final_results)} chunks by page order.")

        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(f"[Consensus][TIMING] total={total_ms:.0f}ms")

        logger.info(f"Consensus: Returning {len(final_results)} chunks to LLM.")
        for i, res in enumerate(final_results[:3]):
            content_snippet = res.get('content', '')[:200].replace('\n', ' ')
            logger.info(f"   - Chunk {i+1} ({res.get('consensus_source')}): {content_snippet}...")

        return final_results