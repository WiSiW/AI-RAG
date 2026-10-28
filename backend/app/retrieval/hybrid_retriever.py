import asyncio
import logging
from collections import defaultdict

from app.indexing.bm25_store import BM25Store
from app.indexing.types import RetrievedChunk
from app.indexing.vector_store import MilvusVectorStore

logger = logging.getLogger(__name__)


def rrf_fuse(
    bm25_results: list[RetrievedChunk],
    vector_results: list[RetrievedChunk],
    *,
    rrf_k: int = 60,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    return weighted_rrf_fuse(
        [(bm25_results, 1.0), (vector_results, 1.0)],
        rrf_k=rrf_k,
        top_k=top_k,
    )


def weighted_rrf_fuse(
    ranked_lists: list[tuple[list[RetrievedChunk], float]],
    *,
    rrf_k: int = 60,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, RetrievedChunk] = {}
    for results, weight in ranked_lists:
        if weight <= 0:
            continue
        for rank, chunk in enumerate(results, 1):
            scores[chunk.chunk_id] += weight / (rrf_k + rank)
            existing = chunks.get(chunk.chunk_id)
            if existing is None:
                chunks[chunk.chunk_id] = chunk
            else:
                existing.metadata.update(chunk.metadata)
    ordered_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:top_k]
    fused: list[RetrievedChunk] = []
    for chunk_id in ordered_ids:
        chunk = chunks[chunk_id]
        chunk.rrf_score = round(scores[chunk_id], 8)
        fused.append(chunk)
    return fused


class HybridRetriever:
    def __init__(self, vector_store: MilvusVectorStore, bm25_store: BM25Store):
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    async def retrieve_routes(
        self,
        *,
        kb_id: str,
        query: str,
        query_vector: list[float] | None,
        dimensions: int,
        candidate_top_k: int = 20,
        document_id: str | None = None,
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        bm25_task = asyncio.to_thread(
            self.bm25_store.search, kb_id, query, candidate_top_k, document_id
        )
        tasks = [bm25_task]
        if query_vector is not None:
            tasks.append(
                asyncio.to_thread(
                    self.vector_store.search,
                    kb_id,
                    query_vector,
                    candidate_top_k,
                    dimensions,
                    document_id,
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        bm25_results = results[0]
        if isinstance(bm25_results, Exception):
            logger.warning("BM25 retrieval failed", exc_info=bm25_results)
            bm25_results = []
        vector_results = results[1] if query_vector is not None else []
        if isinstance(vector_results, Exception):
            logger.warning("vector retrieval failed", exc_info=vector_results)
            vector_results = []
        return bm25_results, vector_results

    async def retrieve(
        self,
        *,
        kb_id: str,
        query: str,
        query_vector: list[float],
        dimensions: int,
        candidate_top_k: int = 20,
        fusion_top_k: int = 20,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        bm25_results, vector_results = await self.retrieve_routes(
            kb_id=kb_id,
            query=query,
            query_vector=query_vector,
            dimensions=dimensions,
            candidate_top_k=candidate_top_k,
            document_id=document_id,
        )
        return rrf_fuse(
            bm25_results,
            vector_results,
            top_k=fusion_top_k,
        )


class Reranker:
    def __init__(self, gateway):
        self.gateway = gateway

    async def rerank(
        self,
        *,
        profile,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
        threshold: float = 0.0,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if profile is None:
            return chunks[:top_k]
        scores = await self.gateway.rerank(
            profile, query, [chunk.content for chunk in chunks]
        )
        for chunk, score in zip(chunks, scores, strict=False):
            chunk.rerank_score = round(float(score), 6)
        ranked = sorted(chunks, key=lambda item: item.rerank_score or 0, reverse=True)
        filtered = [
            chunk for chunk in ranked if (chunk.rerank_score or 0.0) >= threshold
        ]
        return filtered[:top_k]


