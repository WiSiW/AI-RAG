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
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, RetrievedChunk] = {}
    for rank, chunk in enumerate(bm25_results, 1):
        scores[chunk.chunk_id] += 1.0 / (rrf_k + rank)
        chunk.metadata["bm25_rank"] = rank
        chunks[chunk.chunk_id] = chunk
    for rank, chunk in enumerate(vector_results, 1):
        scores[chunk.chunk_id] += 1.0 / (rrf_k + rank)
        chunk.metadata["vector_rank"] = rank
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
        bm25_task = asyncio.to_thread(
            self.bm25_store.search, kb_id, query, candidate_top_k, document_id
        )
        vector_task = asyncio.to_thread(
            self.vector_store.search,
            kb_id,
            query_vector,
            candidate_top_k,
            dimensions,
            document_id,
        )
        bm25_results, vector_results = await asyncio.gather(
            bm25_task, vector_task, return_exceptions=True
        )
        if isinstance(bm25_results, Exception):
            logger.warning("BM25 retrieval failed", exc_info=bm25_results)
            bm25_results = []
        if isinstance(vector_results, Exception):
            logger.warning("vector retrieval failed", exc_info=vector_results)
            vector_results = []
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


