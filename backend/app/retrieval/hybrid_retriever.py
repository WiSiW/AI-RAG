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
        route_names=["bm25", "vector"],
    )


def weighted_rrf_fuse(
    ranked_lists: list[tuple[list[RetrievedChunk], float]],
    *,
    rrf_k: int = 60,
    top_k: int = 20,
    route_names: list[str] | None = None,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, RetrievedChunk] = {}
    for list_index, (results, weight) in enumerate(ranked_lists):
        if weight <= 0:
            continue
        route_name = (
            route_names[list_index]
            if route_names and list_index < len(route_names)
            else results[0].metadata.get("representation_type", f"route_{list_index}")
            if results
            else f"route_{list_index}"
        )
        for rank, chunk in enumerate(results, 1):
            scores[chunk.chunk_id] += weight / (rrf_k + rank)
            existing = chunks.get(chunk.chunk_id)
            if existing is None:
                target = chunk
                target.metadata = dict(chunk.metadata)
                chunks[chunk.chunk_id] = target
            else:
                target = existing

            previous_matched = list(target.metadata.get("matched_representations") or [])
            previous_scores = dict(target.metadata.get("representation_scores") or {})
            previous_ranks = dict(target.metadata.get("representation_ranks") or {})
            previous_vector_score = target.metadata.get("vector_score")
            previous_bm25_score = target.metadata.get("bm25_score")
            incoming_metadata = dict(chunk.metadata)
            incoming_matched = list(incoming_metadata.pop("matched_representations", []) or [])
            incoming_scores = dict(incoming_metadata.pop("representation_scores", {}) or {})
            incoming_ranks = dict(incoming_metadata.pop("representation_ranks", {}) or {})
            route_score = (
                chunk.metadata.get("bm25_score")
                if route_name == "bm25"
                else chunk.metadata.get("vector_score")
            )
            if existing is not None:
                target.metadata.update(incoming_metadata)

            matched = previous_matched
            for candidate in [*incoming_matched, route_name]:
                if candidate not in matched:
                    matched.append(candidate)
            if route_name not in matched:
                matched.append(route_name)
            target.metadata["matched_representations"] = matched
            route_scores = {**previous_scores, **incoming_scores}
            route_scores[route_name] = float(route_score or (1.0 / (rrf_k + rank)))
            target.metadata["representation_scores"] = route_scores
            route_ranks = {**previous_ranks, **incoming_ranks}
            route_ranks[route_name] = rank
            target.metadata["representation_ranks"] = route_ranks
            if route_name == "bm25":
                target.metadata["bm25_score"] = max(
                    float(previous_bm25_score or 0), float(route_score or 0)
                )
                target.metadata["bm25_rank"] = min(
                    int(target.metadata.get("bm25_rank") or rank), rank
                )
            else:
                target.metadata["vector_score"] = max(
                    float(previous_vector_score or 0), float(route_score or 0)
                )
                target.metadata["vector_rank"] = min(
                    int(target.metadata.get("vector_rank") or rank), rank
                )
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
        index_version: str = "v2",
    ) -> dict[str, list[RetrievedChunk]]:
        routes: dict[str, list[RetrievedChunk]] = {
            "bm25": [],
            "l1": [],
            "original": [],
            "summary": [],
            "questions": [],
            "l3": [],
        }
        tasks = [
            asyncio.to_thread(
                self.bm25_store.search,
                kb_id,
                query,
                candidate_top_k,
                document_id,
                index_version=index_version,
            )
        ]
        vector_routes = [("original", "original")]
        if index_version == "v2":
            vector_routes = [
                ("l1", "l1_summary"),
                ("original", "original"),
                ("summary", "summary"),
                ("questions", "questions"),
                ("l3", "l3_atomic"),
            ]
        if query_vector is not None:
            for route, representation_type in vector_routes:
                limit = candidate_top_k * 2 if route == "questions" else candidate_top_k
                tasks.append(
                    asyncio.to_thread(
                        self.vector_store.search,
                        kb_id,
                        query_vector,
                        limit,
                        dimensions,
                        document_id,
                        representation_type=(
                            representation_type if index_version == "v2" else None
                        ),
                        index_version=index_version,
                    )
                )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        bm25_results = results[0]
        if isinstance(bm25_results, Exception):
            logger.warning("BM25 retrieval failed", exc_info=bm25_results)
            bm25_results = []
        routes["bm25"] = bm25_results
        if query_vector is not None:
            for (route, _), result in zip(vector_routes, results[1:], strict=True):
                if isinstance(result, Exception):
                    logger.warning("%s vector retrieval failed", route, exc_info=result)
                    continue
                routes[route] = self._deduplicate_route(result, route)
        return routes

    @staticmethod
    def _deduplicate_route(
        results: list[RetrievedChunk], route_name: str
    ) -> list[RetrievedChunk]:
        unique: list[RetrievedChunk] = []
        seen: set[str] = set()
        for hit in results:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            hit.metadata["representation_type"] = route_name
            hit.metadata["vector_rank"] = len(unique) + 1
            hit.metadata["matched_representations"] = [route_name]
            hit.metadata["representation_scores"] = {route_name: float(hit.metadata.get("vector_score") or 0)}
            unique.append(hit)
        return unique

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
        index_version: str = "v2",
    ) -> list[RetrievedChunk]:
        routes = await self.retrieve_routes(
            kb_id=kb_id,
            query=query,
            query_vector=query_vector,
            dimensions=dimensions,
            candidate_top_k=candidate_top_k,
            document_id=document_id,
            index_version=index_version,
        )
        return rrf_fuse(
            routes["bm25"],
            [
                item
                for route in ("l1", "original", "summary", "questions", "l3")
                for item in routes[route]
            ],
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
        documents = [
            (
                f"{chunk.metadata.get('context_prefix')}\n{chunk.content}"
                if chunk.metadata.get("context_prefix")
                else chunk.content
            )
            for chunk in chunks
        ]
        scores = await self.gateway.rerank(
            profile, query, documents
        )
        for chunk, score in zip(chunks, scores, strict=False):
            chunk.rerank_score = round(float(score), 6)
        ranked = sorted(chunks, key=lambda item: item.rerank_score or 0, reverse=True)
        filtered = [
            chunk for chunk in ranked if (chunk.rerank_score or 0.0) >= threshold
        ]
        return filtered[:top_k]


