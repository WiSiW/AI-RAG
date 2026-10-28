import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.chunking.semantic_chunker import count_tokens
from app.core.config import settings
from app.core.errors import NotFoundError, ValidationError
from app.db.models import (
    Conversation,
    Document,
    DocumentChunk,
    DocumentStatus,
    KnowledgeBase,
    ModelKind,
    ModelProfile,
)
from app.db.session import SessionLocal
from app.generation.citation_parser import parse_citations
from app.generation.prompts import QUERY_REWRITE_PROMPT, SYSTEM_PROMPT, build_chat_prompt
from app.indexing.bm25_store import BM25Store
from app.indexing.model_gateway import ModelGateway
from app.indexing.types import RetrievedChunk
from app.indexing.vector_store import MilvusVectorStore
from app.memory.session_manager import ConversationMemory
from app.retrieval.hybrid_retriever import HybridRetriever, Reranker, weighted_rrf_fuse
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    SearchQualitySummary,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

NO_ANSWER = "根据现有资料无法回答该问题。"
CitationUrlBuilder = Callable[[str], tuple[str | None, str | None]]


@dataclass
class RetrievalOutcome:
    chunks: list[RetrievedChunk]
    failed_knowledge_base_ids: list[str] = field(default_factory=list)


def resolve_search_relevance(
    rerank_score: float | None, rrf_score: float
) -> tuple[float, str]:
    if rerank_score is not None:
        return rerank_score, "rerank"
    return rrf_score, "rrf"


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


logger = logging.getLogger(__name__)


async def summarize_conversation(
    db: Session, conversation_id: str, profile_id: str, memory: ConversationMemory
) -> None:
    try:
        conversation = db.get(Conversation, conversation_id)
        profile = db.get(ModelProfile, profile_id)
        if conversation and profile:
            await memory.maybe_summarize(db, conversation, profile)
    except Exception:
        logger.exception("conversation summary failed")


async def summarize_background(
    conversation_id: str, profile_id: str, memory: ConversationMemory
) -> None:
    db = SessionLocal()
    try:
        await summarize_conversation(db, conversation_id, profile_id, memory)
    finally:
        db.close()


class ChatService:
    def __init__(self):
        self.gateway = ModelGateway()
        self.vector_store = MilvusVectorStore()
        self.bm25_store = BM25Store()
        self.retriever = HybridRetriever(self.vector_store, self.bm25_store)
        self.reranker = Reranker(self.gateway)
        self.memory = ConversationMemory(self.gateway)

    @staticmethod
    def _get_kbs(db: Session, kb_ids: list[str]) -> list[KnowledgeBase]:
        knowledge_bases: list[KnowledgeBase] = []
        for kb_id in dict.fromkeys(kb_ids):
            knowledge_base = db.get(KnowledgeBase, kb_id)
            if knowledge_base is None:
                raise NotFoundError(f"知识库不存在：{kb_id}")
            knowledge_bases.append(knowledge_base)
        if not knowledge_bases:
            raise ValidationError("至少选择一个知识库")
        return knowledge_bases

    @staticmethod
    def _validate_document(
        db: Session, knowledge_bases: list[KnowledgeBase], document_id: str | None
    ) -> Document | None:
        if not document_id:
            return None
        if len(knowledge_bases) != 1:
            raise ValidationError("多知识库检索时不支持仅限定单个文档")
        knowledge_base = knowledge_bases[0]
        document = db.get(Document, document_id)
        if document is None or document.kb_id != knowledge_base.id:
            raise NotFoundError("文档不属于当前知识库")
        if document.status != DocumentStatus.READY:
            raise ValidationError("文档尚未完成入库")
        return document

    @staticmethod
    def _profile(db: Session, profile_id: str | None, expected_kind: str) -> ModelProfile:
        if not profile_id:
            raise ValidationError(f"未配置 {expected_kind} 模型")
        profile = db.get(ModelProfile, profile_id)
        if profile is None or profile.kind != expected_kind or not profile.is_active:
            raise ValidationError(f"{expected_kind} 模型不可用")
        return profile

    def _llm_profile(
        self, db: Session, knowledge_bases: list[KnowledgeBase], override_id: str | None
    ) -> ModelProfile:
        if override_id:
            return self._profile(db, override_id, ModelKind.LLM)
        default_ids = {
            knowledge_base.default_llm_profile_id
            for knowledge_base in knowledge_bases
            if knowledge_base.default_llm_profile_id
        }
        if len(default_ids) != 1:
            message = (
                "多个知识库默认问答模型不一致，请选择问答模型"
                if default_ids
                else "未配置 LLM 模型"
            )
            raise ValidationError(message)
        return self._profile(db, next(iter(default_ids)), ModelKind.LLM)

    def _rerank_profile(
        self,
        db: Session,
        knowledge_bases: list[KnowledgeBase],
        override_id: str | None,
    ) -> ModelProfile | None:
        if override_id:
            return self._profile(db, override_id, ModelKind.RERANK)
        rerank_ids = {kb.rerank_profile_id for kb in knowledge_bases}
        if len(rerank_ids) == 1 and None not in rerank_ids:
            return self._profile(db, next(iter(rerank_ids)), ModelKind.RERANK)
        return None

    async def _rewrite_query(
        self,
        *,
        query: str,
        history: str,
        llm_profile: ModelProfile,
        scope: str = "",
    ) -> str:
        if not history.strip():
            return query
        prompt = QUERY_REWRITE_PROMPT.format(
            scope=scope or "未指定",
            history=history,
            question=query,
        )
        try:
            rewritten = await self.gateway.complete(
                llm_profile,
                [{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=settings.query_rewrite_max_tokens,
            )
            return rewritten.strip().strip('"').strip("'") or query
        except Exception:
            logger.warning("query rewrite failed; using original query", exc_info=True)
            return query

    async def _retrieve(
        self,
        db: Session,
        knowledge_bases: list[KnowledgeBase],
        query: str,
        *,
        top_k: int | None = None,
        rerank: bool = True,
        rerank_profile_id: str | None = None,
        document_id: str | None = None,
        enforce_context_budget: bool = True,
    ) -> RetrievalOutcome:
        embedding_profiles: dict[str, ModelProfile] = {}
        for knowledge_base in knowledge_bases:
            profile = self._profile(db, knowledge_base.embedding_profile_id, ModelKind.EMBEDDING)
            if not profile.dimensions:
                raise ValidationError("Embedding 模型必须配置 dimensions")
            embedding_profiles[profile.id] = profile

        profile_ids = list(embedding_profiles)
        embedding_results = await asyncio.gather(
            *[self.gateway.embed_texts(embedding_profiles[profile_id], [query]) for profile_id in profile_ids],
            return_exceptions=True,
        )
        query_vectors: dict[str, list[float]] = {}
        for profile_id, result in zip(profile_ids, embedding_results, strict=True):
            if isinstance(result, Exception):
                logger.warning("embedding query failed for profile %s", profile_id, exc_info=result)
                continue
            if result:
                query_vectors[profile_id] = result[0]

        semaphore = asyncio.Semaphore(settings.rag_max_concurrency)

        async def retrieve_one(knowledge_base: KnowledgeBase):
            async with semaphore:
                kb_settings = knowledge_base.settings or {}
                return await self.retriever.retrieve_routes(
                    kb_id=knowledge_base.id,
                    query=query,
                    query_vector=query_vectors.get(knowledge_base.embedding_profile_id),
                    dimensions=embedding_profiles[knowledge_base.embedding_profile_id].dimensions or 0,
                    candidate_top_k=int(
                        kb_settings.get("candidate_top_k", settings.rag_candidate_top_k)
                    ),
                    document_id=document_id,
                )

        route_results = await asyncio.gather(
            *[retrieve_one(knowledge_base) for knowledge_base in knowledge_bases],
            return_exceptions=True,
        )
        ranked_lists: list[tuple[list[RetrievedChunk], float]] = []
        failed_knowledge_base_ids: list[str] = []
        for knowledge_base, result in zip(knowledge_bases, route_results, strict=True):
            if isinstance(result, Exception):
                logger.warning(
                    "knowledge base retrieval failed: %s",
                    knowledge_base.id,
                    exc_info=result,
                )
                failed_knowledge_base_ids.append(knowledge_base.id)
                continue
            bm25_results, vector_results = result
            weight = float((knowledge_base.settings or {}).get("fusion_weight", 1.0))
            if bm25_results:
                ranked_lists.append((bm25_results, weight))
            if vector_results:
                ranked_lists.append((vector_results, weight))

        final_limit = top_k or (
            settings.rag_final_top_k
            if len(knowledge_bases) == 1
            else settings.rag_multi_kb_final_top_k
        )
        candidates = weighted_rrf_fuse(
            ranked_lists,
            rrf_k=60,
            top_k=max(settings.rag_global_candidate_top_k, final_limit * max(len(knowledge_bases), 1)),
        )
        kb_names = {kb.id: kb.name for kb in knowledge_bases}
        for candidate in candidates:
            candidate.metadata["kb_name"] = kb_names.get(candidate.kb_id, "")
        candidates = [candidate for candidate in self._hydrate_chunks(db, candidates) if candidate.kb_id]

        if rerank:
            rerank_profile = self._rerank_profile(db, knowledge_bases, rerank_profile_id)
            threshold = (
                float((knowledge_bases[0].settings or {}).get("rerank_threshold", 0.3))
                if len(knowledge_bases) == 1
                else settings.rag_multi_kb_rerank_threshold
            )
            result = await self.reranker.rerank(
                profile=rerank_profile,
                query=query,
                chunks=candidates,
                top_k=final_limit,
                threshold=threshold,
            )
        else:
            result = candidates[:final_limit]
        if enforce_context_budget:
            result = self._limit_context_tokens(result)
        return RetrievalOutcome(
            chunks=result,
            failed_knowledge_base_ids=failed_knowledge_base_ids,
        )

    @staticmethod
    def _limit_context_tokens(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        selected: list[RetrievedChunk] = []
        total = 0
        for chunk in chunks:
            token_count = count_tokens(chunk.content)
            if selected and total + token_count > settings.rag_multi_kb_max_context_tokens:
                continue
            selected.append(chunk)
            total += token_count
            if total >= settings.rag_multi_kb_max_context_tokens:
                break
        return selected

    @staticmethod
    def _hydrate_chunks(db: Session, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        ids = [chunk.chunk_id for chunk in chunks]
        rows = {
            row.id: row
            for row in db.scalars(select(DocumentChunk).where(DocumentChunk.id.in_(ids)))
        }
        for chunk in chunks:
            row = rows.get(chunk.chunk_id)
            if row is None:
                continue
            db_metadata = dict(row.metadata_json or {})
            chunk.section = row.section_path
            chunk.metadata = {**db_metadata, **chunk.metadata}
            parent_content = db_metadata.get("parent_content")
            if parent_content:
                chunk.metadata["child_content"] = row.content
                chunk.content = str(parent_content)
        return chunks

    @staticmethod
    def _context_payload(chunks: list[RetrievedChunk]) -> list[dict]:
        return [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "knowledge_base_id": chunk.kb_id,
                "knowledge_base_name": chunk.metadata.get("kb_name", ""),
                "source": chunk.source,
                "section": chunk.section,
                "content": chunk.content,
                "rrf_score": chunk.rrf_score,
                "rerank_score": chunk.rerank_score,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

    @staticmethod
    def _usage(answer: str, chunks: list[RetrievedChunk], profile: ModelProfile) -> dict:
        return {
            "model": profile.model_name,
            "context_chunks": len(chunks),
            "context_tokens_estimate": sum(count_tokens(chunk.content) for chunk in chunks),
            "answer_tokens_estimate": count_tokens(answer),
        }

    @staticmethod
    def _attach_citation_urls(
        citations: list[Citation], builder: CitationUrlBuilder | None
    ) -> list[Citation]:
        if builder is None:
            return citations
        for citation in citations:
            citation.file_url, citation.preview_url = builder(citation.document_id)
        return citations

    @staticmethod
    def _quality(
        results: list[SearchResult],
        queried_knowledge_base_ids: list[str],
        failed_knowledge_base_ids: list[str],
    ) -> SearchQualitySummary:
        hits_by_knowledge_base: dict[str, int] = {}
        for result in results:
            hits_by_knowledge_base[result.knowledge_base_id] = (
                hits_by_knowledge_base.get(result.knowledge_base_id, 0) + 1
            )
        if not results:
            return SearchQualitySummary(
                queried_knowledge_base_ids=queried_knowledge_base_ids,
                failed_knowledge_base_ids=failed_knowledge_base_ids,
                hits_by_knowledge_base=hits_by_knowledge_base,
            )
        scores = [
            float(item.rerank_score if item.rerank_score is not None else item.rrf_score)
            for item in results
        ]
        return SearchQualitySummary(
            result_count=len(results),
            avg_rrf_score=round(sum(item.rrf_score for item in results) / len(results), 8),
            best_score=round(max(scores), 8),
            reranked_count=sum(1 for item in results if item.rerank_score is not None),
            vector_hits=sum(1 for item in results if item.vector_rank is not None),
            bm25_hits=sum(1 for item in results if item.bm25_rank is not None),
            queried_knowledge_base_ids=queried_knowledge_base_ids,
            failed_knowledge_base_ids=failed_knowledge_base_ids,
            hits_by_knowledge_base=hits_by_knowledge_base,
        )

    async def search(
        self,
        db: Session,
        request: SearchRequest,
        *,
        file_url_builder: CitationUrlBuilder | None = None,
    ) -> SearchResponse:
        knowledge_bases = self._get_kbs(db, request.knowledge_base_ids)
        self._validate_document(db, knowledge_bases, request.document_id)
        rewritten_query = request.query
        if request.rewrite and request.conversation_id:
            llm_profile = self._llm_profile(db, knowledge_bases, request.llm_profile_id)
            conversation = self.memory.get_or_create(
                db,
                kb_ids=request.knowledge_base_ids,
                conversation_id=request.conversation_id,
                llm_profile_id=llm_profile.id,
                first_query=request.query,
                document_id=request.document_id,
            )
            history = self.memory.history_text(db, conversation)
            rewritten_query = await self._rewrite_query(
                query=request.query,
                history=history,
                llm_profile=llm_profile,
                scope="、".join(knowledge_base.name for knowledge_base in knowledge_bases),
            )
        outcome = await self._retrieve(
            db,
            knowledge_bases,
            rewritten_query,
            top_k=request.top_k,
            rerank=request.rerank,
            rerank_profile_id=request.rerank_profile_id,
            document_id=request.document_id,
            enforce_context_budget=False,
        )
        chunks = outcome.chunks
        results: list[SearchResult] = []
        for chunk in chunks:
            relevance_score, relevance_source = resolve_search_relevance(
                chunk.rerank_score, chunk.rrf_score
            )
            file_url = None
            preview_url = None
            if file_url_builder:
                file_url, preview_url = file_url_builder(chunk.document_id)
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    knowledge_base_id=chunk.kb_id,
                    knowledge_base_name=str(chunk.metadata.get("kb_name", "")),
                    source=chunk.source,
                    section=chunk.section,
                    content=chunk.content,
                    rrf_score=chunk.rrf_score,
                    rerank_score=chunk.rerank_score,
                    relevance_score=relevance_score,
                    relevance_source=relevance_source,
                    vector_score=chunk.metadata.get("vector_score"),
                    bm25_score=chunk.metadata.get("bm25_score"),
                    vector_rank=chunk.metadata.get("vector_rank"),
                    bm25_rank=chunk.metadata.get("bm25_rank"),
                    metadata=chunk.metadata,
                    file_url=file_url,
                    preview_url=preview_url,
                )
            )
        return SearchResponse(
            query=request.query,
            rewritten_query=rewritten_query if rewritten_query != request.query else None,
            results=results,
            quality=self._quality(
                results,
                request.knowledge_base_ids,
                outcome.failed_knowledge_base_ids,
            ),
        )

    async def chat(
        self,
        db: Session,
        request: ChatRequest,
        *,
        citation_url_builder: CitationUrlBuilder | None = None,
    ) -> ChatResponse:
        knowledge_bases = self._get_kbs(db, request.knowledge_base_ids)
        self._validate_document(db, knowledge_bases, request.document_id)
        llm_profile = self._llm_profile(db, knowledge_bases, request.llm_profile_id)
        conversation = self.memory.get_or_create(
            db,
            kb_ids=request.knowledge_base_ids,
            conversation_id=request.conversation_id,
            llm_profile_id=llm_profile.id,
            first_query=request.query,
            document_id=request.document_id,
        )
        effective_document_id = request.document_id or conversation.document_id
        self._validate_document(db, knowledge_bases, effective_document_id)
        history = self.memory.history_text(db, conversation)
        rewritten_query = await self._rewrite_query(
            query=request.query,
            history=history,
            llm_profile=llm_profile,
            scope="、".join(knowledge_base.name for knowledge_base in knowledge_bases),
        )
        outcome = await self._retrieve(
            db,
            knowledge_bases,
            rewritten_query,
            rerank_profile_id=request.rerank_profile_id,
            document_id=effective_document_id,
        )
        chunks = outcome.chunks
        if chunks:
            prompt = build_chat_prompt(request.query, chunks, history)
            answer = await self.gateway.complete(
                llm_profile,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        else:
            answer = NO_ANSWER
        citations = self._attach_citation_urls(
            parse_citations(answer, chunks), citation_url_builder
        )
        usage = self._usage(answer, chunks, llm_profile)
        usage["failed_knowledge_base_ids"] = outcome.failed_knowledge_base_ids
        message = self.memory.add_exchange(
            db,
            conversation,
            question=request.query,
            answer=answer,
            rewritten_query=rewritten_query,
            citations=citations,
            context=self._context_payload(chunks),
            usage=usage,
        )
        await summarize_conversation(db, conversation.id, llm_profile.id, self.memory)
        return ChatResponse(
            conversation_id=conversation.id,
            message_id=message.id,
            knowledge_base_ids=request.knowledge_base_ids,
            answer=answer,
            citations=citations,
            rewritten_query=rewritten_query,
            usage=usage,
        )

    async def stream_chat(
        self,
        db: Session,
        request: ChatRequest,
        *,
        citation_url_builder: CitationUrlBuilder | None = None,
    ):
        knowledge_bases = self._get_kbs(db, request.knowledge_base_ids)
        self._validate_document(db, knowledge_bases, request.document_id)
        llm_profile = self._llm_profile(db, knowledge_bases, request.llm_profile_id)
        conversation = self.memory.get_or_create(
            db,
            kb_ids=request.knowledge_base_ids,
            conversation_id=request.conversation_id,
            llm_profile_id=llm_profile.id,
            first_query=request.query,
            document_id=request.document_id,
        )
        effective_document_id = request.document_id or conversation.document_id
        self._validate_document(db, knowledge_bases, effective_document_id)
        history = self.memory.history_text(db, conversation)
        rewritten_query = await self._rewrite_query(
            query=request.query,
            history=history,
            llm_profile=llm_profile,
            scope="、".join(knowledge_base.name for knowledge_base in knowledge_bases),
        )
        outcome = await self._retrieve(
            db,
            knowledge_bases,
            rewritten_query,
            rerank_profile_id=request.rerank_profile_id,
            document_id=effective_document_id,
        )
        chunks = outcome.chunks
        yield sse_event(
            "start",
            {
                "conversation_id": conversation.id,
                "document_id": conversation.document_id,
                "knowledge_base_ids": request.knowledge_base_ids,
                "knowledge_bases": [
                    {"id": knowledge_base.id, "name": knowledge_base.name}
                    for knowledge_base in knowledge_bases
                ],
                "rewritten_query": rewritten_query,
                "context_count": len(chunks),
                "failed_knowledge_base_ids": outcome.failed_knowledge_base_ids,
            },
        )
        answer_parts: list[str] = []
        try:
            if chunks:
                prompt = build_chat_prompt(request.query, chunks, history)
                async for token in self.gateway.stream(
                    llm_profile,
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                ):
                    answer_parts.append(token)
                    yield sse_event("token", {"content": token})
            else:
                answer_parts.append(NO_ANSWER)
                yield sse_event("token", {"content": NO_ANSWER})
            answer = "".join(answer_parts)
            citations = self._attach_citation_urls(
                parse_citations(answer, chunks), citation_url_builder
            )
            usage = self._usage(answer, chunks, llm_profile)
            usage["failed_knowledge_base_ids"] = outcome.failed_knowledge_base_ids
            message = self.memory.add_exchange(
                db,
                conversation,
                question=request.query,
                answer=answer,
                rewritten_query=rewritten_query,
                citations=citations,
                context=self._context_payload(chunks),
                usage=usage,
            )
            yield sse_event(
                "citations",
                {"citations": [citation.model_dump() for citation in citations]},
            )
            yield sse_event(
                "done",
                {
                    "conversation_id": conversation.id,
                    "message_id": message.id,
                    "usage": usage,
                },
            )
            asyncio.create_task(
                summarize_background(conversation.id, llm_profile.id, self.memory)
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("streaming chat failed")
            yield sse_event("error", {"message": str(exc)})




