import asyncio
import json
import logging
from collections.abc import Callable

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
from app.retrieval.hybrid_retriever import HybridRetriever, Reranker
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
    def _get_kb(db: Session, kb_id: str) -> KnowledgeBase:
        knowledge_base = db.get(KnowledgeBase, kb_id)
        if knowledge_base is None:
            raise NotFoundError("知识库不存在")
        return knowledge_base

    @staticmethod
    def _validate_document(
        db: Session, knowledge_base: KnowledgeBase, document_id: str | None
    ) -> Document | None:
        if not document_id:
            return None
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
        self, db: Session, knowledge_base: KnowledgeBase, override_id: str | None
    ) -> ModelProfile:
        return self._profile(
            db,
            override_id or knowledge_base.default_llm_profile_id,
            ModelKind.LLM,
        )

    async def _rewrite_query(
        self,
        *,
        query: str,
        history: str,
        llm_profile: ModelProfile,
    ) -> str:
        if not history.strip():
            return query
        prompt = QUERY_REWRITE_PROMPT.format(history=history, question=query)
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
        knowledge_base: KnowledgeBase,
        query: str,
        *,
        top_k: int | None = None,
        rerank: bool = True,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        embedding_profile = self._profile(
            db, knowledge_base.embedding_profile_id, ModelKind.EMBEDDING
        )
        if not embedding_profile.dimensions:
            raise ValidationError("Embedding 模型必须配置 dimensions")
        query_vectors = await self.gateway.embed_texts(embedding_profile, [query])
        if not query_vectors:
            raise ValidationError("Embedding 服务未返回查询向量")
        kb_settings = knowledge_base.settings or {}
        candidates = await self.retriever.retrieve(
            kb_id=knowledge_base.id,
            query=query,
            query_vector=query_vectors[0],
            dimensions=embedding_profile.dimensions,
            candidate_top_k=int(kb_settings.get("candidate_top_k", settings.rag_candidate_top_k)),
            fusion_top_k=int(kb_settings.get("candidate_top_k", settings.rag_candidate_top_k)),
            document_id=document_id,
        )
        if not rerank:
            result = candidates[:top_k] if top_k else candidates
        else:
            rerank_profile = (
                self._profile(db, knowledge_base.rerank_profile_id, ModelKind.RERANK)
                if knowledge_base.rerank_profile_id
                else None
            )
            result = await self.reranker.rerank(
                profile=rerank_profile,
                query=query,
                chunks=candidates,
                top_k=top_k or int(kb_settings.get("final_top_k", settings.rag_final_top_k)),
                threshold=float(kb_settings.get("rerank_threshold", 0.3)),
            )
        return self._hydrate_chunks(db, result)

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
    def _quality(results: list[SearchResult]) -> SearchQualitySummary:
        if not results:
            return SearchQualitySummary()
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
        )

    async def search(
        self,
        db: Session,
        request: SearchRequest,
        *,
        file_url_builder: CitationUrlBuilder | None = None,
    ) -> SearchResponse:
        knowledge_base = self._get_kb(db, request.knowledge_base_id)
        self._validate_document(db, knowledge_base, request.document_id)
        rewritten_query = request.query
        if request.rewrite and request.conversation_id:
            conversation = self.memory.get_or_create(
                db,
                kb_id=knowledge_base.id,
                conversation_id=request.conversation_id,
                llm_profile_id=request.llm_profile_id
                or knowledge_base.default_llm_profile_id
                or "",
                first_query=request.query,
                document_id=request.document_id,
            )
            llm_profile = self._llm_profile(db, knowledge_base, request.llm_profile_id)
            history = self.memory.history_text(db, conversation)
            rewritten_query = await self._rewrite_query(
                query=request.query, history=history, llm_profile=llm_profile
            )
        chunks = await self._retrieve(
            db,
            knowledge_base,
            rewritten_query,
            top_k=request.top_k,
            rerank=request.rerank,
            document_id=request.document_id,
        )
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
            quality=self._quality(results),
        )

    async def chat(
        self,
        db: Session,
        request: ChatRequest,
        *,
        citation_url_builder: CitationUrlBuilder | None = None,
    ) -> ChatResponse:
        knowledge_base = self._get_kb(db, request.knowledge_base_id)
        self._validate_document(db, knowledge_base, request.document_id)
        llm_profile = self._llm_profile(db, knowledge_base, request.llm_profile_id)
        conversation = self.memory.get_or_create(
            db,
            kb_id=knowledge_base.id,
            conversation_id=request.conversation_id,
            llm_profile_id=llm_profile.id,
            first_query=request.query,
            document_id=request.document_id,
        )
        effective_document_id = request.document_id or conversation.document_id
        self._validate_document(db, knowledge_base, effective_document_id)
        history = self.memory.history_text(db, conversation)
        rewritten_query = await self._rewrite_query(
            query=request.query, history=history, llm_profile=llm_profile
        )
        chunks = await self._retrieve(
            db, knowledge_base, rewritten_query, document_id=effective_document_id
        )
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
        knowledge_base = self._get_kb(db, request.knowledge_base_id)
        self._validate_document(db, knowledge_base, request.document_id)
        llm_profile = self._llm_profile(db, knowledge_base, request.llm_profile_id)
        conversation = self.memory.get_or_create(
            db,
            kb_id=knowledge_base.id,
            conversation_id=request.conversation_id,
            llm_profile_id=llm_profile.id,
            first_query=request.query,
            document_id=request.document_id,
        )
        effective_document_id = request.document_id or conversation.document_id
        self._validate_document(db, knowledge_base, effective_document_id)
        history = self.memory.history_text(db, conversation)
        rewritten_query = await self._rewrite_query(
            query=request.query, history=history, llm_profile=llm_profile
        )
        chunks = await self._retrieve(
            db, knowledge_base, rewritten_query, document_id=effective_document_id
        )
        yield sse_event(
            "start",
            {
                "conversation_id": conversation.id,
                "document_id": conversation.document_id,
                "rewritten_query": rewritten_query,
                "context_count": len(chunks),
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




