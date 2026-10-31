import asyncio
import json
import logging
import re
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
    DocumentTable,
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
from app.retrieval.complexity import BERTComplexityClient, classify_complexity
from app.retrieval.context_window import (
    clip_around_anchor,
    clip_head_tokens,
    clip_tail_tokens,
)
from app.retrieval.hybrid_retriever import HybridRetriever, Reranker, weighted_rrf_fuse
from app.retrieval.multihop_planner import MultiHopPlanner
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    ComplexityDecision,
    MultiHopStep,
    RetrievalTrace,
    SearchQualitySummary,
    SearchRequest,
    SearchResponse,
    SearchResult,
    TableContextRead,
    TraceChunkSnapshot,
)

NO_ANSWER = "根据现有资料无法回答该问题。"
CitationUrlBuilder = Callable[[str], tuple[str | None, str | None]]


@dataclass
class RetrievalOutcome:
    chunks: list[RetrievedChunk]
    failed_knowledge_base_ids: list[str] = field(default_factory=list)
    trace: RetrievalTrace | None = None


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
        self.complexity_client = BERTComplexityClient()
        self.multihop_planner = MultiHopPlanner(self.gateway)

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

    @staticmethod
    def _planner_profile(
        db: Session, knowledge_bases: list[KnowledgeBase]
    ) -> ModelProfile | None:
        for knowledge_base in knowledge_bases:
            for profile_id in (
                getattr(knowledge_base, "enrichment_profile_id", None),
                getattr(knowledge_base, "default_llm_profile_id", None),
            ):
                if not profile_id:
                    continue
                profile = db.get(ModelProfile, profile_id)
                if (
                    profile is not None
                    and profile.is_active
                    and profile.kind in {ModelKind.ENRICHMENT, ModelKind.LLM}
                ):
                    return profile
        return None

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
        expand_context: bool = False,
        retrieval_mode: str = "single",
    ) -> RetrievalOutcome:
        planner_profile = self._planner_profile(db, knowledge_bases)
        decision = await classify_complexity(
            query,
            retrieval_mode=retrieval_mode,
            llm_profile=planner_profile,
            gateway=self.gateway,
            classifier_client=self.complexity_client,
        )
        if not decision.is_multi_hop:
            outcome = await self._retrieve_single(
                db,
                knowledge_bases,
                query,
                top_k=top_k,
                rerank=rerank,
                rerank_profile_id=rerank_profile_id,
                document_id=document_id,
                enforce_context_budget=enforce_context_budget,
                expand_context=expand_context,
            )
            outcome.trace = RetrievalTrace(
                requested_mode=retrieval_mode,
                complexity=decision,
                stop_reason="single_hop",
                total_rounds=1,
                total_chunks=len(outcome.chunks),
            )
            return outcome
        return await self._retrieve_multi_hop(
            db,
            knowledge_bases,
            query,
            decision=decision,
            planner_profile=planner_profile,
            top_k=top_k,
            rerank=rerank,
            rerank_profile_id=rerank_profile_id,
            document_id=document_id,
            enforce_context_budget=enforce_context_budget,
            expand_context=expand_context,
        )

    async def _retrieve_multi_hop(
        self,
        db: Session,
        knowledge_bases: list[KnowledgeBase],
        query: str,
        *,
        decision: ComplexityDecision,
        planner_profile: ModelProfile | None,
        top_k: int | None,
        rerank: bool,
        rerank_profile_id: str | None,
        document_id: str | None,
        enforce_context_budget: bool,
        expand_context: bool,
    ) -> RetrievalOutcome:
        kb_settings = knowledge_bases[0].settings or {}
        max_steps = max(
            2,
            min(5, int(kb_settings.get("max_multi_hop_steps", settings.multihop_max_steps))),
        )
        round_top_k = max(
            1,
            int(kb_settings.get("multihop_round_top_k", settings.multihop_round_top_k)),
        )
        accumulated: dict[str, RetrievedChunk] = {}
        steps: list[MultiHopStep] = []
        failed_ids: set[str] = set()
        prior_questions = [query]
        prior_entities: list[str] = []
        current_query = query
        stop_reason = ""
        uncertain = False

        for round_number in range(1, max_steps + 1):
            round_outcome = await self._retrieve_single(
                db,
                knowledge_bases,
                current_query,
                top_k=round_top_k,
                rerank=False,
                document_id=document_id,
                enforce_context_budget=False,
                expand_context=False,
            )
            failed_ids.update(round_outcome.failed_knowledge_base_ids)
            new_count = 0
            for chunk in round_outcome.chunks:
                if chunk.chunk_id not in accumulated:
                    accumulated[chunk.chunk_id] = chunk
                    new_count += 1
            if not accumulated:
                stop_reason = "no_evidence"
                uncertain = True
                steps.append(
                    self._trace_step(
                        round_number=round_number,
                        query=current_query,
                        chunks=[],
                        new_chunk_count=0,
                        limit=round_top_k,
                        need_more=False,
                        reason="未检索到证据",
                    )
                )
                break
            plan = await self.multihop_planner.plan(
                profile=planner_profile,
                original_question=query,
                current_question=current_query,
                round_number=round_number,
                evidence=sorted(
                    accumulated.values(),
                    key=lambda item: item.rrf_score,
                    reverse=True,
                ),
                prior_questions=prior_questions,
                prior_entities=prior_entities,
            )
            if plan is None:
                stop_reason = "planner_unavailable"
                uncertain = True
                steps.append(
                    self._trace_step(
                        round_number=round_number,
                        query=current_query,
                        chunks=round_outcome.chunks,
                        new_chunk_count=new_count,
                        limit=round_top_k,
                        need_more=True,
                        reason="规划模型调用失败",
                    )
                )
                break
            for entity in plan.key_entities:
                if entity not in prior_entities:
                    prior_entities.append(entity)
            steps.append(
                self._trace_step(
                    round_number=round_number,
                    query=current_query,
                    chunks=round_outcome.chunks,
                    new_chunk_count=new_count,
                    limit=round_top_k,
                    key_entities=plan.key_entities,
                    need_more=plan.need_more,
                    confidence=plan.confidence,
                    reason=plan.reason,
                )
            )
            if not plan.need_more:
                stop_reason = "planner_sufficient"
                break
            if new_count == 0:
                stop_reason = "no_new_evidence"
                uncertain = True
                break
            if round_number >= max_steps:
                stop_reason = "max_steps"
                uncertain = True
                break
            next_query = self._normalize_query(plan.next_question)
            if not next_query or next_query in {self._normalize_query(item) for item in prior_questions}:
                stop_reason = "duplicate_or_empty_subquestion"
                uncertain = True
                break
            if self._evidence_tokens(accumulated.values()) >= settings.rag_multi_kb_max_context_tokens:
                stop_reason = "context_budget"
                uncertain = True
                break
            current_query = plan.next_question
            prior_questions.append(current_query)

        candidates = sorted(
            accumulated.values(), key=lambda item: item.rrf_score, reverse=True
        )
        final_limit = top_k or (
            settings.rag_final_top_k
            if len(knowledge_bases) == 1
            else settings.rag_multi_kb_final_top_k
        )
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
        if expand_context:
            result = self._expand_context_windows(db, result)
        if enforce_context_budget:
            result = self._limit_context_tokens(result)
        if not result:
            uncertain = True
        notice = (
            "注意：该问题需要多步推理，但当前资料可能不足以完整回答，以下结论仅供参考。"
            if uncertain
            else ""
        )
        trace = RetrievalTrace(
            requested_mode="auto" if decision.decision_source != "user_override" else "multi",
            complexity=decision,
            steps=steps,
            stop_reason=stop_reason or "completed",
            uncertain=uncertain,
            uncertainty_notice=notice,
            total_rounds=len(steps),
            total_chunks=len(accumulated),
        )
        return RetrievalOutcome(
            chunks=result,
            failed_knowledge_base_ids=sorted(failed_ids),
            trace=trace,
        )

    @staticmethod
    def _trace_score(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _trace_chunk_snapshots(
        cls,
        chunks: list[RetrievedChunk],
        *,
        limit: int,
    ) -> list[TraceChunkSnapshot]:
        unique_chunks: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            unique_chunks.setdefault(chunk.chunk_id, chunk)
        snapshots: list[TraceChunkSnapshot] = []
        for chunk in list(unique_chunks.values())[:limit]:
            metadata = chunk.metadata or {}
            matched_representations = list(
                dict.fromkeys(
                    str(route)
                    for route in (metadata.get("matched_representations") or [])
                    if route
                )
            )
            snapshots.append(
                TraceChunkSnapshot(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source=chunk.source,
                    section=chunk.section,
                    granularity_level=str(metadata.get("granularity_level") or "L2"),
                    granule_type=str(metadata.get("granule_type") or "paragraph"),
                    rrf_score=float(chunk.rrf_score or 0),
                    vector_score=cls._trace_score(metadata.get("vector_score")),
                    bm25_score=cls._trace_score(metadata.get("bm25_score")),
                    matched_representations=matched_representations,
                    content_snippet=(chunk.context_text or chunk.content or "").strip()[:240],
                )
            )
        return snapshots

    @classmethod
    def _trace_step(
        cls,
        *,
        round_number: int,
        query: str,
        chunks: list[RetrievedChunk],
        new_chunk_count: int,
        limit: int,
        key_entities: list[str] | None = None,
        need_more: bool = False,
        confidence: float | None = None,
        reason: str = "",
    ) -> MultiHopStep:
        snapshots = cls._trace_chunk_snapshots(chunks, limit=limit)
        route_counts: dict[str, int] = {}
        granularity_counts = {"L1": 0, "L2": 0, "L3": 0}
        for snapshot in snapshots:
            for route in snapshot.matched_representations:
                route_counts[route] = route_counts.get(route, 0) + 1
            if snapshot.granularity_level in granularity_counts:
                granularity_counts[snapshot.granularity_level] += 1
        return MultiHopStep(
            round=round_number,
            query=query,
            chunk_ids=list(dict.fromkeys(item.chunk_id for item in chunks)),
            chunks=snapshots,
            route_counts=route_counts,
            granularity_counts=granularity_counts,
            new_chunk_count=new_chunk_count,
            key_entities=key_entities or [],
            need_more=need_more,
            confidence=confidence,
            reason=reason,
        )

    @staticmethod
    def _normalize_query(query: str) -> str:
        return re.sub(r"[\s，,。.！!？?；;：:]+", "", query or "").casefold()

    @staticmethod
    def _evidence_tokens(chunks) -> int:
        return sum(count_tokens(chunk.context_text or chunk.content) for chunk in chunks)

    async def _retrieve_single(
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
        expand_context: bool = False,
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
                index_version = str(kb_settings.get("index_version", "v1"))
                if index_version not in {"v1", "v2"}:
                    index_version = "v1"
                return await self.retriever.retrieve_routes(
                    kb_id=knowledge_base.id,
                    query=query,
                    query_vector=query_vectors.get(knowledge_base.embedding_profile_id),
                    dimensions=embedding_profiles[knowledge_base.embedding_profile_id].dimensions or 0,
                    candidate_top_k=int(
                        kb_settings.get("candidate_top_k", settings.rag_candidate_top_k)
                    ),
                    document_id=document_id,
                    index_version=index_version,
                )

        route_results = await asyncio.gather(
            *[retrieve_one(knowledge_base) for knowledge_base in knowledge_bases],
            return_exceptions=True,
        )
        ranked_lists: list[tuple[list[RetrievedChunk], float]] = []
        route_names: list[str] = []
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
            kb_settings = knowledge_base.settings or {}
            fusion_weight = float(kb_settings.get("fusion_weight", 1.0))
            retrieval_weights = dict(kb_settings.get("retrieval_weights") or {})
            for route_name in ("bm25", "l1", "original", "summary", "questions", "l3"):
                route_results = result.get(route_name) or []
                if not route_results:
                    continue
                default_weight = 0.8 if route_name == "l1" else 0.6 if route_name == "l3" else 1.0
                route_weight = fusion_weight * float(
                    retrieval_weights.get(route_name, default_weight)
                )
                if route_weight <= 0:
                    continue
                ranked_lists.append((route_results, route_weight))
                route_names.append(route_name)

        final_limit = top_k or (
            settings.rag_final_top_k
            if len(knowledge_bases) == 1
            else settings.rag_multi_kb_final_top_k
        )
        candidates = weighted_rrf_fuse(
            ranked_lists,
            rrf_k=60,
            top_k=max(settings.rag_global_candidate_top_k, final_limit * max(len(knowledge_bases), 1)),
            route_names=route_names,
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
        if expand_context:
            result = self._expand_context_windows(db, result)
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
            token_count = count_tokens(chunk.context_text or chunk.content)
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
            chunk.metadata["summary"] = row.summary or ""
            chunk.metadata["keywords"] = list(row.keywords or [])
            chunk.metadata["hypothetical_questions"] = list(row.hypothetical_questions or [])
            chunk.metadata["qa_pairs"] = list(row.qa_pairs or [])
            chunk.metadata["enrichment_status"] = row.enrichment_status
            chunk.metadata["enrichment_meta"] = dict(row.enrichment_meta or {})
            chunk.metadata["granularity_level"] = row.granularity_level
            chunk.metadata["granule_type"] = row.granule_type
            chunk.metadata["parent_chunk_id"] = row.parent_chunk_id
            chunk.metadata["context_prefix"] = row.context_prefix or ""
            chunk.content = row.content
            parent_content = db_metadata.get("parent_content")
            if parent_content:
                chunk.metadata["child_content"] = row.content
                chunk.metadata["parent_content"] = str(parent_content)
        return chunks

    @staticmethod
    def _attach_search_table_refs(db: Session, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen_tables: set[str] = set()
        for chunk in chunks:
            row = db.get(DocumentChunk, chunk.chunk_id)
            if row is None:
                continue
            table_ids = list((row.metadata_json or {}).get("table_ids") or [])
            if not table_ids and row.parent_chunk_id:
                parent = db.get(DocumentChunk, row.parent_chunk_id)
                table_ids = list((parent.metadata_json or {}).get("table_ids") or []) if parent else []
            if not table_ids:
                continue
            tables = list(
                db.scalars(
                    select(DocumentTable).where(DocumentTable.id.in_(list(dict.fromkeys(table_ids))))
                )
            )
            refs = []
            used_tokens = 0
            for table in tables:
                if table.id in seen_tables:
                    continue
                seen_tables.add(table.id)
                full_text = f"[关联表格]\n{table.caption}\n{table.markdown}".strip()
                full_tokens = count_tokens(full_text)
                if used_tokens + full_tokens <= settings.rag_table_context_tokens:
                    injected_mode = "full"
                    used_tokens += full_tokens
                else:
                    injected_mode = "description"
                    used_tokens += count_tokens(table.description)
                refs.append(
                    {
                        "id": table.id,
                        "caption": table.caption,
                        "page_start": table.page_start,
                        "page_end": table.page_end,
                        "markdown": table.markdown if injected_mode == "full" else "",
                        "description": table.description,
                        "extraction_engine": table.extraction_engine,
                        "quality_score": table.quality_score,
                        "injected_mode": injected_mode,
                    }
                )
            chunk.metadata["context_tables"] = refs
        return chunks

    @staticmethod
    def _expand_context_windows(
        db: Session, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        seen_tables: set[str] = set()
        for chunk in chunks:
            row = db.get(DocumentChunk, chunk.chunk_id)
            if row is None:
                chunk.context_text = chunk.content
                continue
            anchor_row = row
            parent_text = str((row.metadata_json or {}).get("parent_content") or "")
            parent_source = "parent_content" if parent_text else ""
            if row.granularity_level == "L3" and row.parent_chunk_id:
                parent_row = db.get(DocumentChunk, row.parent_chunk_id)
                if parent_row is not None:
                    anchor_row = parent_row
                    parent_text = parent_row.content
                    parent_source = "l3_parent"
            elif row.granularity_level == "L2" and not parent_text:
                section_rows = list(
                    db.scalars(
                        select(DocumentChunk)
                        .where(
                            DocumentChunk.document_id == row.document_id,
                            DocumentChunk.granularity_level == "L2",
                            DocumentChunk.section_path == row.section_path,
                        )
                        .order_by(DocumentChunk.chunk_index.asc())
                    )
                )
                parent_text = "\n\n".join(item.content for item in section_rows)
                parent_source = "section" if parent_text else ""

            prefix = row.context_prefix or str(chunk.metadata.get("context_prefix") or "")
            parts = [prefix] if prefix else []
            parts.append(f"[命中片段]\n{row.content}")
            included = {"parent": False, "previous": False, "next": False, "tables": False}

            parent_window = clip_around_anchor(
                parent_text,
                row.content,
                before_tokens=settings.rag_parent_context_tokens,
                after_tokens=settings.rag_parent_context_tokens,
            )
            parent_context = parent_window.replace(row.content, "", 1).strip()
            if parent_context:
                base_tokens = count_tokens("\n\n".join(parts))
                remaining = settings.rag_context_window_max_tokens - base_tokens
                parent_context = clip_head_tokens(parent_context, remaining)
                if parent_context:
                    parts.append(f"[父文档局部上下文]\n{parent_context}")
                    included["parent"] = True

            table_ids = list((row.metadata_json or {}).get("table_ids") or [])
            if not table_ids and anchor_row.id != row.id:
                table_ids = list((anchor_row.metadata_json or {}).get("table_ids") or [])
            context_tables: list[dict] = []
            table_budget = max(0, settings.rag_table_context_tokens)
            used_table_tokens = 0
            if table_ids:
                tables = list(
                    db.scalars(
                        select(DocumentTable)
                        .where(DocumentTable.id.in_(list(dict.fromkeys(table_ids))))
                        .order_by(DocumentTable.page_start.asc(), DocumentTable.created_at.asc())
                    )
                )
                for table in tables:
                    if table.id in seen_tables:
                        continue
                    seen_tables.add(table.id)
                    if used_table_tokens >= table_budget:
                        break
                    base_tokens = count_tokens("\n\n".join(parts))
                    remaining = min(
                        table_budget - used_table_tokens,
                        settings.rag_context_window_max_tokens - base_tokens,
                    )
                    if remaining <= 0:
                        break
                    full_text = f"[关联表格]\n{table.caption}\n{table.markdown}".strip()
                    if table.markdown and table.markdown in row.content:
                        injected_mode = "existing_chunk"
                    elif count_tokens(full_text) <= remaining:
                        parts.append(full_text)
                        injected_mode = "full"
                        used_table_tokens += count_tokens(full_text)
                    else:
                        description = table.description or "完整表格已独立存储，请查看原始文档。"
                        description = clip_head_tokens(description, remaining)
                        if not description:
                            break
                        parts.append(f"[关联表格摘要]\n{table.caption}\n{description}".strip())
                        injected_mode = "description"
                        used_table_tokens += count_tokens(parts[-1])
                    context_tables.append(
                        {
                            "id": table.id,
                            "caption": table.caption,
                            "page_start": table.page_start,
                            "page_end": table.page_end,
                            "markdown": table.markdown if injected_mode == "full" else "",
                            "description": table.description,
                            "extraction_engine": table.extraction_engine,
                            "quality_score": table.quality_score,
                            "injected_mode": injected_mode,
                        }
                    )
                included["tables"] = bool(context_tables)
            chunk.metadata["context_tables"] = context_tables

            previous_row = db.scalar(
                select(DocumentChunk)
                .where(
                    DocumentChunk.document_id == row.document_id,
                    DocumentChunk.granularity_level == anchor_row.granularity_level,
                    DocumentChunk.chunk_index < anchor_row.chunk_index,
                )
                .order_by(DocumentChunk.chunk_index.desc())
                .limit(1)
            )
            next_row = db.scalar(
                select(DocumentChunk)
                .where(
                    DocumentChunk.document_id == row.document_id,
                    DocumentChunk.granularity_level == anchor_row.granularity_level,
                    DocumentChunk.chunk_index > anchor_row.chunk_index,
                )
                .order_by(DocumentChunk.chunk_index.asc())
                .limit(1)
            )
            neighbor_texts: list[tuple[str, str, str]] = []
            for label, neighbor, text, flag in (
                (
                    "前一段上下文",
                    previous_row,
                    clip_tail_tokens(
                        previous_row.content if previous_row else "",
                        settings.rag_neighbor_context_tokens,
                    ),
                    "previous",
                ),
                (
                    "后一段上下文",
                    next_row,
                    clip_head_tokens(
                        next_row.content if next_row else "",
                        settings.rag_neighbor_context_tokens,
                    ),
                    "next",
                ),
            ):
                if not neighbor or not text:
                    continue
                if neighbor.id == row.id or text.strip() == row.content.strip():
                    continue
                if parent_window and neighbor.content in parent_window:
                    continue
                neighbor_texts.append((label, text, flag))
                included[flag] = True

            for label, text, flag in neighbor_texts:
                candidate = "\n\n".join([*parts, f"[{label}]\n{text}"])
                if count_tokens(candidate) > settings.rag_context_window_max_tokens:
                    included[flag] = False
                    continue
                parts.append(f"[{label}]\n{text}")

            chunk.context_text = "\n\n".join(parts)
            chunk.metadata["context_parent_included"] = included["parent"]
            chunk.metadata["context_parent_source"] = parent_source
            chunk.metadata["context_previous_included"] = included["previous"]
            chunk.metadata["context_next_included"] = included["next"]
            chunk.metadata["context_tokens"] = count_tokens(chunk.context_text)
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
                "granularity_level": chunk.metadata.get("granularity_level", "L2"),
                "granule_type": chunk.metadata.get("granule_type", "paragraph"),
                "parent_chunk_id": chunk.metadata.get("parent_chunk_id"),
                "context_prefix": chunk.metadata.get("context_prefix", ""),
                "context_text": chunk.context_text or chunk.content,
                "matched_questions": list(chunk.metadata.get("matched_questions") or []),
                "qa_pairs": list(chunk.metadata.get("qa_pairs") or []),
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
            "context_tokens_estimate": sum(
                count_tokens(chunk.context_text or chunk.content) for chunk in chunks
            ),
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
        route_hits: dict[str, int] = {}
        for item in results:
            for route in item.metadata.get("matched_representations", []):
                route_hits[str(route)] = route_hits.get(str(route), 0) + 1
        return SearchQualitySummary(
            result_count=len(results),
            avg_rrf_score=round(sum(item.rrf_score for item in results) / len(results), 8),
            best_score=round(max(scores), 8),
            reranked_count=sum(1 for item in results if item.rerank_score is not None),
            vector_hits=sum(1 for item in results if item.vector_rank is not None),
            bm25_hits=sum(1 for item in results if item.bm25_rank is not None),
            route_hits=route_hits,
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
            retrieval_mode=request.retrieval_mode,
        )
        chunks = ChatService._attach_search_table_refs(db, outcome.chunks)
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
                    matched_representations=list(
                        chunk.metadata.get("matched_representations") or []
                    ),
                    representation_scores=dict(
                        chunk.metadata.get("representation_scores") or {}
                    ),
                    representation_ranks=dict(
                        chunk.metadata.get("representation_ranks") or {}
                    ),
                    granularity_level=str(chunk.metadata.get("granularity_level", "L2")),
                    granule_type=str(chunk.metadata.get("granule_type", "paragraph")),
                    parent_chunk_id=chunk.metadata.get("parent_chunk_id"),
                    context_prefix=str(chunk.metadata.get("context_prefix", "")),
                    context_text=chunk.context_text or chunk.content,
                    matched_questions=list(chunk.metadata.get("matched_questions") or []),
                    tables=[
                        TableContextRead(**item)
                        for item in (chunk.metadata.get("context_tables") or [])
                        if isinstance(item, dict)
                    ],
                    metadata=chunk.metadata,
                    file_url=file_url,
                    preview_url=preview_url,
                )
            )
        return SearchResponse(
            query=request.query,
            rewritten_query=rewritten_query if rewritten_query != request.query else None,
            results=results,
            retrieval_trace=outcome.trace,
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
            expand_context=True,
            retrieval_mode=request.retrieval_mode,
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
        uncertainty_notice = (
            outcome.trace.uncertainty_notice
            if outcome.trace and outcome.trace.uncertain
            else ""
        )
        if uncertainty_notice:
            answer = f"{answer.rstrip()}\n\n{uncertainty_notice}"
        citations = self._attach_citation_urls(
            parse_citations(answer, chunks), citation_url_builder
        )
        usage = self._usage(answer, chunks, llm_profile)
        usage["failed_knowledge_base_ids"] = outcome.failed_knowledge_base_ids
        if outcome.trace:
            usage["retrieval_trace"] = outcome.trace.model_dump()
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
            retrieval_trace=outcome.trace,
            uncertainty_notice=uncertainty_notice,
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
            expand_context=True,
            retrieval_mode=request.retrieval_mode,
        )
        chunks = outcome.chunks
        if outcome.trace:
            yield sse_event(
                "retrieval_plan",
                outcome.trace.complexity.model_dump() if outcome.trace.complexity else {},
            )
            for step in outcome.trace.steps:
                yield sse_event("retrieval_step", step.model_dump())
            yield sse_event(
                "retrieval_done",
                {
                    "stop_reason": outcome.trace.stop_reason,
                    "uncertain": outcome.trace.uncertain,
                    "total_rounds": outcome.trace.total_rounds,
                    "total_chunks": outcome.trace.total_chunks,
                    "uncertainty_notice": outcome.trace.uncertainty_notice,
                },
            )
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
                "retrieval_mode": request.retrieval_mode,
                "multi_hop": bool(outcome.trace and outcome.trace.complexity and outcome.trace.complexity.is_multi_hop),
                "uncertain": bool(outcome.trace and outcome.trace.uncertain),
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
            uncertainty_notice = (
                outcome.trace.uncertainty_notice
                if outcome.trace and outcome.trace.uncertain
                else ""
            )
            if uncertainty_notice:
                yield sse_event("notice", {"message": uncertainty_notice})
                notice_chunk = f"\n\n{uncertainty_notice}"
                answer_parts.append(notice_chunk)
                yield sse_event("token", {"content": notice_chunk})
            answer = "".join(answer_parts)
            citations = self._attach_citation_urls(
                parse_citations(answer, chunks), citation_url_builder
            )
            usage = self._usage(answer, chunks, llm_profile)
            usage["failed_knowledge_base_ids"] = outcome.failed_knowledge_base_ids
            if outcome.trace:
                usage["retrieval_trace"] = outcome.trace.model_dump()
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
                    "uncertainty_notice": uncertainty_notice,
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




