import asyncio
import math
import random
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.db.models import (
    ChunkEvaluationItem,
    ChunkEvaluationResult,
    ChunkEvaluationRun,
    Document,
    DocumentChunk,
    DocumentStatus,
    EvaluationItemStatus,
    EvaluationStage,
    EvaluationStatus,
    KnowledgeBase,
    ModelKind,
    ModelProfile,
)
from app.evaluation.chunk_evaluator import (
    LLM_PROMPT_VERSION,
    RULE_VERSION,
    SCORE_WEIGHTS,
    EvaluationSample,
    LLMChunkJudge,
    combine_scores,
    evaluate_rules,
    normalize_text,
    source_units,
)
from app.indexing.model_gateway import ModelGateway
from app.retrieval.context_window import clip_around_anchor, clip_head_tokens, clip_tail_tokens
from app.schemas import (
    ChunkEvaluationChunkRead,
    ChunkEvaluationCreate,
    ChunkEvaluationRetryFailed,
    ChunkEvaluationReviewUpdate,
)

REVIEW_SAMPLE_TARGET = 24
REVIEW_PER_LEVEL_MIN = 4
DEFAULT_INITIAL_CHUNK_LIMIT = 200
DEFAULT_LLM_CONCURRENCY = 2
MAX_RUN_ERRORS = 100
PAUSABLE_STATUSES = {EvaluationStatus.QUEUED, EvaluationStatus.RUNNING}
EDITABLE_STATUSES = {
    EvaluationStatus.PAUSED,
    EvaluationStatus.PARTIAL,
    EvaluationStatus.COMPLETED,
    EvaluationStatus.FAILED,
}
PENDING_ITEM_STATUSES = {
    EvaluationItemStatus.QUEUED,
    EvaluationItemStatus.RULES_RUNNING,
    EvaluationItemStatus.LLM_QUEUED,
    EvaluationItemStatus.LLM_RUNNING,
}


class ChunkEvaluationService:
    def __init__(self, gateway: ModelGateway | None = None):
        self.gateway = gateway or ModelGateway()
        self.judge = LLMChunkJudge(self.gateway)

    def create_run(self, db: Session, request: ChunkEvaluationCreate) -> ChunkEvaluationRun:
        document: Document | None = None
        if request.document_id:
            document = db.get(Document, request.document_id)
            if document is None:
                raise NotFoundError("文档不存在")
            if document.status != DocumentStatus.READY:
                raise ValidationError("文档尚未完成入库")
            knowledge_base = db.get(KnowledgeBase, document.kb_id)
            if knowledge_base is None:
                raise NotFoundError("知识库不存在")
            knowledge_base_id = knowledge_base.id
            scope_type = "document"
        else:
            knowledge_base = db.get(KnowledgeBase, request.knowledge_base_id)
            if knowledge_base is None:
                raise NotFoundError("知识库不存在")
            knowledge_base_id = knowledge_base.id
            scope_type = "knowledge_base"
        judge_profile = (
            self._validate_judge_profile(db, request.judge_profile_id)
            if request.judge_profile_id
            else None
        )
        if request.llm_enabled and judge_profile is None:
            judge_profile = self._default_judge_profile(db, knowledge_base)
        run = ChunkEvaluationRun(
            id=str(uuid.uuid4()),
            scope_type=scope_type,
            knowledge_base_id=knowledge_base_id,
            document_id=request.document_id,
            granularities=request.granularities,
            judge_profile_id=request.judge_profile_id,
            llm_enabled=request.llm_enabled,
            status=EvaluationStatus.QUEUED,
            stage=EvaluationStage.QUEUED,
            progress={
                "total_documents": 0,
                "processed_documents": 0,
                "evaluated_chunks": 0,
                "llm_failed_chunks": 0,
                "chunks_total": 0,
                "chunks_succeeded": 0,
                "chunks_failed": 0,
                "chunks_pending": 0,
                "chunks_interrupted": 0,
                "chunks_not_selected": 0,
                "batch_total": 0,
                "batch_completed": 0,
                "batch_running": 0,
            },
            config_snapshot={
                "scope_type": scope_type,
                "knowledge_base_name": knowledge_base.name,
                "document_name": document.filename if document else None,
                "granularities": request.granularities,
                "llm_enabled": request.llm_enabled,
                "judge_profile_id": request.judge_profile_id,
                "judge_profile_name": judge_profile.name if judge_profile else None,
                "judge_model_name": judge_profile.model_name if judge_profile else None,
                "initial_chunk_limit": request.initial_chunk_limit,
                "llm_concurrency": request.llm_concurrency,
                "selection_strategy": "stratified",
                "rule_version": RULE_VERSION,
                "llm_prompt_version": LLM_PROMPT_VERSION,
                "score_weights": SCORE_WEIGHTS,
                "review_sample_target": REVIEW_SAMPLE_TARGET,
            },
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return run

    @staticmethod
    def _validate_judge_profile(db: Session, profile_id: str) -> ModelProfile:
        profile = db.get(ModelProfile, profile_id)
        if (
            profile is None
            or not profile.is_active
            or profile.kind not in {ModelKind.LLM, ModelKind.ENRICHMENT}
        ):
            raise ValidationError("评判模型不可用，必须为启用的 LLM 或增强模型")
        return profile

    @staticmethod
    def _default_judge_profile(
        db: Session,
        knowledge_base: KnowledgeBase,
    ) -> ModelProfile | None:
        for profile_id in (
            knowledge_base.enrichment_profile_id,
            knowledge_base.default_llm_profile_id,
        ):
            if not profile_id:
                continue
            profile = db.get(ModelProfile, profile_id)
            if (
                profile is not None
                and profile.is_active
                and profile.kind in {ModelKind.LLM, ModelKind.ENRICHMENT}
            ):
                return profile
        return None

    def _judge_profile(self, db: Session, run: ChunkEvaluationRun) -> ModelProfile | None:
        if run.judge_profile_id:
            return self._validate_judge_profile(db, run.judge_profile_id)
        knowledge_base = db.get(KnowledgeBase, run.knowledge_base_id)
        if knowledge_base is None:
            return None
        return self._default_judge_profile(db, knowledge_base)

    async def run(self, db: Session, run_id: str) -> None:
        run = db.get(ChunkEvaluationRun, run_id)
        if run is None or run.status in {
            EvaluationStatus.PAUSED,
            EvaluationStatus.PAUSING,
            EvaluationStatus.PARTIAL,
            EvaluationStatus.COMPLETED,
            EvaluationStatus.FAILED,
        }:
            return
        run.status = EvaluationStatus.RUNNING
        run.stage = EvaluationStage.RULES_RUNNING
        run.started_at = run.started_at or datetime.now(UTC)
        run.finished_at = None
        db.commit()

        try:
            self._ensure_items(db, run)
            self._update_progress(db, run)
            db.commit()
            if not run.items:
                self._finalize(db, run, [{"scope": "run", "error": "没有可评估的 chunk"}])
                return

            paused = await self._run_rules(db, run)
            if paused or self._pause_requested(db, run):
                self._mark_paused(db, run)
                return

            if not run.llm_enabled:
                self._complete_without_llm(db, run)
            else:
                run = db.get(ChunkEvaluationRun, run_id)
                if run is None:
                    return
                run.stage = EvaluationStage.LLM_RUNNING
                self._update_progress(db, run)
                db.commit()
                profile = self._judge_profile(db, run)
                if profile is None:
                    self._degrade_all_pending(db, run, "未配置可用的 LLM 评判模型，已降级为规则评估")
                else:
                    await self._run_llm_queue(db, run, profile)
                if self._pause_requested(db, run):
                    self._mark_paused(db, run)
                    return

            run = db.get(ChunkEvaluationRun, run_id)
            if run is None:
                return
            run.stage = EvaluationStage.AGGREGATING
            self._update_progress(db, run)
            db.commit()
            self._finalize(db, run, [])
        except asyncio.CancelledError:
            db.rollback()
            self._mark_paused(db, run)
            raise
        except Exception as exc:
            db.rollback()
            run = db.get(ChunkEvaluationRun, run_id)
            if run is None:
                return
            if self._pause_requested(db, run):
                self._mark_paused(db, run)
                return
            self._append_error(run, {"scope": "run", "error": str(exc)[:1000]})
            run.status = EvaluationStatus.FAILED
            run.stage = EvaluationStage.FAILED
            run.finished_at = datetime.now(UTC)
            self._update_progress(db, run)
            db.commit()
    def _documents_for_run(self, db: Session, run: ChunkEvaluationRun) -> list[Document]:
        if run.document_id:
            document = db.get(Document, run.document_id)
            return [document] if document and document.status == DocumentStatus.READY else []
        return list(
            db.scalars(
                select(Document)
                .where(
                    Document.kb_id == run.knowledge_base_id,
                    Document.status == DocumentStatus.READY,
                )
                .order_by(Document.created_at.asc())
            )
        )

    def _eligible_chunks(
        self,
        db: Session,
        run: ChunkEvaluationRun,
        documents: list[Document],
    ) -> dict[str, list[DocumentChunk]]:
        document_ids = [document.id for document in documents]
        if not document_ids:
            return {}
        rows = list(
            db.scalars(
                select(DocumentChunk)
                .where(
                    DocumentChunk.document_id.in_(document_ids),
                    DocumentChunk.granularity_level.in_(run.granularities),
                )
                .order_by(DocumentChunk.document_id.asc(), DocumentChunk.chunk_index.asc())
            )
        )
        grouped: dict[str, list[DocumentChunk]] = defaultdict(list)
        for row in rows:
            grouped[row.document_id].append(row)
        return grouped

    @staticmethod
    def _level_quotas(groups: dict[str, list[DocumentChunk]], limit: int) -> dict[str, int]:
        total = sum(len(items) for items in groups.values())
        if total <= limit:
            return {level: len(items) for level, items in groups.items()}
        quotas = {
            level: max(1, int(limit * len(items) / total))
            for level, items in groups.items()
            if items
        }
        while sum(quotas.values()) > limit:
            candidates = [level for level, value in quotas.items() if value > 1]
            if not candidates:
                break
            level = max(candidates, key=lambda item: (quotas[item], item))
            quotas[level] -= 1
        levels = sorted(groups)
        while sum(quotas.values()) < limit:
            candidates = [level for level in levels if quotas.get(level, 0) < len(groups[level])]
            if not candidates:
                break
            level = max(
                candidates,
                key=lambda item: (
                    len(groups[item]) / (quotas.get(item, 0) + 1),
                    -levels.index(item),
                ),
            )
            quotas[level] = quotas.get(level, 0) + 1
        return quotas

    def _ensure_items(self, db: Session, run: ChunkEvaluationRun) -> None:
        documents = self._documents_for_run(db, run)
        config = dict(run.config_snapshot or {})
        config["chunk_configs"] = [
            {
                "document_id": document.id,
                "chunk_mode": document.chunk_mode,
                "chunk_size": document.chunk_size,
                "chunk_overlap": document.chunk_overlap,
            }
            for document in documents
        ]
        config["total_documents"] = len(documents)
        grouped = self._eligible_chunks(db, run, documents)
        config["candidate_chunk_count"] = sum(len(items) for items in grouped.values())
        run.config_snapshot = config
        run.progress = {
            **dict(run.progress or {}),
            "total_documents": len(documents),
        }
        if db.scalar(
            select(func.count())
            .select_from(ChunkEvaluationItem)
            .where(ChunkEvaluationItem.run_id == run.id)
        ):
            self._sync_selection_snapshot(db, run)
            return

        limit = int(config.get("initial_chunk_limit", DEFAULT_INITIAL_CHUNK_LIMIT))
        positions = 0
        selected: list[DocumentChunk] = []
        for document in documents:
            chunks = grouped.get(document.id, [])
            if not chunks:
                continue
            by_level: dict[str, list[DocumentChunk]] = defaultdict(list)
            for item in chunks:
                by_level[item.granularity_level].append(item)
            rng = random.Random(document.id)
            for level in sorted(by_level):
                rng.shuffle(by_level[level])
            quotas = self._level_quotas(by_level, limit)
            document_selected = [
                item
                for level in sorted(by_level)
                for item in by_level[level][: quotas.get(level, 0)]
            ]
            document_selected.sort(key=lambda item: (item.chunk_index, item.granularity_level))
            selected.extend(document_selected)
        for chunk in selected:
            positions += 1
            db.add(self._item_from_chunk(run, chunk, positions, "automatic"))
        db.flush()
        self._sync_selection_snapshot(db, run)

    def _sync_selection_snapshot(self, db: Session, run: ChunkEvaluationRun) -> None:
        items = list(
            db.scalars(
                select(ChunkEvaluationItem).where(ChunkEvaluationItem.run_id == run.id)
            )
        )
        config = dict(run.config_snapshot or {})
        selected_by_document = Counter(item.document_id for item in items)
        config["selection_by_document"] = dict(selected_by_document)
        config["selected_chunk_count"] = len(items)
        run.config_snapshot = config
        self._update_progress(db, run)

    @staticmethod
    def _item_from_chunk(
        run: ChunkEvaluationRun,
        chunk: DocumentChunk,
        queue_position: int,
        selection_source: str,
    ) -> ChunkEvaluationItem:
        return ChunkEvaluationItem(
            id=str(uuid.uuid4()),
            run_id=run.id,
            document_id=chunk.document_id,
            chunk_id=chunk.id,
            chunk_index=chunk.chunk_index,
            granularity_level=chunk.granularity_level,
            section_path=chunk.section_path,
            content_snapshot=chunk.content,
            selection_source=selection_source,
            queue_position=queue_position,
            status=EvaluationItemStatus.QUEUED,
        )

    def _pause_requested(self, db: Session, run: ChunkEvaluationRun) -> bool:
        db.refresh(run)
        return run.status == EvaluationStatus.PAUSING or bool(run.pause_requested_at)

    async def _run_rules(self, db: Session, run: ChunkEvaluationRun) -> bool:
        contexts: dict[str, dict[str, Any]] = {}
        while True:
            if self._pause_requested(db, run):
                return True
            item = db.scalar(
                select(ChunkEvaluationItem)
                .where(
                    ChunkEvaluationItem.run_id == run.id,
                    ChunkEvaluationItem.status == EvaluationItemStatus.QUEUED,
                )
                .order_by(ChunkEvaluationItem.queue_position.asc())
                .limit(1)
            )
            if item is None:
                return False
            item.status = EvaluationItemStatus.RULES_RUNNING
            item.started_at = item.started_at or datetime.now(UTC)
            item.error = None
            self._update_progress(db, run)
            db.commit()
            try:
                chunk = db.get(DocumentChunk, item.chunk_id)
                if chunk is None:
                    raise ValidationError("chunk 已不存在")
                if chunk.document_id not in contexts:
                    document = db.get(Document, chunk.document_id)
                    if document is None:
                        raise ValidationError("文档已不存在")
                    contexts[chunk.document_id] = self._document_context(db, document)
                context = contexts[chunk.document_id]
                sample_context = self._context_snapshot(
                    chunk,
                    context["all_chunks"],
                    context["chunks_by_id"],
                )
                rule_metrics = evaluate_rules(
                    chunk,
                    source_text=(
                        str(sample_context.get("section_source", ""))
                        if chunk.granularity_level == "L1"
                        else context["document"].extracted_text
                    ),
                    source_unit_list=context["source_units"],
                    all_l2_content=context["l2_content"],
                    duplicate_units=context["duplicate_units"],
                    has_pages=context["has_pages"],
                    previous_content=str(sample_context.get("previous_excerpt", "")),
                    next_content=str(sample_context.get("next_excerpt", "")),
                )
                result = self._store_rule_result(
                    db,
                    run,
                    item,
                    chunk,
                    sample_context,
                    rule_metrics,
                )
                item.result_id = result.id
                item.rule_succeeded = True
                item.status = (
                    EvaluationItemStatus.LLM_QUEUED
                    if run.llm_enabled
                    else EvaluationItemStatus.SUCCEEDED
                )
                item.finished_at = None if run.llm_enabled else datetime.now(UTC)
            except Exception as exc:
                self._mark_item_failed(db, run, item, str(exc), fallback=False)
            self._update_progress(db, run)
            db.commit()

    def _document_context(self, db: Session, document: Document) -> dict[str, Any]:
        all_chunks = list(
            db.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document.id)
                .order_by(DocumentChunk.chunk_index.asc())
            )
        )
        all_l2 = [item for item in all_chunks if item.granularity_level == "L2"]
        units = source_units(document.extracted_text)
        unit_chunks: dict[str, set[str]] = defaultdict(set)
        for chunk in all_l2:
            normalized = normalize_text(chunk.content)
            for unit in units:
                normalized_unit = normalize_text(unit)
                if normalized_unit and normalized_unit in normalized:
                    unit_chunks[normalized_unit].add(chunk.id)
        duplicate_units = {unit for unit, owners in unit_chunks.items() if len(owners) > 1}
        metadata = document.metadata_json or {}
        return {
            "document": document,
            "all_chunks": all_chunks,
            "chunks_by_id": {item.id: item for item in all_chunks},
            "l2_content": [item.content for item in all_l2],
            "source_units": units,
            "duplicate_units": duplicate_units,
            "has_pages": bool(metadata.get("pages") or metadata.get("page_count")),
        }

    def _store_rule_result(
        self,
        db: Session,
        run: ChunkEvaluationRun,
        item: ChunkEvaluationItem,
        chunk: DocumentChunk,
        context: dict[str, Any],
        rule_metrics: dict[str, Any],
    ) -> ChunkEvaluationResult:
        composite, grade, issues, suggestions, components = combine_scores(
            chunk.granularity_level,
            rule_metrics,
            {},
            llm_available=False,
        )
        result = db.get(ChunkEvaluationResult, item.result_id) if item.result_id else None
        if result is None:
            result = ChunkEvaluationResult(
                id=str(uuid.uuid4()),
                run_id=run.id,
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                kb_id=chunk.kb_id,
                granularity_level=chunk.granularity_level,
                granule_type=chunk.granule_type,
                chunk_index=chunk.chunk_index,
                content_snapshot=chunk.content,
                section_path=chunk.section_path,
                context_prefix=chunk.context_prefix or "",
                parent_chunk_id=chunk.parent_chunk_id,
                created_at=datetime.now(UTC),
            )
            db.add(result)
        result.metadata_snapshot = {
            **dict(chunk.metadata_json or {}),
            "token_count": chunk.token_count,
            "summary": chunk.summary,
            "keywords": list(chunk.keywords or []),
            "qa_pairs": list(chunk.qa_pairs or []),
        }
        result.context_snapshot = {
            **context,
            "component_scores": components,
            "llm_fallback_reason": None,
        }
        result.rule_metrics = rule_metrics
        result.llm_scores = {}
        result.llm_reason = ""
        result.issue_codes = issues
        result.suggestions = suggestions
        result.composite_score = composite
        result.grade = grade
        db.flush()
        return result
    def _sample_from_result(self, result: ChunkEvaluationResult) -> EvaluationSample:
        chunk = SimpleNamespace(
            id=result.chunk_id,
            content=result.content_snapshot,
            granularity_level=result.granularity_level,
            section_path=result.section_path,
            context_prefix=result.context_prefix,
        )
        return EvaluationSample(
            chunk=chunk,
            context=dict(result.context_snapshot or {}),
            source_units=[],
        )

    async def _request_llm(
        self,
        profile: ModelProfile,
        sample: EvaluationSample,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        results, errors = await self.judge.evaluate(profile, [sample])
        return results.get(sample.chunk.id), errors

    async def _run_llm_queue(
        self,
        db: Session,
        run: ChunkEvaluationRun,
        profile: ModelProfile,
    ) -> None:
        concurrency = max(
            1,
            min(4, int((run.config_snapshot or {}).get("llm_concurrency", DEFAULT_LLM_CONCURRENCY))),
        )
        tasks: dict[asyncio.Task, str] = {}
        failure_streak = 0
        circuit_open = False
        while True:
            db.refresh(run)
            if self._pause_requested(db, run):
                await self._cancel_llm_tasks(db, run, tasks, interrupted=True)
                return

            if not circuit_open:
                while len(tasks) < concurrency:
                    item = db.scalar(
                        select(ChunkEvaluationItem)
                        .where(
                            ChunkEvaluationItem.run_id == run.id,
                            ChunkEvaluationItem.status == EvaluationItemStatus.LLM_QUEUED,
                        )
                        .order_by(ChunkEvaluationItem.queue_position.asc())
                        .limit(1)
                    )
                    if item is None:
                        break
                    result = db.get(ChunkEvaluationResult, item.result_id) if item.result_id else None
                    if result is None:
                        self._mark_item_failed(db, run, item, "规则结果不存在，无法调用 LLM", False)
                        self._update_progress(db, run)
                        db.commit()
                        continue
                    item.status = EvaluationItemStatus.LLM_RUNNING
                    item.attempt_count += 1
                    item.started_at = datetime.now(UTC)
                    item.finished_at = None
                    item.error = None
                    sample = self._sample_from_result(result)
                    tasks[asyncio.create_task(self._request_llm(profile, sample))] = item.id
                    self._update_progress(db, run)
                    db.commit()

            if not tasks:
                if circuit_open:
                    return
                pending = db.scalar(
                    select(func.count())
                    .select_from(ChunkEvaluationItem)
                    .where(
                        ChunkEvaluationItem.run_id == run.id,
                        ChunkEvaluationItem.status == EvaluationItemStatus.LLM_QUEUED,
                    )
                )
                if not pending:
                    return
                continue

            done, _ = await asyncio.wait(
                tasks,
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                continue
            for task in done:
                item_id = tasks.pop(task)
                item = db.get(ChunkEvaluationItem, item_id)
                if item is None:
                    continue
                try:
                    llm_scores, errors = task.result()
                except asyncio.CancelledError:
                    self._mark_item_interrupted(db, run, item, "LLM 请求已取消")
                    llm_scores, errors = None, ["LLM 请求已取消"]
                except Exception as exc:
                    llm_scores, errors = None, [str(exc)]
                if llm_scores:
                    self._apply_llm_success(db, item, llm_scores)
                    failure_streak = 0
                else:
                    error = errors[-1] if errors else "LLM 未返回有效结果"
                    self._mark_item_failed(db, run, item, error, fallback=item.rule_succeeded)
                    failure_streak += 1
                self._update_progress(db, run)
                db.commit()

            if failure_streak >= 2 and not circuit_open:
                circuit_open = True
                self._append_error(
                    run,
                    {"scope": "llm", "error": "连续 2 个 LLM 请求失败，已熔断并降级为规则结果"},
                )
                await self._cancel_llm_tasks(db, run, tasks, interrupted=False)
                self._degrade_all_pending(db, run, "LLM 熔断，已降级为规则结果")
                return

    async def _cancel_llm_tasks(
        self,
        db: Session,
        run: ChunkEvaluationRun,
        tasks: dict[asyncio.Task, str],
        *,
        interrupted: bool,
    ) -> None:
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for item_id in tasks.values():
            item = db.get(ChunkEvaluationItem, item_id)
            if item is None:
                continue
            if interrupted:
                self._mark_item_interrupted(db, run, item, "评估已暂停，LLM 请求中断")
            else:
                self._mark_item_failed(db, run, item, "LLM 已熔断，规则结果继续生效", True)
        tasks.clear()
        self._update_progress(db, run)
        db.commit()

    def _apply_llm_success(
        self,
        db: Session,
        item: ChunkEvaluationItem,
        llm_scores: dict[str, Any],
    ) -> None:
        result = db.get(ChunkEvaluationResult, item.result_id) if item.result_id else None
        if result is None:
            run = db.get(ChunkEvaluationRun, item.run_id)
            if run is not None:
                self._mark_item_failed(db, run, item, "评估结果不存在", False)
            return
        composite, grade, issues, suggestions, components = combine_scores(
            result.granularity_level,
            result.rule_metrics or {},
            llm_scores,
            llm_available=True,
        )
        result.llm_scores = llm_scores
        result.llm_reason = str(llm_scores.get("reason") or "")
        result.issue_codes = issues
        result.suggestions = suggestions
        result.composite_score = composite
        result.grade = grade
        result.context_snapshot = {
            **dict(result.context_snapshot or {}),
            "component_scores": components,
            "missing_facts": llm_scores.get("missing_facts") or [],
            "confidence": llm_scores.get("confidence"),
            "llm_fallback_reason": None,
        }
        item.status = EvaluationItemStatus.SUCCEEDED
        item.llm_succeeded = True
        item.fallback_applied = False
        item.error = None
        item.finished_at = datetime.now(UTC)

    def _mark_item_failed(
        self,
        db: Session,
        run: ChunkEvaluationRun,
        item: ChunkEvaluationItem,
        error: str,
        fallback: bool,
    ) -> None:
        item.status = EvaluationItemStatus.FAILED
        item.llm_succeeded = False
        item.fallback_applied = bool(fallback and item.rule_succeeded)
        item.error = error[:1000]
        item.finished_at = datetime.now(UTC)
        result = db.get(ChunkEvaluationResult, item.result_id) if item.result_id else None
        if result is not None:
            result.context_snapshot = {
                **dict(result.context_snapshot or {}),
                "llm_fallback_reason": item.error if item.fallback_applied else None,
            }

    def _mark_item_interrupted(
        self,
        db: Session,
        run: ChunkEvaluationRun,
        item: ChunkEvaluationItem,
        error: str,
    ) -> None:
        item.status = EvaluationItemStatus.INTERRUPTED
        item.error = error[:1000]
        item.finished_at = datetime.now(UTC)
        result = db.get(ChunkEvaluationResult, item.result_id) if item.result_id else None
        if result is not None:
            result.context_snapshot = {
                **dict(result.context_snapshot or {}),
                "llm_fallback_reason": "LLM 请求因暂停中断，恢复后重试",
            }

    def _complete_without_llm(self, db: Session, run: ChunkEvaluationRun) -> None:
        items = list(
            db.scalars(
                select(ChunkEvaluationItem).where(
                    ChunkEvaluationItem.run_id == run.id,
                    ChunkEvaluationItem.status == EvaluationItemStatus.LLM_QUEUED,
                )
            )
        )
        for item in items:
            item.status = EvaluationItemStatus.SUCCEEDED
            item.finished_at = datetime.now(UTC)
        self._update_progress(db, run)
        db.commit()

    def _degrade_all_pending(
        self,
        db: Session,
        run: ChunkEvaluationRun,
        reason: str,
    ) -> None:
        items = list(
            db.scalars(
                select(ChunkEvaluationItem).where(
                    ChunkEvaluationItem.run_id == run.id,
                    ChunkEvaluationItem.status.in_(
                        [EvaluationItemStatus.LLM_QUEUED, EvaluationItemStatus.LLM_RUNNING]
                    ),
                )
            )
        )
        for item in items:
            self._mark_item_failed(db, run, item, reason, fallback=item.rule_succeeded)
        self._append_error(run, {"scope": "llm", "error": reason})
        self._update_progress(db, run)
        db.commit()

    @staticmethod
    def _append_error(run: ChunkEvaluationRun, error: dict[str, Any]) -> None:
        errors = list(run.errors or [])
        if error not in errors and len(errors) < MAX_RUN_ERRORS:
            errors.append(error)
        run.errors = errors

    def _mark_paused(self, db: Session, run: ChunkEvaluationRun) -> None:
        db.refresh(run)
        now = datetime.now(UTC)
        running_items = list(
            db.scalars(
                select(ChunkEvaluationItem).where(
                    ChunkEvaluationItem.run_id == run.id,
                    ChunkEvaluationItem.status.in_(
                        [EvaluationItemStatus.RULES_RUNNING, EvaluationItemStatus.LLM_RUNNING]
                    ),
                )
            )
        )
        for item in running_items:
            self._mark_item_interrupted(db, run, item, "评估已暂停")
        run.status = EvaluationStatus.PAUSED
        run.stage = EvaluationStage.PAUSED
        run.paused_at = now
        run.heartbeat_at = now
        self._update_progress(db, run)
        db.commit()

    def _finalize(
        self,
        db: Session,
        run: ChunkEvaluationRun,
        extra_errors: list[dict[str, Any]],
    ) -> None:
        for error in extra_errors:
            self._append_error(run, error)
        self._select_review_samples(db, run)
        run.aggregate = self._aggregate(db, run)
        failed_count = int((run.progress or {}).get("chunks_failed", 0) or 0)
        if not run.aggregate.get("result_count"):
            run.status = EvaluationStatus.FAILED
            run.stage = EvaluationStage.FAILED
            self._append_error(run, {"scope": "run", "error": "没有可评估的 chunk"})
        elif failed_count or run.errors:
            run.status = EvaluationStatus.PARTIAL
            run.stage = EvaluationStage.COMPLETED
        else:
            run.status = EvaluationStatus.COMPLETED
            run.stage = EvaluationStage.COMPLETED
        run.finished_at = datetime.now(UTC)
        self._update_progress(db, run)
        db.commit()

    def _update_progress(self, db: Session, run: ChunkEvaluationRun) -> None:
        db.flush()
        items = list(
            db.scalars(
                select(ChunkEvaluationItem).where(ChunkEvaluationItem.run_id == run.id)
            )
        )
        counts = Counter(item.status for item in items)
        pending = sum(counts[status] for status in PENDING_ITEM_STATUSES)
        succeeded = counts[EvaluationItemStatus.SUCCEEDED]
        failed = counts[EvaluationItemStatus.FAILED]
        running = counts[EvaluationItemStatus.LLM_RUNNING]
        rules_completed = sum(1 for item in items if item.rule_succeeded)
        configured_total = int((run.config_snapshot or {}).get("candidate_chunk_count", 0) or 0)
        total_documents = int((run.progress or {}).get("total_documents", 0) or 0)
        by_document: dict[str, list[ChunkEvaluationItem]] = defaultdict(list)
        for item in items:
            by_document[item.document_id].append(item)
        processed_documents = sum(
            1
            for rows in by_document.values()
            if rows and all(item.status not in PENDING_ITEM_STATUSES for item in rows)
        )
        run.progress = {
            **dict(run.progress or {}),
            "total_documents": total_documents,
            "processed_documents": processed_documents,
            "evaluated_chunks": rules_completed,
            "llm_failed_chunks": failed,
            "rules_completed": rules_completed,
            "chunks_total": len(items),
            "chunks_succeeded": succeeded,
            "chunks_failed": failed,
            "chunks_pending": pending,
            "chunks_interrupted": counts[EvaluationItemStatus.INTERRUPTED],
            "chunks_not_selected": max(configured_total - len(items), 0),
            "batch_total": len(items) if run.llm_enabled else 0,
            "batch_completed": succeeded + failed if run.llm_enabled else 0,
            "batch_running": running if run.llm_enabled else 0,
            "current_batch": min(succeeded + failed + running, len(items)) if run.llm_enabled else 0,
        }
        run.heartbeat_at = datetime.now(UTC)
    def _context_snapshot(
        self,
        chunk: DocumentChunk,
        all_chunks: list[DocumentChunk],
        chunks_by_id: dict[str, DocumentChunk],
    ) -> dict[str, Any]:
        l2_chunks = [item for item in all_chunks if item.granularity_level == "L2"]
        section_chunks = [item for item in l2_chunks if item.section_path == chunk.section_path]
        section_source = "\n\n".join(item.content for item in section_chunks)[:8000]
        anchor = chunk
        parent_text = str((chunk.metadata_json or {}).get("parent_content") or "")
        if chunk.granularity_level == "L3" and chunk.parent_chunk_id:
            parent = chunks_by_id.get(chunk.parent_chunk_id)
            if parent:
                anchor = parent
                parent_text = parent.content
        elif chunk.granularity_level == "L2" and not parent_text:
            parent_text = section_source
        parent_excerpt = clip_around_anchor(
            parent_text,
            chunk.content,
            before_tokens=640,
            after_tokens=640,
        )
        siblings = [
            item
            for item in all_chunks
            if item.granularity_level == anchor.granularity_level
        ]
        position = next(
            (index for index, item in enumerate(siblings) if item.id == anchor.id),
            -1,
        )
        previous = siblings[position - 1].content if position > 0 else ""
        next_content = (
            siblings[position + 1].content
            if 0 <= position < len(siblings) - 1
            else ""
        )
        return {
            "section_source": section_source,
            "parent_excerpt": parent_excerpt,
            "previous_excerpt": clip_tail_tokens(previous, 320),
            "next_excerpt": clip_head_tokens(next_content, 320),
        }

    def list_chunks(
        self,
        db: Session,
        run_id: str,
        *,
        document_id: str | None = None,
        granularity: str | None = None,
        status: str | None = None,
        search: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[ChunkEvaluationChunkRead], int]:
        run = self._require_run(db, run_id)
        documents = self._documents_for_run(db, run)
        document_ids = [document.id for document in documents]
        if not document_ids:
            return [], 0
        join_condition = and_(
            ChunkEvaluationItem.run_id == run.id,
            ChunkEvaluationItem.chunk_id == DocumentChunk.id,
        )
        filters = [
            DocumentChunk.document_id.in_(document_ids),
            DocumentChunk.granularity_level.in_(run.granularities),
        ]
        if document_id:
            if document_id not in document_ids:
                raise ValidationError("文档不属于当前评估范围")
            filters.append(DocumentChunk.document_id == document_id)
        if granularity:
            if granularity not in run.granularities:
                raise ValidationError("粒度不在当前评估范围内")
            filters.append(DocumentChunk.granularity_level == granularity)
        if search:
            pattern = f"%{search.strip()}%"
            filters.append(
                or_(
                    DocumentChunk.content.ilike(pattern),
                    DocumentChunk.section_path.ilike(pattern),
                )
            )
        status_expression = func.coalesce(ChunkEvaluationItem.status, "not_selected")
        if status:
            filters.append(status_expression == status)
        statement = (
            select(DocumentChunk, ChunkEvaluationItem)
            .outerjoin(ChunkEvaluationItem, join_condition)
            .where(*filters)
            .order_by(DocumentChunk.document_id.asc(), DocumentChunk.chunk_index.asc())
        )
        count_statement = (
            select(func.count())
            .select_from(DocumentChunk)
            .outerjoin(ChunkEvaluationItem, join_condition)
            .where(*filters)
        )
        total = int(db.scalar(count_statement) or 0)
        rows = db.execute(
            statement.offset((page - 1) * page_size).limit(page_size)
        ).all()
        items = [
            ChunkEvaluationChunkRead(
                id=item.id if item else None,
                run_id=run.id,
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                chunk_index=chunk.chunk_index,
                granularity_level=chunk.granularity_level,
                section_path=chunk.section_path,
                content_snapshot=item.content_snapshot if item else chunk.content,
                status=item.status if item else "not_selected",
                selection_source=item.selection_source if item else "none",
                queue_position=item.queue_position if item else None,
                attempt_count=item.attempt_count if item else 0,
                rule_succeeded=item.rule_succeeded if item else False,
                llm_succeeded=item.llm_succeeded if item else False,
                fallback_applied=item.fallback_applied if item else False,
                error=item.error if item else None,
                result_id=item.result_id if item else None,
                started_at=item.started_at if item else None,
                finished_at=item.finished_at if item else None,
            )
            for chunk, item in rows
        ]
        return items, total

    def select_chunks(
        self,
        db: Session,
        run_id: str,
        chunk_ids: list[str],
    ) -> ChunkEvaluationRun:
        run = self._require_run(db, run_id)
        if run.status not in EDITABLE_STATUSES:
            raise ConflictError("当前运行状态不允许加入 chunk，请先暂停或等待完成")
        documents = self._documents_for_run(db, run)
        document_ids = [document.id for document in documents]
        chunks = list(
            db.scalars(
                select(DocumentChunk)
                .where(
                    DocumentChunk.id.in_(chunk_ids),
                    DocumentChunk.document_id.in_(document_ids),
                    DocumentChunk.granularity_level.in_(run.granularities),
                )
                .order_by(DocumentChunk.document_id.asc(), DocumentChunk.chunk_index.asc())
            )
        )
        if len(chunks) != len(set(chunk_ids)):
            raise ValidationError("部分 chunk 不存在或不属于当前评估范围")
        existing = set(
            db.scalars(
                select(ChunkEvaluationItem.chunk_id).where(
                    ChunkEvaluationItem.run_id == run.id
                )
            )
        )
        position = max((item.queue_position for item in run.items), default=0)
        added = 0
        for chunk in chunks:
            if chunk.id in existing:
                continue
            position += 1
            db.add(self._item_from_chunk(run, chunk, position, "manual"))
            added += 1
        if not added:
            raise ValidationError("所选 chunk 已在当前评估队列中")
        self._prepare_for_execution(run)
        db.flush()
        self._sync_selection_snapshot(db, run)
        db.commit()
        db.refresh(run)
        return run
    def retry_item(self, db: Session, run_id: str, item_id: str) -> ChunkEvaluationRun:
        run = self._require_run(db, run_id)
        if run.status not in EDITABLE_STATUSES:
            raise ConflictError("当前运行状态不允许重试，请先暂停或等待完成")
        item = db.get(ChunkEvaluationItem, item_id)
        if item is None or item.run_id != run.id:
            raise NotFoundError("评估 chunk 不存在")
        if item.status not in {EvaluationItemStatus.FAILED, EvaluationItemStatus.INTERRUPTED}:
            raise ConflictError("仅失败或中断的 chunk 可以重试")
        self._reset_item_for_work(run, item)
        self._prepare_for_execution(run)
        self._update_progress(db, run)
        db.commit()
        db.refresh(run)
        return run

    def retry_failed(
        self,
        db: Session,
        run_id: str,
        request: ChunkEvaluationRetryFailed,
    ) -> ChunkEvaluationRun:
        run = self._require_run(db, run_id)
        if run.status not in EDITABLE_STATUSES:
            raise ConflictError("当前运行状态不允许重试，请先暂停或等待完成")
        filters = [
            ChunkEvaluationItem.run_id == run.id,
            ChunkEvaluationItem.status.in_(
                [EvaluationItemStatus.FAILED, EvaluationItemStatus.INTERRUPTED]
            ),
        ]
        if request.document_id:
            filters.append(ChunkEvaluationItem.document_id == request.document_id)
        items = list(db.scalars(select(ChunkEvaluationItem).where(*filters)))
        if not items:
            raise ValidationError("没有可重试的失败 chunk")
        for item in items:
            self._reset_item_for_work(run, item)
        self._prepare_for_execution(run)
        self._update_progress(db, run)
        db.commit()
        db.refresh(run)
        return run

    @staticmethod
    def _reset_item_for_work(run: ChunkEvaluationRun, item: ChunkEvaluationItem) -> None:
        item.error = None
        item.llm_succeeded = False
        item.fallback_applied = False
        item.finished_at = None
        if item.rule_succeeded and item.result_id:
            item.status = (
                EvaluationItemStatus.LLM_QUEUED
                if run.llm_enabled
                else EvaluationItemStatus.SUCCEEDED
            )
        else:
            item.status = EvaluationItemStatus.QUEUED

    @staticmethod
    def _prepare_for_execution(run: ChunkEvaluationRun) -> None:
        run.status = EvaluationStatus.QUEUED
        run.stage = EvaluationStage.QUEUED
        run.pause_requested_at = None
        run.paused_at = None
        run.finished_at = None
        run.errors = [
            error for error in (run.errors or []) if str(error.get("scope")) != "llm"
        ]

    def pause_run(self, db: Session, run_id: str) -> ChunkEvaluationRun:
        run = self._require_run(db, run_id)
        if run.status in {EvaluationStatus.PAUSING, EvaluationStatus.PAUSED}:
            return run
        if run.status not in PAUSABLE_STATUSES:
            raise ConflictError("当前运行不可暂停")
        now = datetime.now(UTC)
        run.pause_requested_at = now
        if run.status == EvaluationStatus.QUEUED or (
            run.status == EvaluationStatus.RUNNING and not run.celery_task_id
        ):
            run.status = EvaluationStatus.PAUSED
            run.stage = EvaluationStage.PAUSED
            run.paused_at = now
        else:
            run.status = EvaluationStatus.PAUSING
            run.stage = EvaluationStage.PAUSING
        self._update_progress(db, run)
        db.commit()
        db.refresh(run)
        return run

    def resume_run(self, db: Session, run_id: str) -> ChunkEvaluationRun:
        run = self._require_run(db, run_id)
        if run.status != EvaluationStatus.PAUSED:
            raise ConflictError("仅暂停的评估可以继续")
        items = list(
            db.scalars(
                select(ChunkEvaluationItem).where(
                    ChunkEvaluationItem.run_id == run.id,
                    ChunkEvaluationItem.status == EvaluationItemStatus.INTERRUPTED,
                )
            )
        )
        if not items:
            pending = db.scalar(
                select(func.count())
                .select_from(ChunkEvaluationItem)
                .where(
                    ChunkEvaluationItem.run_id == run.id,
                    ChunkEvaluationItem.status.in_(
                        [EvaluationItemStatus.QUEUED, EvaluationItemStatus.LLM_QUEUED]
                    ),
                )
            )
            if not pending:
                raise ValidationError("没有待继续处理的 chunk")
        for item in items:
            self._reset_item_for_work(run, item)
        self._prepare_for_execution(run)
        self._update_progress(db, run)
        db.commit()
        db.refresh(run)
        return run

    @staticmethod
    def _require_run(db: Session, run_id: str) -> ChunkEvaluationRun:
        run = db.get(ChunkEvaluationRun, run_id)
        if run is None:
            raise NotFoundError("评估运行不存在")
        return run
    @staticmethod
    def _select_review_samples(db: Session, run: ChunkEvaluationRun) -> None:
        results = list(
            db.scalars(
                select(ChunkEvaluationResult)
                .where(ChunkEvaluationResult.run_id == run.id)
                .order_by(ChunkEvaluationResult.chunk_index.asc())
            )
        )
        grouped: dict[str, dict[str, list[ChunkEvaluationResult]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for result in results:
            grouped[result.document_id or ""][result.granularity_level].append(result)
        selected: list[ChunkEvaluationResult] = []
        for document_id, by_level in grouped.items():
            rng = random.Random(f"{run.id}:{document_id}")
            document_selected: list[ChunkEvaluationResult] = []
            for level in sorted(by_level):
                items = by_level[level]
                rng.shuffle(items)
                document_selected.extend(items[: min(REVIEW_PER_LEVEL_MIN, len(items))])
            document_results = [item for items in by_level.values() for item in items]
            remaining = [item for item in document_results if item not in document_selected]
            rng.shuffle(remaining)
            slots = max(0, REVIEW_SAMPLE_TARGET - len(document_selected))
            document_selected.extend(remaining[:slots])
            selected.extend(document_selected)
        selected_ids = {item.id for item in selected}
        for result in results:
            result.review_sample = result.id in selected_ids
        db.commit()

    def _aggregate(self, db: Session, run: ChunkEvaluationRun) -> dict:
        results = list(
            db.scalars(
                select(ChunkEvaluationResult).where(ChunkEvaluationResult.run_id == run.id)
            )
        )
        if not results:
            return {"result_count": 0, "level_scores": {}, "overall_score": 0}
        by_level: dict[str, list[ChunkEvaluationResult]] = defaultdict(list)
        issue_counts: dict[str, int] = defaultdict(int)
        grade_counts: dict[str, int] = defaultdict(int)
        for result in results:
            by_level[result.granularity_level].append(result)
            grade_counts[result.grade] += 1
            for issue in result.issue_codes:
                issue_counts[str(issue)] += 1
        level_scores = {
            level: round(sum(item.composite_score for item in items) / len(items), 2)
            for level, items in by_level.items()
        }
        overall = round(sum(level_scores.values()) / len(level_scores), 2)
        reviewed = [item for item in results if item.human_score is not None]
        calibration = self._calibration(reviewed)
        return {
            "result_count": len(results),
            "overall_score": overall,
            "grade": "pass" if overall >= 80 else "warn" if overall >= 60 else "fail",
            "level_scores": level_scores,
            "grade_counts": dict(grade_counts),
            "issue_counts": dict(sorted(issue_counts.items(), key=lambda item: -item[1])),
            "review_sample_count": sum(1 for item in results if item.review_sample),
            "reviewed_count": len(reviewed),
            "calibration": calibration,
        }

    @staticmethod
    def _calibration(results: list[ChunkEvaluationResult]) -> dict:
        if not results:
            return {"sample_count": 0, "mae": None, "correlation": None}
        pairs = [
            (float(item.composite_score), float(item.human_score or 0) * 20)
            for item in results
        ]
        mae = sum(abs(auto - human) for auto, human in pairs) / len(pairs)
        correlation = None
        if len(pairs) >= 2:
            auto_ranks = _ranks([item[0] for item in pairs])
            human_ranks = _ranks([item[1] for item in pairs])
            correlation = round(_pearson(auto_ranks, human_ranks), 4)
        return {
            "sample_count": len(pairs),
            "mae": round(mae, 2),
            "correlation": correlation,
        }

    def update_review(
        self,
        db: Session,
        run_id: str,
        result_id: str,
        request: ChunkEvaluationReviewUpdate,
    ) -> ChunkEvaluationRun:
        run = db.get(ChunkEvaluationRun, run_id)
        if run is None:
            raise NotFoundError("评估运行不存在")
        result = db.get(ChunkEvaluationResult, result_id)
        if result is None or result.run_id != run_id:
            raise NotFoundError("评估结果不存在")
        result.human_score = request.human_score
        result.human_issue_codes = request.human_issue_codes
        result.human_comment = request.human_comment
        result.reviewed_at = datetime.now(UTC)
        db.flush()
        run.aggregate = self._aggregate(db, run)
        db.commit()
        db.refresh(run)
        return run

    def compare(self, db: Session, run_ids: list[str]) -> dict:
        if len(run_ids) != 2 or len(set(run_ids)) != 2:
            raise ValidationError("比较必须提供两个不同的 run_id")
        runs = [db.get(ChunkEvaluationRun, run_id) for run_id in run_ids]
        if any(run is None for run in runs):
            raise NotFoundError("评估运行不存在")
        base, compare = runs
        base_results = self._result_map(db, base.id)
        compare_results = self._result_map(db, compare.id)
        regressions = []
        for key, base_result in base_results.items():
            compared = compare_results.get(key)
            if compared is None:
                continue
            delta = round(compared.composite_score - base_result.composite_score, 2)
            if delta < 0:
                regressions.append(
                    {
                        "key": key,
                        "chunk_id": compared.chunk_id,
                        "granularity_level": compared.granularity_level,
                        "section_path": compared.section_path,
                        "delta": delta,
                        "base_score": base_result.composite_score,
                        "compare_score": compared.composite_score,
                    }
                )
        regressions.sort(key=lambda item: item["delta"])
        base_levels = dict((base.aggregate or {}).get("level_scores") or {})
        compare_levels = dict((compare.aggregate or {}).get("level_scores") or {})
        base_issues = dict((base.aggregate or {}).get("issue_counts") or {})
        compare_issues = dict((compare.aggregate or {}).get("issue_counts") or {})
        return {
            "base_run": _run_summary(base),
            "compare_run": _run_summary(compare),
            "overall_delta": round(
                float((compare.aggregate or {}).get("overall_score", 0))
                - float((base.aggregate or {}).get("overall_score", 0)),
                2,
            ),
            "level_deltas": {
                level: round(
                    float(compare_levels.get(level, 0)) - float(base_levels.get(level, 0)),
                    2,
                )
                for level in sorted(set(base_levels) | set(compare_levels))
            },
            "issue_deltas": {
                issue: compare_issues.get(issue, 0) - base_issues.get(issue, 0)
                for issue in sorted(set(base_issues) | set(compare_issues))
            },
            "regressions": regressions[:10],
        }

    @staticmethod
    def _result_map(db: Session, run_id: str) -> dict[str, ChunkEvaluationResult]:
        results = list(
            db.scalars(
                select(ChunkEvaluationResult).where(ChunkEvaluationResult.run_id == run_id)
            )
        )
        return {
            f"{item.document_id}:{item.granularity_level}:{item.chunk_index}": item
            for item in results
        }


def _run_summary(run: ChunkEvaluationRun) -> dict:
    return {
        "id": run.id,
        "status": run.status,
        "stage": run.stage,
        "scope_type": run.scope_type,
        "knowledge_base_id": run.knowledge_base_id,
        "document_id": run.document_id,
        "granularities": list(run.granularities or []),
        "overall_score": float((run.aggregate or {}).get("overall_score", 0)),
        "level_scores": dict((run.aggregate or {}).get("level_scores") or {}),
        "created_at": run.created_at.isoformat(),
    }


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_var * right_var)
    return 0.0 if denominator == 0 else numerator / denominator