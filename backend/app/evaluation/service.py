import math
import random
import uuid
from collections import defaultdict
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError, ValidationError
from app.db.models import (
    ChunkEvaluationResult,
    ChunkEvaluationRun,
    Document,
    DocumentChunk,
    DocumentStatus,
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
from app.schemas import ChunkEvaluationCreate, ChunkEvaluationReviewUpdate

REVIEW_SAMPLE_TARGET = 24
REVIEW_PER_LEVEL_MIN = 4


class ChunkEvaluationService:
    def __init__(self, gateway: ModelGateway | None = None):
        self.gateway = gateway or ModelGateway()
        self.judge = LLMChunkJudge(self.gateway)

    def create_run(self, db: Session, request: ChunkEvaluationCreate) -> ChunkEvaluationRun:
        if request.document_id:
            document = db.get(Document, request.document_id)
            if document is None:
                raise NotFoundError("文档不存在")
            if document.status != DocumentStatus.READY:
                raise ValidationError("文档尚未完成入库")
            knowledge_base_id = document.kb_id
            scope_type = "document"
        else:
            knowledge_base = db.get(KnowledgeBase, request.knowledge_base_id)
            if knowledge_base is None:
                raise NotFoundError("知识库不存在")
            knowledge_base_id = knowledge_base.id
            scope_type = "knowledge_base"
        if request.judge_profile_id:
            self._validate_judge_profile(db, request.judge_profile_id)
        run = ChunkEvaluationRun(
            id=str(uuid.uuid4()),
            scope_type=scope_type,
            knowledge_base_id=knowledge_base_id,
            document_id=request.document_id,
            granularities=request.granularities,
            judge_profile_id=request.judge_profile_id,
            llm_enabled=request.llm_enabled,
            status=EvaluationStatus.QUEUED,
            progress={"total_documents": 0, "processed_documents": 0, "evaluated_chunks": 0},
            config_snapshot={
                "granularities": request.granularities,
                "llm_enabled": request.llm_enabled,
                "judge_profile_id": request.judge_profile_id,
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

    def _judge_profile(self, db: Session, run: ChunkEvaluationRun) -> ModelProfile | None:
        if run.judge_profile_id:
            return self._validate_judge_profile(db, run.judge_profile_id)
        knowledge_base = db.get(KnowledgeBase, run.knowledge_base_id)
        if knowledge_base is None:
            return None
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

    async def run(self, db: Session, run_id: str) -> None:
        run = db.get(ChunkEvaluationRun, run_id)
        if run is None:
            return
        run.status = EvaluationStatus.RUNNING
        run.started_at = datetime.now(UTC)
        db.commit()
        errors: list[dict] = []
        llm_error_count = 0
        try:
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
            run.config_snapshot = config
            run.progress = {
                "total_documents": len(documents),
                "processed_documents": 0,
                "evaluated_chunks": 0,
                "llm_failed_chunks": 0,
            }
            db.commit()
            llm_profile = self._judge_profile(db, run) if run.llm_enabled else None
            if run.llm_enabled and llm_profile is None:
                errors.append({"scope": "run", "error": "未配置可用的 LLM 评判模型，已降级为规则评估"})

            for document in documents:
                try:
                    doc_llm_errors, failed_chunks = await self._evaluate_document(
                        db, run, document, llm_profile
                    )
                    errors.extend(
                        {
                            "scope": "llm",
                            "document_id": document.id,
                            "error": error,
                        }
                        for error in doc_llm_errors
                    )
                    llm_error_count += failed_chunks
                    run.progress = {
                        **dict(run.progress or {}),
                        "processed_documents": int((run.progress or {}).get("processed_documents", 0)) + 1,
                        "evaluated_chunks": int((run.progress or {}).get("evaluated_chunks", 0)),
                        "llm_failed_chunks": llm_error_count,
                    }
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    run = db.get(ChunkEvaluationRun, run_id)
                    if run is None:
                        return
                    errors.append({"document_id": document.id, "error": str(exc)[:1000]})
                    run.progress = {
                        **dict(run.progress or {}),
                        "processed_documents": int((run.progress or {}).get("processed_documents", 0)) + 1,
                    }
                    db.commit()

            run = db.get(ChunkEvaluationRun, run_id)
            if run is None:
                return
            self._select_review_samples(db, run)
            run.aggregate = self._aggregate(db, run)
            run.errors = errors
            if errors or (run.llm_enabled and llm_error_count):
                run.status = EvaluationStatus.PARTIAL
            else:
                run.status = EvaluationStatus.COMPLETED
            if not run.aggregate.get("result_count"):
                run.status = EvaluationStatus.FAILED
                errors.append({"scope": "run", "error": "没有可评估的 chunk"})
                run.errors = errors
                run.aggregate = self._aggregate(db, run)
            run.finished_at = datetime.now(UTC)
            db.commit()
        except Exception as exc:
            db.rollback()
            run = db.get(ChunkEvaluationRun, run_id)
            if run:
                run.status = EvaluationStatus.FAILED
                run.errors = [*errors, {"scope": "run", "error": str(exc)[:1000]}]
                run.finished_at = datetime.now(UTC)
                db.commit()

    def _documents_for_run(
        self, db: Session, run: ChunkEvaluationRun
    ) -> list[Document]:
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

    async def _evaluate_document(
        self,
        db: Session,
        run: ChunkEvaluationRun,
        document: Document,
        llm_profile: ModelProfile | None,
    ) -> tuple[list[str], int]:
        chunks = list(
            db.scalars(
                select(DocumentChunk)
                .where(
                    DocumentChunk.document_id == document.id,
                    DocumentChunk.granularity_level.in_(run.granularities),
                )
                .order_by(DocumentChunk.chunk_index.asc())
            )
        )
        if not chunks:
            return [], 0
        all_l2 = list(
            db.scalars(
                select(DocumentChunk)
                .where(
                    DocumentChunk.document_id == document.id,
                    DocumentChunk.granularity_level == "L2",
                )
                .order_by(DocumentChunk.chunk_index.asc())
            )
        )
        l2_content = [chunk.content for chunk in all_l2]
        units = source_units(document.extracted_text)
        unit_chunks: dict[str, set[str]] = defaultdict(set)
        for chunk in all_l2:
            normalized = normalize_text(chunk.content)
            for unit in units:
                normalized_unit = normalize_text(unit)
                if normalized_unit and normalized_unit in normalized:
                    unit_chunks[normalized_unit].add(chunk.id)
        duplicate_units = {unit for unit, owners in unit_chunks.items() if len(owners) > 1}
        all_chunks = list(
            db.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document.id)
                .order_by(DocumentChunk.chunk_index.asc())
            )
        )
        chunks_by_id = {chunk.id: chunk for chunk in all_chunks}
        document_metadata = document.metadata_json or {}
        has_pages = bool(document_metadata.get("pages") or document_metadata.get("page_count"))

        samples: list[EvaluationSample] = []
        rule_by_chunk: dict[str, dict] = {}
        for chunk in chunks:
            context = self._context_snapshot(chunk, all_chunks, chunks_by_id)
            rule_metrics = evaluate_rules(
                chunk,
                source_text=(
                    str(context.get("section_source", ""))
                    if chunk.granularity_level == "L1"
                    else document.extracted_text
                ),
                source_unit_list=units,
                all_l2_content=l2_content,
                duplicate_units=duplicate_units,
                has_pages=has_pages,
                previous_content=str(context.get("previous_excerpt", "")),
                next_content=str(context.get("next_excerpt", "")),
            )
            rule_by_chunk[chunk.id] = rule_metrics
            samples.append(EvaluationSample(chunk=chunk, context=context, source_units=units))

        llm_results: dict[str, dict] = {}
        llm_errors: list[str] = []
        failed_chunks = 0
        if run.llm_enabled:
            if llm_profile is None:
                failed_chunks = len(samples)
                llm_errors.append("未配置可用的 LLM 评判模型，已降级为规则评估")
            else:
                llm_results, batch_errors = await self.judge.evaluate(llm_profile, samples)
                missing = sum(
                    1 for sample in samples if sample.chunk.id not in llm_results
                )
                failed_chunks = missing
                llm_errors.extend(batch_errors)
                if missing:
                    llm_errors.append(f"{missing} 个 chunk 未获得 LLM 评判结果")

        for sample in samples:
            chunk = sample.chunk
            llm_scores = llm_results.get(chunk.id, {})
            composite, grade, issues, suggestions, components = combine_scores(
                chunk.granularity_level,
                rule_by_chunk[chunk.id],
                llm_scores,
                llm_available=bool(llm_scores),
            )
            db.add(
                ChunkEvaluationResult(
                    id=str(uuid.uuid4()),
                    run_id=run.id,
                    chunk_id=chunk.id,
                    document_id=document.id,
                    kb_id=document.kb_id,
                    granularity_level=chunk.granularity_level,
                    granule_type=chunk.granule_type,
                    chunk_index=chunk.chunk_index,
                    content_snapshot=chunk.content,
                    section_path=chunk.section_path,
                    context_prefix=chunk.context_prefix or "",
                    parent_chunk_id=chunk.parent_chunk_id,
                    metadata_snapshot={
                        **dict(chunk.metadata_json or {}),
                        "token_count": chunk.token_count,
                        "summary": chunk.summary,
                        "keywords": list(chunk.keywords or []),
                        "qa_pairs": list(chunk.qa_pairs or []),
                    },
                    context_snapshot={
                        **sample.context,
                        "component_scores": components,
                        "missing_facts": llm_scores.get("missing_facts") or [],
                        "confidence": llm_scores.get("confidence"),
                    },
                    rule_metrics=rule_by_chunk[chunk.id],
                    llm_scores=llm_scores,
                    llm_reason=str(llm_scores.get("reason") or ""),
                    issue_codes=issues,
                    suggestions=suggestions,
                    composite_score=composite,
                    grade=grade,
                )
            )
        db.flush()
        run.progress = {
            **dict(run.progress or {}),
            "evaluated_chunks": int((run.progress or {}).get("evaluated_chunks", 0)) + len(samples),
        }
        db.commit()
        return llm_errors, failed_chunks

    def _context_snapshot(
        self,
        chunk: DocumentChunk,
        all_chunks: list[DocumentChunk],
        chunks_by_id: dict[str, DocumentChunk],
    ) -> dict:
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
