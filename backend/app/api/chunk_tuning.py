import io
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.security import get_current_admin
from app.db.models import (
    ChunkTuningCaseResult,
    ChunkTuningRun,
    ChunkTuningStatus,
    ChunkTuningTrial,
    Document,
    KnowledgeBase,
    ModelProfile,
    RetrievalEvaluationDataset,
)
from app.db.session import get_db
from app.schemas import (
    ChunkSettingsReindexRequest,
    ChunkSettingsReindexResponse,
    ChunkTuningApplyRequest,
    ChunkTuningCaseResultRead,
    ChunkTuningCaseResultsPage,
    ChunkTuningRunCreate,
    ChunkTuningRunRead,
    ChunkTuningTrialRead,
    ChunkTuningVerificationCreate,
    KnowledgeBaseRead,
    RetrievalEvaluationCaseCreate,
    RetrievalEvaluationCaseRead,
    RetrievalEvaluationDatasetCreate,
    RetrievalEvaluationDatasetRead,
    RetrievalEvaluationDatasetUpdate,
)
from app.services.chunk_tuning_service import ChunkTuningService

router = APIRouter(tags=["chunk-tuning"])
service = ChunkTuningService()


def _dataset_read(dataset: RetrievalEvaluationDataset) -> RetrievalEvaluationDatasetRead:
    response = RetrievalEvaluationDatasetRead.model_validate(dataset)
    response.case_count = len(dataset.cases)
    response.answerable_count = sum(1 for case in dataset.cases if case.answerable)
    return response


def _kb_read(db: Session, kb: KnowledgeBase) -> KnowledgeBaseRead:
    document_count = db.scalar(
        select(func.count()).select_from(Document).where(Document.kb_id == kb.id)
    )
    embedding = db.get(ModelProfile, kb.embedding_profile_id)
    return KnowledgeBaseRead(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        embedding_profile_id=kb.embedding_profile_id,
        rerank_profile_id=kb.rerank_profile_id,
        enrichment_profile_id=kb.enrichment_profile_id,
        default_llm_profile_id=kb.default_llm_profile_id,
        settings=kb.settings,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
        document_count=int(document_count or 0),
        embedding_profile_name=embedding.name if embedding else None,
    )


def _enqueue_run(run: ChunkTuningRun, db: Session) -> None:
    try:
        from app.workers.celery_app import celery_app

        celery_app.send_task("zhiyuan.run_chunk_tuning", args=[run.id])
    except Exception as exc:
        run.status = ChunkTuningStatus.FAILED
        run.errors = [*list(run.errors or []), {"scope": "queue", "error": str(exc)[:1000]}]
        db.commit()
        raise ConflictError(f"任务队列不可用：{exc}") from exc


@router.get(
    "/retrieval-evaluation-datasets",
    response_model=list[RetrievalEvaluationDatasetRead],
)
def list_retrieval_evaluation_datasets(
    knowledge_base_id: str | None = None,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[RetrievalEvaluationDatasetRead]:
    return [_dataset_read(item) for item in service.list_datasets(db, knowledge_base_id)]


@router.post(
    "/retrieval-evaluation-datasets",
    response_model=RetrievalEvaluationDatasetRead,
    status_code=201,
)
def create_retrieval_evaluation_dataset(
    request: RetrievalEvaluationDatasetCreate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> RetrievalEvaluationDatasetRead:
    return _dataset_read(service.create_dataset(db, request))


@router.get(
    "/retrieval-evaluation-datasets/{dataset_id}",
    response_model=RetrievalEvaluationDatasetRead,
)
def get_retrieval_evaluation_dataset(
    dataset_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> RetrievalEvaluationDatasetRead:
    return _dataset_read(service.get_dataset(db, dataset_id))


@router.patch(
    "/retrieval-evaluation-datasets/{dataset_id}",
    response_model=RetrievalEvaluationDatasetRead,
)
def update_retrieval_evaluation_dataset(
    dataset_id: str,
    request: RetrievalEvaluationDatasetUpdate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> RetrievalEvaluationDatasetRead:
    return _dataset_read(service.update_dataset(db, dataset_id, request))


@router.delete("/retrieval-evaluation-datasets/{dataset_id}", status_code=204)
def delete_retrieval_evaluation_dataset(
    dataset_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Response:
    service.delete_dataset(db, dataset_id)
    return Response(status_code=204)


@router.post(
    "/retrieval-evaluation-datasets/{dataset_id}/cases",
    response_model=RetrievalEvaluationCaseRead,
    status_code=201,
)
def create_retrieval_evaluation_case(
    dataset_id: str,
    request: RetrievalEvaluationCaseCreate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> RetrievalEvaluationCaseRead:
    return RetrievalEvaluationCaseRead.model_validate(service.add_case(db, dataset_id, request))


@router.patch(
    "/retrieval-evaluation-datasets/{dataset_id}/cases/{case_id}",
    response_model=RetrievalEvaluationCaseRead,
)
def update_retrieval_evaluation_case(
    dataset_id: str,
    case_id: str,
    request: RetrievalEvaluationCaseCreate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> RetrievalEvaluationCaseRead:
    return RetrievalEvaluationCaseRead.model_validate(
        service.replace_case(db, dataset_id, case_id, request)
    )


@router.delete(
    "/retrieval-evaluation-datasets/{dataset_id}/cases/{case_id}",
    status_code=204,
)
def delete_retrieval_evaluation_case(
    dataset_id: str,
    case_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Response:
    service.delete_case(db, dataset_id, case_id)
    return Response(status_code=204)


@router.post(
    "/retrieval-evaluation-datasets/{dataset_id}/import",
    response_model=RetrievalEvaluationDatasetRead,
)
async def import_retrieval_evaluation_cases(
    dataset_id: str,
    file: Annotated[UploadFile, File()],
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> RetrievalEvaluationDatasetRead:
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise ValidationError("JSONL 文件不能超过 10 MB")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationError("JSONL 文件必须使用 UTF-8 编码") from exc
    return _dataset_read(service.import_jsonl(db, dataset_id, text))


@router.get("/retrieval-evaluation-datasets/{dataset_id}/export")
def export_retrieval_evaluation_cases(
    dataset_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    dataset = service.get_dataset(db, dataset_id)
    content = service.export_jsonl(db, dataset_id).encode("utf-8")
    filename = f"retrieval-evaluation-{dataset.id}.jsonl"
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/chunk-tuning-runs", response_model=ChunkTuningRunRead, status_code=201)
def create_chunk_tuning_run(
    request: ChunkTuningRunCreate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkTuningRunRead:
    run = service.create_screening_run(db, request)
    _enqueue_run(run, db)
    db.refresh(run)
    return ChunkTuningRunRead.model_validate(run)


@router.get("/chunk-tuning-runs", response_model=list[ChunkTuningRunRead])
def list_chunk_tuning_runs(
    knowledge_base_id: str | None = None,
    dataset_id: str | None = None,
    stage: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[ChunkTuningRunRead]:
    statement = select(ChunkTuningRun).order_by(ChunkTuningRun.created_at.desc())
    if knowledge_base_id:
        statement = statement.where(ChunkTuningRun.kb_id == knowledge_base_id)
    if dataset_id:
        statement = statement.where(ChunkTuningRun.dataset_id == dataset_id)
    if stage:
        statement = statement.where(ChunkTuningRun.stage == stage)
    rows = list(db.scalars(statement.limit(limit)))
    return [ChunkTuningRunRead.model_validate(row) for row in rows]


@router.get("/chunk-tuning-runs/{run_id}", response_model=ChunkTuningRunRead)
def get_chunk_tuning_run(
    run_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkTuningRunRead:
    run = db.get(ChunkTuningRun, run_id)
    if run is None:
        raise NotFoundError("调优运行不存在")
    return ChunkTuningRunRead.model_validate(run)


@router.get("/chunk-tuning-runs/{run_id}/trials", response_model=list[ChunkTuningTrialRead])
def list_chunk_tuning_trials(
    run_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[ChunkTuningTrialRead]:
    if db.get(ChunkTuningRun, run_id) is None:
        raise NotFoundError("调优运行不存在")
    rows = list(
        db.scalars(
            select(ChunkTuningTrial)
            .where(ChunkTuningTrial.run_id == run_id)
            .order_by(ChunkTuningTrial.created_at.asc())
        )
    )
    return [ChunkTuningTrialRead.model_validate(row) for row in rows]


@router.get(
    "/chunk-tuning-runs/{run_id}/results",
    response_model=ChunkTuningCaseResultsPage,
)
def list_chunk_tuning_results(
    run_id: str,
    trial_id: str | None = None,
    question_type: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkTuningCaseResultsPage:
    if db.get(ChunkTuningRun, run_id) is None:
        raise NotFoundError("调优运行不存在")
    trial_ids = [
        row
        for row in db.scalars(
            select(ChunkTuningTrial.id).where(ChunkTuningTrial.run_id == run_id)
        )
    ]
    if trial_id and trial_id not in trial_ids:
        raise NotFoundError("调优 trial 不存在")
    selected_ids = [trial_id] if trial_id else list(trial_ids)
    statement = select(ChunkTuningCaseResult).where(
        ChunkTuningCaseResult.trial_id.in_(selected_ids)
    )
    if question_type:
        statement = statement.where(ChunkTuningCaseResult.question_type == question_type)
    rows = list(
        db.scalars(
            statement.order_by(ChunkTuningCaseResult.recall_at_10.asc()).offset(
                (page - 1) * page_size
            ).limit(page_size)
        )
    )
    count_statement = select(func.count()).select_from(ChunkTuningCaseResult).where(
        ChunkTuningCaseResult.trial_id.in_(selected_ids)
    )
    if question_type:
        count_statement = count_statement.where(
            ChunkTuningCaseResult.question_type == question_type
        )
    total = int(db.scalar(count_statement) or 0)
    return ChunkTuningCaseResultsPage(
        items=[ChunkTuningCaseResultRead.model_validate(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post(
    "/chunk-tuning-runs/{run_id}/verifications",
    response_model=ChunkTuningRunRead,
    status_code=201,
)
def create_chunk_tuning_verification(
    run_id: str,
    request: ChunkTuningVerificationCreate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkTuningRunRead:
    run = service.create_verification_run(db, run_id, request)
    _enqueue_run(run, db)
    db.refresh(run)
    return ChunkTuningRunRead.model_validate(run)


@router.post("/chunk-tuning-runs/{run_id}/cancel", response_model=ChunkTuningRunRead)
def cancel_chunk_tuning_run(
    run_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkTuningRunRead:
    return ChunkTuningRunRead.model_validate(service.cancel_run(db, run_id))


@router.post("/chunk-tuning-runs/{run_id}/apply", response_model=KnowledgeBaseRead)
def apply_chunk_tuning_trial(
    run_id: str,
    request: ChunkTuningApplyRequest,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> KnowledgeBaseRead:
    return _kb_read(db, service.apply_trial(db, run_id, request.trial_id))


@router.delete("/chunk-tuning-runs/{run_id}", status_code=204)
def delete_chunk_tuning_run(
    run_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Response:
    service.delete_run(db, run_id)
    return Response(status_code=204)


@router.post(
    "/knowledge-bases/{kb_id}/chunk-settings/reindex",
    response_model=ChunkSettingsReindexResponse,
)
def reindex_documents_with_chunk_settings(
    kb_id: str,
    request: ChunkSettingsReindexRequest,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkSettingsReindexResponse:
    return service.batch_reindex(db, kb_id, request)
