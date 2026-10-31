from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError
from app.core.security import get_current_admin
from app.db.models import (
    ChunkEvaluationResult,
    ChunkEvaluationRun,
    EvaluationStatus,
)
from app.db.session import get_db
from app.evaluation.service import ChunkEvaluationService
from app.schemas import (
    ChunkEvaluationChunkSelect,
    ChunkEvaluationChunksPage,
    ChunkEvaluationCreate,
    ChunkEvaluationResultRead,
    ChunkEvaluationResultsPage,
    ChunkEvaluationRetryFailed,
    ChunkEvaluationReviewUpdate,
    ChunkEvaluationRunRead,
)

router = APIRouter(prefix="/chunk-evaluations", tags=["chunk-evaluations"])
service = ChunkEvaluationService()


def _enqueue_run(db: Session, run: ChunkEvaluationRun) -> None:
    try:
        from app.workers.celery_app import celery_app

        task = celery_app.send_task("zhiyuan.evaluate_chunks", args=[run.id])
        task_id = getattr(task, "id", None)
        if task_id:
            run.celery_task_id = str(task_id)
        db.commit()
        db.refresh(run)
    except Exception as exc:
        run.status = EvaluationStatus.FAILED
        run.stage = "failed"
        run.errors = [*list(run.errors or []), {"scope": "queue", "error": str(exc)[:1000]}]
        db.commit()
        raise ConflictError(f"任务队列不可用：{exc}") from exc


def _revoke_task(run: ChunkEvaluationRun) -> None:
    if not run.celery_task_id:
        return
    try:
        from app.workers.celery_app import celery_app

        celery_app.control.revoke(run.celery_task_id, terminate=False)
    except Exception:
        return


@router.post("", response_model=ChunkEvaluationRunRead, status_code=201)
def create_evaluation(
    request: ChunkEvaluationCreate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkEvaluationRunRead:
    run = service.create_run(db, request)
    _enqueue_run(db, run)
    return ChunkEvaluationRunRead.model_validate(run)


@router.get("", response_model=list[ChunkEvaluationRunRead])
def list_evaluations(
    document_id: str | None = None,
    knowledge_base_id: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[ChunkEvaluationRunRead]:
    statement = select(ChunkEvaluationRun).order_by(ChunkEvaluationRun.created_at.desc())
    if document_id:
        statement = statement.where(ChunkEvaluationRun.document_id == document_id)
    if knowledge_base_id:
        statement = statement.where(
            ChunkEvaluationRun.knowledge_base_id == knowledge_base_id
        )
    if status:
        statement = statement.where(ChunkEvaluationRun.status == status)
    runs = db.scalars(statement.limit(limit))
    return [ChunkEvaluationRunRead.model_validate(run) for run in runs]


@router.get("/compare")
def compare_evaluations(
    run_ids: str = Query(min_length=1),
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> dict:
    ids = [item.strip() for item in run_ids.split(",") if item.strip()]
    return service.compare(db, ids)


@router.get("/{run_id}", response_model=ChunkEvaluationRunRead)
def get_evaluation(
    run_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkEvaluationRunRead:
    run = db.get(ChunkEvaluationRun, run_id)
    if run is None:
        raise NotFoundError("评估运行不存在")
    return ChunkEvaluationRunRead.model_validate(run)


@router.post("/{run_id}/pause", response_model=ChunkEvaluationRunRead)
def pause_evaluation(
    run_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkEvaluationRunRead:
    existing = db.get(ChunkEvaluationRun, run_id)
    if existing is None:
        raise NotFoundError("评估运行不存在")
    was_queued = existing.status == EvaluationStatus.QUEUED
    run = service.pause_run(db, run_id)
    if was_queued:
        _revoke_task(run)
    return ChunkEvaluationRunRead.model_validate(run)


@router.post("/{run_id}/resume", response_model=ChunkEvaluationRunRead)
def resume_evaluation(
    run_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkEvaluationRunRead:
    run = service.resume_run(db, run_id)
    _enqueue_run(db, run)
    return ChunkEvaluationRunRead.model_validate(run)


@router.get("/{run_id}/chunks", response_model=ChunkEvaluationChunksPage)
def list_evaluation_chunks(
    run_id: str,
    document_id: str | None = None,
    granularity: str | None = None,
    status: str | None = None,
    search: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkEvaluationChunksPage:
    items, total = service.list_chunks(
        db,
        run_id,
        document_id=document_id,
        granularity=granularity,
        status=status,
        search=search,
        page=page,
        page_size=page_size,
    )
    return ChunkEvaluationChunksPage(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/{run_id}/chunks/select", response_model=ChunkEvaluationRunRead)
def select_evaluation_chunks(
    run_id: str,
    request: ChunkEvaluationChunkSelect,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkEvaluationRunRead:
    run = service.select_chunks(db, run_id, request.chunk_ids)
    _enqueue_run(db, run)
    return ChunkEvaluationRunRead.model_validate(run)


@router.post("/{run_id}/chunks/{item_id}/retry", response_model=ChunkEvaluationRunRead)
def retry_evaluation_chunk(
    run_id: str,
    item_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkEvaluationRunRead:
    run = service.retry_item(db, run_id, item_id)
    _enqueue_run(db, run)
    return ChunkEvaluationRunRead.model_validate(run)


@router.post("/{run_id}/retry-failed", response_model=ChunkEvaluationRunRead)
def retry_failed_evaluation_chunks(
    run_id: str,
    request: ChunkEvaluationRetryFailed | None = None,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkEvaluationRunRead:
    run = service.retry_failed(db, run_id, request or ChunkEvaluationRetryFailed())
    _enqueue_run(db, run)
    return ChunkEvaluationRunRead.model_validate(run)


@router.get("/{run_id}/results", response_model=ChunkEvaluationResultsPage)
def list_evaluation_results(
    run_id: str,
    granularity: str | None = None,
    grade: str | None = None,
    issue: str | None = None,
    review_sample: bool | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkEvaluationResultsPage:
    run = db.get(ChunkEvaluationRun, run_id)
    if run is None:
        raise NotFoundError("评估运行不存在")
    statement = (
        select(ChunkEvaluationResult)
        .where(ChunkEvaluationResult.run_id == run_id)
        .order_by(ChunkEvaluationResult.composite_score.asc())
    )
    if granularity:
        statement = statement.where(
            ChunkEvaluationResult.granularity_level == granularity
        )
    if grade:
        statement = statement.where(ChunkEvaluationResult.grade == grade)
    if review_sample is not None:
        statement = statement.where(ChunkEvaluationResult.review_sample == review_sample)
    rows = list(db.scalars(statement))
    if issue:
        rows = [row for row in rows if issue in (row.issue_codes or [])]
    total = len(rows)
    start = (page - 1) * page_size
    items = rows[start : start + page_size]
    return ChunkEvaluationResultsPage(
        items=[ChunkEvaluationResultRead.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.patch(
    "/{run_id}/results/{result_id}",
    response_model=ChunkEvaluationRunRead,
)
def review_evaluation_result(
    run_id: str,
    result_id: str,
    request: ChunkEvaluationReviewUpdate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkEvaluationRunRead:
    run = service.update_review(db, run_id, result_id, request)
    return ChunkEvaluationRunRead.model_validate(run)


@router.delete("/{run_id}", status_code=204)
def delete_evaluation(
    run_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Response:
    run = db.get(ChunkEvaluationRun, run_id)
    if run is None:
        raise NotFoundError("评估运行不存在")
    if not run.can_delete:
        raise ConflictError("运行中任务必须先暂停，暂停完成后才能删除")
    db.delete(run)
    db.commit()
    return Response(status_code=204)