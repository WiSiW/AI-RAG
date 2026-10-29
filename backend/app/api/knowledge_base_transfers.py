import tempfile
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.security import get_current_admin
from app.core.storage import delete_object, get_object_stream, upload_file
from app.db.models import (
    KnowledgeBaseTransferJob,
    ModelKind,
    TransferJobKind,
    TransferStatus,
)
from app.db.session import get_db
from app.schemas import (
    KnowledgeBaseExportRequest,
    KnowledgeBasePackageInspection,
    KnowledgeBaseTransferJobRead,
)
from app.services.knowledge_base_transfer_service import KnowledgeBaseTransferService

router = APIRouter(tags=["knowledge-base-transfers"])
service = KnowledgeBaseTransferService()


@router.post(
    "/knowledge-bases/{kb_id}/exports",
    response_model=KnowledgeBaseTransferJobRead,
    status_code=201,
)
def create_export(
    kb_id: str,
    request: KnowledgeBaseExportRequest,
    username: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> KnowledgeBaseTransferJobRead:
    job = service.create_export_job(db, kb_id, request.mode, username)
    try:
        from app.workers.celery_app import celery_app

        celery_app.send_task("zhiyuan.export_knowledge_base", args=[job.id])
    except Exception as exc:
        job.status = TransferStatus.FAILED
        job.errors = [{"error": f"任务队列不可用：{exc}"}]
        db.commit()
        raise ConflictError(f"任务队列不可用：{exc}") from exc
    return KnowledgeBaseTransferJobRead.model_validate(job)


@router.post(
    "/knowledge-base-imports/inspect",
    response_model=KnowledgeBasePackageInspection,
)
async def inspect_import_package(
    package: UploadFile = File(...),
    _: str = Depends(get_current_admin),
) -> KnowledgeBasePackageInspection:
    path = await _save_upload(package, settings.kb_package_max_bytes)
    try:
        inspection = service.inspect_package(path, package.filename or path.name)
        return KnowledgeBasePackageInspection(**inspection)
    finally:
        path.unlink(missing_ok=True)


@router.post(
    "/knowledge-base-imports",
    response_model=KnowledgeBaseTransferJobRead,
    status_code=201,
)
async def create_import(
    package: UploadFile = File(...),
    embedding_profile_id: str = Form(...),
    enrichment_profile_id: str = Form(...),
    rerank_profile_id: str | None = Form(default=None),
    llm_profile_id: str | None = Form(default=None),
    name: str = Form(default=""),
    index_policy: str = Form(default="strict"),
    username: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> KnowledgeBaseTransferJobRead:
    if index_policy not in {"strict", "rebuild"}:
        raise ValidationError("index_policy 必须为 strict 或 rebuild")
    service._profile(db, embedding_profile_id, ModelKind.EMBEDDING)
    service._profile(db, enrichment_profile_id, ModelKind.ENRICHMENT)
    service._optional_profile(db, rerank_profile_id, ModelKind.RERANK)
    service._optional_profile(db, llm_profile_id, ModelKind.LLM)
    path = await _save_upload(package, settings.kb_package_max_bytes)
    object_key: str | None = None
    try:
        inspection = service.inspect_package(path, package.filename or path.name)
        manifest = inspection["manifest"]
        object_key = f"imports/{uuid.uuid4().hex}/package.zip"
        upload_file(object_key, path, "application/zip")
        job = service.create_import_job(
            db,
            object_key=object_key,
            package_name=package.filename or path.name,
            created_by=username,
            request_options={
                "embedding_profile_id": embedding_profile_id,
                "enrichment_profile_id": enrichment_profile_id,
                "rerank_profile_id": rerank_profile_id,
                "llm_profile_id": llm_profile_id,
                "name": name.strip(),
                "index_policy": index_policy,
                "mode": manifest["mode"],
            },
        )
        try:
            from app.workers.celery_app import celery_app

            celery_app.send_task("zhiyuan.import_knowledge_base", args=[job.id])
        except Exception as exc:
            job.status = TransferStatus.FAILED
            job.errors = [{"error": f"任务队列不可用：{exc}"}]
            db.commit()
            if object_key:
                delete_object(object_key)
            raise ConflictError(f"任务队列不可用：{exc}") from exc
        return KnowledgeBaseTransferJobRead.model_validate(job)
    except Exception as exc:
        if object_key:
            delete_object(object_key)
        if isinstance(exc, (ValidationError, ConflictError, NotFoundError)):
            raise
        raise ConflictError(f"创建导入任务失败：{exc}") from exc
    finally:
        path.unlink(missing_ok=True)


@router.get(
    "/knowledge-base-transfers/{job_id}",
    response_model=KnowledgeBaseTransferJobRead,
)
def get_transfer_job(
    job_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> KnowledgeBaseTransferJobRead:
    job = db.get(KnowledgeBaseTransferJob, job_id)
    if job is None:
        raise NotFoundError("传输任务不存在")
    return KnowledgeBaseTransferJobRead.model_validate(job)


@router.get("/knowledge-base-transfers/{job_id}/download")
def download_transfer_package(
    job_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    job = db.get(KnowledgeBaseTransferJob, job_id)
    if job is None:
        raise NotFoundError("传输任务不存在")
    if job.kind != TransferJobKind.EXPORT or job.status != TransferStatus.COMPLETED:
        raise ConflictError("导出包尚未生成")
    if not job.object_key:
        raise NotFoundError("导出包不存在")
    response = get_object_stream(job.object_key)

    def iterator():
        try:
            while chunk := response.read(1024 * 1024):
                yield chunk
        finally:
            response.close()
            response.release_conn()

    filename = quote(job.package_name)
    return StreamingResponse(
        iterator(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )


@router.delete("/knowledge-base-transfers/{job_id}", status_code=204)
def delete_transfer_job(
    job_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> None:
    job = db.get(KnowledgeBaseTransferJob, job_id)
    if job is None:
        raise NotFoundError("传输任务不存在")
    if job.status in {TransferStatus.QUEUED, TransferStatus.RUNNING}:
        raise ConflictError("任务尚未结束，无法删除")
    if job.object_key:
        delete_object(job.object_key)
    db.delete(job)
    db.commit()


async def _save_upload(upload: UploadFile, max_bytes: int) -> Path:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    path = Path(handle.name)
    size = 0
    try:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise ValidationError("知识库包超过大小限制")
            handle.write(chunk)
        handle.flush()
        return path
    except Exception:
        handle.close()
        path.unlink(missing_ok=True)
        raise
    finally:
        handle.close()
