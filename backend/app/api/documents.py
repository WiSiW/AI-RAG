import asyncio
import logging
import shutil
import tarfile
import tempfile
import uuid
import zipfile
from collections.abc import Generator
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.chunking.modes import validate_chunk_mode
from app.core.config import settings
from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.security import get_current_admin
from app.core.storage import (
    build_object_key,
    delete_object,
    get_object_stream,
    upload_file,
)
from app.db.models import (
    Document,
    DocumentChunk,
    DocumentStatus,
    DocumentTable,
    IngestionJob,
    JobStatus,
    KnowledgeBase,
)
from app.db.session import get_db
from app.indexing.bm25_store import BM25Store
from app.indexing.vector_store import MilvusVectorStore
from app.parsing.registry import SUPPORTED_EXTENSIONS
from app.schemas import (
    BatchUploadItem,
    BatchUploadResponse,
    ChunkRead,
    DocumentDetail,
    DocumentRead,
    DocumentTableRead,
    IngestionJobRead,
    ReparsePdfResponse,
    ServerArchiveRequest,
    UploadResponse,
)
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)
router = APIRouter(tags=["documents"])
vector_store = MilvusVectorStore()
bm25_store = BM25Store()
ARCHIVE_EXTENSIONS = (".zip", ".tar", ".tgz", ".tar.gz")
MAX_ARCHIVE_ENTRIES = 2000


def _document_read(document: Document, chunk_count: int = 0) -> DocumentRead:
    return DocumentRead(
        id=document.id,
        kb_id=document.kb_id,
        filename=document.filename,
        content_type=document.content_type,
        size_bytes=document.size_bytes,
        status=document.status,
        error=document.error,
        tags=document.tags or [],
        metadata_json=document.metadata_json or {},
        chunk_mode=document.chunk_mode or "structure",
        chunk_size=document.chunk_size,
        chunk_overlap=document.chunk_overlap,
        created_at=document.created_at,
        updated_at=document.updated_at,
        chunk_count=chunk_count,
    )


def _job_read(job: IngestionJob) -> IngestionJobRead:
    return IngestionJobRead.model_validate(job)


def _enqueue(document: Document, job: IngestionJob) -> bool:
    try:
        queue = "pdf_parse" if document.filename.lower().endswith(".pdf") else "celery"
        celery_app.send_task(
            "zhiyuan.ingest_document",
            args=[document.id, job.id],
            task_id=job.id,
            queue=queue,
        )
        return True
    except Exception as exc:
        logger.exception("failed to enqueue ingestion task")
        job.status = JobStatus.FAILED
        job.error = f"任务队列不可用：{exc}"[:4000]
        document.status = DocumentStatus.FAILED
        document.error = job.error
        return False


def _get_kb(db: Session, kb_id: str) -> KnowledgeBase:
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise NotFoundError("知识库不存在")
    return kb


def _ensure_kb_writable(db: Session, kb_id: str) -> None:
    from app.services.knowledge_base_transfer_service import KnowledgeBaseTransferService

    KnowledgeBaseTransferService.ensure_kb_writable(db, kb_id)


def _upload_options(
    kb: KnowledgeBase,
    chunk_size: int | None,
    chunk_overlap: int | None,
    chunk_mode: str | None,
) -> tuple[int, int, str]:
    kb_settings = kb.settings or {}
    mode = chunk_mode or str(kb_settings.get("chunk_mode", "structure"))
    try:
        validate_chunk_mode(mode)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    actual_size = chunk_size or int(kb_settings.get("chunk_size", 512))
    actual_overlap = (
        chunk_overlap
        if chunk_overlap is not None
        else int(kb_settings.get("chunk_overlap", 64))
    )
    if not 32 <= actual_size <= 8192:
        raise ValidationError("chunk_size 必须在 32 到 8192 之间")
    if not 0 <= actual_overlap < actual_size:
        raise ValidationError("chunk_overlap 必须大于等于 0 且小于 chunk_size")
    return actual_size, actual_overlap, mode


def _parse_tags(tags: str) -> list[str]:
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def _validate_filename(filename: str) -> tuple[str, str]:
    safe_name = Path(filename or "document").name
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValidationError(f"不支持的文件格式，支持：{', '.join(SUPPORTED_EXTENSIONS)}")
    return safe_name, extension


async def _save_upload(upload: UploadFile, max_bytes: int, suffix: str = "") -> tuple[Path, int]:
    upload_dir = Path(settings.upload_temp_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_suffix = suffix or Path(upload.filename or "").suffix
    handle = tempfile.NamedTemporaryFile(delete=False, dir=upload_dir, suffix=file_suffix)
    path = Path(handle.name)
    size = 0
    try:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise ValidationError(f"文件不能超过 {max_bytes // (1024 * 1024)} MB")
            handle.write(chunk)
        handle.flush()
        if size == 0:
            raise ValidationError("上传文件为空")
        return path, size
    except Exception:
        handle.close()
        path.unlink(missing_ok=True)
        raise
    finally:
        handle.close()


async def _make_document(
    db: Session,
    kb: KnowledgeBase,
    *,
    filename: str,
    local_path: Path,
    content_type: str,
    size_bytes: int,
    tags: list[str],
    chunk_size: int,
    chunk_overlap: int,
    chunk_mode: str,
    extra_metadata: dict | None = None,
) -> UploadResponse:
    safe_name, _ = _validate_filename(filename)
    object_key = build_object_key(safe_name)
    await asyncio.to_thread(
        upload_file, object_key, local_path, content_type or "application/octet-stream"
    )
    document = Document(
        id=str(uuid.uuid4()),
        kb_id=kb.id,
        filename=safe_name,
        object_key=object_key,
        content_type=content_type or "application/octet-stream",
        size_bytes=size_bytes,
        status=DocumentStatus.PENDING,
        tags=tags,
        metadata_json={
            **(extra_metadata or {}),
            "upload_source": (extra_metadata or {}).get("upload_source", "browser"),
        },
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    job = IngestionJob(
        id=str(uuid.uuid4()),
        document_id=document.id,
        status=JobStatus.PENDING,
    )
    db.add_all([document, job])
    db.commit()
    db.refresh(document)
    db.refresh(job)
    _enqueue(document, job)
    db.commit()
    db.refresh(document)
    db.refresh(job)
    return UploadResponse(document=_document_read(document), job=_job_read(job))


async def _register_upload(
    db: Session,
    kb: KnowledgeBase,
    upload: UploadFile,
    *,
    chunk_size: int | None,
    chunk_overlap: int | None,
    chunk_mode: str | None,
    tags: str,
    extra_metadata: dict | None = None,
) -> UploadResponse:
    _ensure_kb_writable(db, kb.id)
    safe_name, _ = _validate_filename(upload.filename or "document")
    actual_size, actual_overlap, actual_mode = _upload_options(
        kb, chunk_size, chunk_overlap, chunk_mode
    )
    local_path, size_bytes = await _save_upload(upload, settings.upload_max_bytes)
    try:
        return await _make_document(
            db,
            kb,
            filename=safe_name,
            local_path=local_path,
            content_type=upload.content_type or "application/octet-stream",
            size_bytes=size_bytes,
            tags=_parse_tags(tags),
            chunk_size=actual_size,
            chunk_overlap=actual_overlap,
            chunk_mode=actual_mode,
            extra_metadata=extra_metadata,
        )
    finally:
        local_path.unlink(missing_ok=True)


async def _register_local_file(
    db: Session,
    kb: KnowledgeBase,
    *,
    filename: str,
    local_path: Path,
    content_type: str,
    chunk_size: int | None,
    chunk_overlap: int | None,
    chunk_mode: str | None,
    tags: str,
    extra_metadata: dict | None = None,
) -> UploadResponse:
    _ensure_kb_writable(db, kb.id)
    safe_name, _ = _validate_filename(filename)
    size_bytes = local_path.stat().st_size
    if size_bytes > settings.upload_max_bytes:
        raise ValidationError(f"文件不能超过 {settings.upload_max_mb} MB")
    actual_size, actual_overlap, actual_mode = _upload_options(
        kb, chunk_size, chunk_overlap, chunk_mode
    )
    return await _make_document(
        db,
        kb,
        filename=safe_name,
        local_path=local_path,
        content_type=content_type,
        size_bytes=size_bytes,
        tags=_parse_tags(tags),
        chunk_size=actual_size,
        chunk_overlap=actual_overlap,
        chunk_mode=actual_mode,
        extra_metadata=extra_metadata,
    )


def _archive_kind(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".zip"):
        return "zip"
    if lower.endswith((".tar", ".tar.gz", ".tgz")):
        return "tar"
    raise ValidationError("仅支持 ZIP、TAR、TAR.GZ 压缩包")


def _extract_archive(path: Path) -> tuple[list[tuple[str, Path, int]], list[str]]:
    kind = _archive_kind(path)
    root = Path(settings.upload_temp_dir) / f"extract-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, Path, int]] = []
    errors: list[str] = []
    total_size = 0
    try:
        if kind == "zip":
            with zipfile.ZipFile(path) as archive:
                infos = [info for info in archive.infolist() if not info.is_dir()]
                if len(infos) > MAX_ARCHIVE_ENTRIES:
                    raise ValidationError(f"压缩包文件数量不能超过 {MAX_ARCHIVE_ENTRIES}")
                for index, info in enumerate(infos):
                    member_name = info.filename.replace("\\", "/")
                    member_path = Path(member_name)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        errors.append(f"{member_name}: 非法路径")
                        continue
                    total_size += info.file_size
                    if total_size > settings.archive_max_bytes:
                        raise ValidationError(f"压缩包解压后不能超过 {settings.archive_max_mb} MB")
                    filename = member_path.name
                    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
                        errors.append(f"{member_name}: 不支持的格式")
                        continue
                    target = root / f"{index}-{Path(filename).name}"
                    with archive.open(info) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    entries.append((member_name, target, info.file_size))
        else:
            with tarfile.open(path, "r:*") as archive:
                members = [member for member in archive.getmembers() if member.isfile()]
                if len(members) > MAX_ARCHIVE_ENTRIES:
                    raise ValidationError(f"压缩包文件数量不能超过 {MAX_ARCHIVE_ENTRIES}")
                for index, member in enumerate(members):
                    member_name = member.name.replace("\\", "/")
                    member_path = Path(member_name)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        errors.append(f"{member_name}: 非法路径")
                        continue
                    total_size += member.size
                    if total_size > settings.archive_max_bytes:
                        raise ValidationError(f"压缩包解压后不能超过 {settings.archive_max_mb} MB")
                    filename = member_path.name
                    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
                        errors.append(f"{member_name}: 不支持的格式")
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        errors.append(f"{member_name}: 无法读取")
                        continue
                    target = root / f"{index}-{Path(filename).name}"
                    with source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    entries.append((member_name, target, member.size))
        if not entries:
            shutil.rmtree(root, ignore_errors=True)
        return entries, errors
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


async def _register_archive(
    db: Session,
    kb: KnowledgeBase,
    archive_path: Path,
    *,
    source_name: str,
    chunk_size: int | None,
    chunk_overlap: int | None,
    chunk_mode: str | None,
    tags: str,
    upload_source: str,
) -> BatchUploadResponse:
    entries, errors = await asyncio.to_thread(_extract_archive, archive_path)
    items: list[BatchUploadItem] = [
        BatchUploadItem(filename=error.split(":", 1)[0], ok=False, error=error)
        for error in errors
    ]
    extraction_root = entries[0][1].parent if entries else None
    try:
        for member_name, local_path, _ in entries:
            try:
                response = await _register_local_file(
                    db,
                    kb,
                    filename=Path(member_name).name,
                    local_path=local_path,
                    content_type="application/octet-stream",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chunk_mode=chunk_mode,
                    tags=tags,
                    extra_metadata={
                        "upload_source": upload_source,
                        "archive_name": source_name,
                        "archive_path": member_name,
                    },
                )
                items.append(
                    BatchUploadItem(
                        filename=member_name,
                        ok=True,
                        document=response.document,
                        job=response.job,
                    )
                )
            except Exception as exc:
                logger.exception("archive member registration failed name=%s", member_name)
                items.append(
                    BatchUploadItem(filename=member_name, ok=False, error=str(exc)[:1000])
                )
    finally:
        if extraction_root:
            shutil.rmtree(extraction_root, ignore_errors=True)
    accepted = sum(1 for item in items if item.ok)
    return BatchUploadResponse(items=items, accepted=accepted, failed=len(items) - accepted)


@router.post(
    "/knowledge-bases/{kb_id}/documents",
    response_model=UploadResponse,
    status_code=201,
)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    chunk_size: int | None = Form(default=None),
    chunk_overlap: int | None = Form(default=None),
    chunk_mode: str | None = Form(default=None),
    tags: str = Form(default=""),
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> UploadResponse:
    kb = _get_kb(db, kb_id)
    return await _register_upload(
        db,
        kb,
        file,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_mode=chunk_mode,
        tags=tags,
        extra_metadata={"upload_source": "browser"},
    )


@router.post(
    "/knowledge-bases/{kb_id}/documents/batch",
    response_model=BatchUploadResponse,
    status_code=201,
)
async def upload_documents_batch(
    kb_id: str,
    files: list[UploadFile] = File(...),
    chunk_size: int | None = Form(default=None),
    chunk_overlap: int | None = Form(default=None),
    chunk_mode: str | None = Form(default=None),
    tags: str = Form(default=""),
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> BatchUploadResponse:
    kb = _get_kb(db, kb_id)
    items: list[BatchUploadItem] = []
    for upload in files:
        try:
            response = await _register_upload(
                db,
                kb,
                upload,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_mode=chunk_mode,
                tags=tags,
                extra_metadata={"upload_source": "browser"},
            )
            items.append(
                BatchUploadItem(
                    filename=upload.filename or "document",
                    ok=True,
                    document=response.document,
                    job=response.job,
                )
            )
        except Exception as exc:
            items.append(
                BatchUploadItem(
                    filename=upload.filename or "document",
                    ok=False,
                    error=str(exc)[:1000],
                )
            )
    accepted = sum(1 for item in items if item.ok)
    return BatchUploadResponse(items=items, accepted=accepted, failed=len(items) - accepted)


@router.post(
    "/knowledge-bases/{kb_id}/documents/archive",
    response_model=BatchUploadResponse,
    status_code=201,
)
async def upload_document_archive(
    kb_id: str,
    file: UploadFile = File(...),
    chunk_size: int | None = Form(default=None),
    chunk_overlap: int | None = Form(default=None),
    chunk_mode: str | None = Form(default=None),
    tags: str = Form(default=""),
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> BatchUploadResponse:
    kb = _get_kb(db, kb_id)
    archive_path, _ = await _save_upload(file, settings.archive_max_bytes)
    try:
        return await _register_archive(
            db,
            kb,
            archive_path,
            source_name=Path(file.filename or "archive").name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_mode=chunk_mode,
            tags=tags,
            upload_source="archive_upload",
        )
    finally:
        archive_path.unlink(missing_ok=True)


@router.post(
    "/knowledge-bases/{kb_id}/documents/server-archive",
    response_model=BatchUploadResponse,
    status_code=201,
)
async def import_server_archive(
    kb_id: str,
    request: ServerArchiveRequest,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> BatchUploadResponse:
    kb = _get_kb(db, kb_id)
    root = Path(settings.import_root).resolve()
    candidate = Path(request.path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ValidationError("只允许访问服务器导入目录中的压缩包")
    if not candidate.is_file():
        raise NotFoundError("服务器压缩包不存在")
    return await _register_archive(
        db,
        kb,
        candidate,
        source_name=candidate.name,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        chunk_mode=request.chunk_mode,
        tags=request.tags,
        upload_source="server_archive",
    )


@router.get("/knowledge-bases/{kb_id}/documents", response_model=list[DocumentRead])
def list_documents(
    kb_id: str,
    status: str | None = Query(default=None),
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[DocumentRead]:
    _get_kb(db, kb_id)
    statement = (
        select(Document, func.count(DocumentChunk.id))
        .outerjoin(DocumentChunk, DocumentChunk.document_id == Document.id)
        .where(Document.kb_id == kb_id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
    )
    if status:
        statement = statement.where(Document.status == status)
    return [_document_read(document, count) for document, count in db.execute(statement)]


@router.get("/documents/{document_id}", response_model=DocumentDetail)
def get_document(
    document_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> DocumentDetail:
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError("文档不存在")
    count = db.scalar(
        select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == document.id)
    ) or 0
    base = _document_read(document, count)
    return DocumentDetail(**base.model_dump(), extracted_text=document.extracted_text)


@router.get("/documents/{document_id}/content")
def get_document_content(
    document_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> dict:
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError("文档不存在")
    return {
        "document_id": document.id,
        "filename": document.filename,
        "content_type": document.content_type,
        "text": document.extracted_text,
        "status": document.status,
        "metadata": document.metadata_json or {},
    }


@router.get("/documents/{document_id}/file")
def download_document_file(
    document_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError("文档不存在")
    return _stream_document(document)


@router.get("/documents/{document_id}/chunks", response_model=list[ChunkRead])
def list_chunks(
    document_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[ChunkRead]:
    if db.get(Document, document_id) is None:
        raise NotFoundError("文档不存在")
    chunks = db.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
    )
    return [ChunkRead.model_validate(chunk) for chunk in chunks]


@router.get("/documents/{document_id}/chunks/{chunk_id}", response_model=ChunkRead)
def get_document_chunk(
    document_id: str,
    chunk_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChunkRead:
    chunk = db.get(DocumentChunk, chunk_id)
    if chunk is None or chunk.document_id != document_id:
        raise NotFoundError("分块不存在")
    return ChunkRead.model_validate(chunk)


@router.get("/documents/{document_id}/tables", response_model=list[DocumentTableRead])
def list_document_tables(
    document_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[DocumentTableRead]:
    if db.get(Document, document_id) is None:
        raise NotFoundError("文档不存在")
    tables = db.scalars(
        select(DocumentTable)
        .where(DocumentTable.document_id == document_id)
        .order_by(DocumentTable.page_start.asc(), DocumentTable.created_at.asc())
    )
    return [DocumentTableRead.model_validate(table) for table in tables]


@router.get("/documents/{document_id}/tables/{table_id}", response_model=DocumentTableRead)
def get_document_table(
    document_id: str,
    table_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> DocumentTableRead:
    table = db.get(DocumentTable, table_id)
    if table is None or table.document_id != document_id:
        raise NotFoundError("表格不存在")
    return DocumentTableRead.model_validate(table)


@router.post("/documents/{document_id}/reindex", response_model=IngestionJobRead)
def reindex_document(
    document_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> IngestionJobRead:
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError("文档不存在")
    _ensure_kb_writable(db, document.kb_id)
    if document.status in {DocumentStatus.PENDING, DocumentStatus.PROCESSING}:
        raise ConflictError("文档正在处理中")
    document.status = DocumentStatus.PENDING
    document.error = None
    job = IngestionJob(
        id=str(uuid.uuid4()),
        document_id=document.id,
        status=JobStatus.PENDING,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    _enqueue(document, job)
    db.commit()
    db.refresh(job)
    return _job_read(job)


@router.post("/knowledge-bases/{kb_id}/documents/reparse-pdfs", response_model=ReparsePdfResponse)
def reparse_pdf_documents(
    kb_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ReparsePdfResponse:
    _get_kb(db, kb_id)
    _ensure_kb_writable(db, kb_id)
    documents = list(
        db.scalars(
            select(Document).where(
                Document.kb_id == kb_id,
                func.lower(Document.filename).like("%.pdf"),
            )
        )
    )
    job_ids: list[str] = []
    skipped = 0
    for document in documents:
        if document.status in {DocumentStatus.PENDING, DocumentStatus.PROCESSING}:
            skipped += 1
            continue
        document.status = DocumentStatus.PENDING
        document.error = None
        job = IngestionJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            status=JobStatus.PENDING,
        )
        db.add(job)
        db.flush()
        if _enqueue(document, job):
            job_ids.append(job.id)
        else:
            skipped += 1
    db.commit()
    return ReparsePdfResponse(queued=len(job_ids), skipped=skipped, job_ids=job_ids)


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Response:
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError("文档不存在")
    _ensure_kb_writable(db, document.kb_id)
    if document.status in {DocumentStatus.PENDING, DocumentStatus.PROCESSING}:
        raise ConflictError("文档正在处理中，暂时无法删除")
    vector_store.delete_document(document.kb_id, document.id, index_version="v1")
    vector_store.delete_document(document.kb_id, document.id, index_version="v2")
    bm25_store.delete_document(document.kb_id, document.id, index_version="v1")
    bm25_store.delete_document(document.kb_id, document.id, index_version="v2")
    delete_object(document.object_key)
    db.delete(document)
    db.commit()
    return Response(status_code=204)


@router.get("/ingestion-jobs/{job_id}", response_model=IngestionJobRead)
def get_ingestion_job(
    job_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> IngestionJobRead:
    job = db.get(IngestionJob, job_id)
    if job is None:
        raise NotFoundError("入库任务不存在")
    return _job_read(job)


def _stream_document(document: Document):
    response = get_object_stream(document.object_key)

    def iterator() -> Generator[bytes, None, None]:
        try:
            while data := response.read(1024 * 1024):
                yield data
        finally:
            response.close()
            response.release_conn()

    safe_name = document.filename.replace('"', "")
    headers = {
        "Content-Disposition": f'inline; filename="{safe_name}"',
        "Cache-Control": "private, max-age=60",
    }
    return StreamingResponse(
        iterator(),
        media_type=document.content_type or "application/octet-stream",
        headers=headers,
    )




