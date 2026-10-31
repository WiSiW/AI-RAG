import hashlib
import json
import logging
import re
import shutil
import tempfile
import uuid
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import ConflictError, NotFoundError, ValidationError
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
    KnowledgeBase,
    KnowledgeBaseTransferJob,
    ModelKind,
    ModelProfile,
    TransferJobKind,
    TransferStatus,
)
from app.indexing.bm25_store import BM25Store
from app.indexing.types import IndexedChunk, IndexedRepresentation
from app.indexing.vector_store import MilvusVectorStore
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

PACKAGE_SCHEMA_VERSION = 2
EXPORT_JOB_STALE_SECONDS = 6 * 60 * 60


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _jsonl(records: list[dict]) -> bytes:
    return (
        "\n".join(json.dumps(record, ensure_ascii=False, default=_json_default) for record in records)
        + ("\n" if records else "")
    ).encode("utf-8")


def _safe_filename(name: str) -> str:
    filename = Path(name or "document").name
    filename = re.sub(r"[^0-9A-Za-z._\-\u3400-\u9fff]+", "_", filename).strip("._")
    return filename or "document"


def _package_filename(kb_name: str, mode: str) -> str:
    safe = _safe_filename(kb_name).rsplit(".", 1)[0]
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{safe}-{mode}-{timestamp}.zip"


def _model_descriptor(profile: ModelProfile | None) -> dict | None:
    if profile is None:
        return None
    return {
        "name": profile.name,
        "kind": profile.kind,
        "provider": profile.provider,
        "model_name": profile.model_name,
        "dimensions": profile.dimensions,
    }


class KnowledgeBaseTransferService:
    def __init__(self):
        self.vector_store = MilvusVectorStore()
        self.bm25_store = BM25Store()

    @staticmethod
    def ensure_kb_writable(db: Session, kb_id: str) -> None:
        job = db.scalar(
            select(KnowledgeBaseTransferJob)
            .where(
                KnowledgeBaseTransferJob.kind == TransferJobKind.EXPORT,
                KnowledgeBaseTransferJob.source_knowledge_base_id == kb_id,
                KnowledgeBaseTransferJob.status.in_(
                    [TransferStatus.QUEUED, TransferStatus.RUNNING]
                ),
            )
            .order_by(KnowledgeBaseTransferJob.created_at.desc())
            .limit(1)
        )
        if job:
            age = (datetime.now(UTC) - _aware(job.created_at)).total_seconds()
            if age < EXPORT_JOB_STALE_SECONDS:
                raise ConflictError("知识库正在导出，暂时不能修改")

    def create_export_job(
        self, db: Session, kb_id: str, mode: str, created_by: str
    ) -> KnowledgeBaseTransferJob:
        knowledge_base = db.get(KnowledgeBase, kb_id)
        if knowledge_base is None:
            raise NotFoundError("知识库不存在")
        self.ensure_kb_writable(db, kb_id)
        documents = list(
            db.scalars(select(Document).where(Document.kb_id == kb_id))
        )
        if any(
            document.status in {DocumentStatus.PENDING, DocumentStatus.PROCESSING}
            for document in documents
        ):
            raise ConflictError("知识库存在未完成入库的文档，暂时不能导出")
        if mode == "full":
            index_version = str((knowledge_base.settings or {}).get("index_version", "v1"))
            if index_version != "v2":
                raise ConflictError("完整包导出前需要先完成增强索引重建")
        job = KnowledgeBaseTransferJob(
            id=str(uuid.uuid4()),
            kind=TransferJobKind.EXPORT,
            mode=mode,
            index_policy="strict",
            status=TransferStatus.QUEUED,
            source_knowledge_base_id=kb_id,
            package_name=_package_filename(knowledge_base.name, mode),
            progress={"documents": len(documents), "chunks": 0},
            created_by=created_by,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    def export_knowledge_base(self, db: Session, job_id: str) -> None:
        job = db.get(KnowledgeBaseTransferJob, job_id)
        if job is None:
            return
        knowledge_base = db.get(KnowledgeBase, job.source_knowledge_base_id)
        if knowledge_base is None:
            self._fail_job(db, job, "知识库不存在")
            return
        job.status = TransferStatus.RUNNING
        job.started_at = datetime.now(UTC)
        db.commit()
        temp_dir = Path(tempfile.mkdtemp(prefix="zhiyuan-export-"))
        try:
            documents = list(
                db.scalars(
                    select(Document)
                    .where(Document.kb_id == knowledge_base.id)
                    .order_by(Document.created_at.asc())
                )
            )
            chunks = list(
                db.scalars(
                    select(DocumentChunk)
                    .where(DocumentChunk.kb_id == knowledge_base.id)
                    .order_by(DocumentChunk.document_id.asc(), DocumentChunk.chunk_index.asc())
                )
            )
            if any(
                document.status != DocumentStatus.READY for document in documents
            ):
                raise ConflictError("知识库存在未完成入库的文档，不能导出")
            embedding_profile = db.get(ModelProfile, knowledge_base.embedding_profile_id)
            if embedding_profile is None or not embedding_profile.dimensions:
                raise ValidationError("知识库 Embedding 模型不可用")
            tables = list(
                db.scalars(
                    select(DocumentTable)
                    .where(DocumentTable.kb_id == knowledge_base.id)
                    .order_by(DocumentTable.document_id.asc(), DocumentTable.page_start.asc())
                )
            )
            manifest = self._build_manifest(
                knowledge_base, documents, chunks, tables, embedding_profile, job.mode, db
            )
            file_hashes: dict[str, str] = {}

            documents_records = []
            for document in documents:
                documents_records.append(
                    {
                        "id": document.id,
                        "filename": document.filename,
                        "content_type": document.content_type,
                        "size_bytes": document.size_bytes,
                        "tags": list(document.tags or []),
                        "metadata_json": dict(document.metadata_json or {}),
                        "chunk_mode": document.chunk_mode,
                        "chunk_size": document.chunk_size,
                        "chunk_overlap": document.chunk_overlap,
                        "extracted_text": document.extracted_text,
                        "original_path": f"originals/{document.id}/{_safe_filename(document.filename)}",
                    }
                )
            document_bytes = _jsonl(documents_records)
            (temp_dir / "documents.jsonl").write_bytes(document_bytes)
            file_hashes["documents.jsonl"] = _sha256_bytes(document_bytes)

            chunk_records = [self._chunk_record(chunk) for chunk in chunks]
            chunk_bytes = _jsonl(chunk_records)
            (temp_dir / "chunks.jsonl").write_bytes(chunk_bytes)
            file_hashes["chunks.jsonl"] = _sha256_bytes(chunk_bytes)

            table_records = [self._table_record(table) for table in tables]
            table_bytes = _jsonl(table_records)
            (temp_dir / "tables.jsonl").write_bytes(table_bytes)
            file_hashes["tables.jsonl"] = _sha256_bytes(table_bytes)

            if job.mode == "full":
                vector_rows = self.vector_store.export_representations(
                    knowledge_base.id,
                    embedding_profile.dimensions,
                    index_version="v2",
                )
                if chunks and not vector_rows:
                    raise ValidationError("知识库 v2 向量索引为空，无法导出完整包")
                vector_bytes = _jsonl(vector_rows)
                index_dir = temp_dir / "index"
                index_dir.mkdir()
                (index_dir / "milvus_vectors.jsonl").write_bytes(vector_bytes)
                file_hashes["index/milvus_vectors.jsonl"] = _sha256_bytes(vector_bytes)
                bm25_records = self._bm25_records(documents, chunks)
                bm25_bytes = _jsonl(bm25_records)
                (index_dir / "bm25_documents.jsonl").write_bytes(bm25_bytes)
                file_hashes["index/bm25_documents.jsonl"] = _sha256_bytes(bm25_bytes)
                manifest["index"]["vector_count"] = len(vector_rows)
                manifest["index"]["bm25_document_count"] = len(bm25_records)

            manifest["file_hashes"] = file_hashes
            manifest_bytes = json.dumps(
                manifest, ensure_ascii=False, indent=2, default=_json_default
            ).encode("utf-8")
            (temp_dir / "manifest.json").write_bytes(manifest_bytes)
            file_hashes["manifest.json"] = _sha256_bytes(manifest_bytes)

            zip_path = temp_dir / job.package_name
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for relative_path in ("documents.jsonl", "chunks.jsonl"):
                    archive.write(temp_dir / relative_path, relative_path)
                if job.mode == "full":
                    archive.write(
                        temp_dir / "index" / "milvus_vectors.jsonl",
                        "index/milvus_vectors.jsonl",
                    )
                    archive.write(
                        temp_dir / "index" / "bm25_documents.jsonl",
                        "index/bm25_documents.jsonl",
                    )
                for document in documents:
                    relative_path = f"originals/{document.id}/{_safe_filename(document.filename)}"
                    digest = hashlib.sha256()
                    response = get_object_stream(document.object_key)
                    try:
                        with archive.open(relative_path, "w") as output:
                            while chunk := response.read(1024 * 1024):
                                digest.update(chunk)
                                output.write(chunk)
                    finally:
                        response.close()
                        response.release_conn()
                    file_hashes[relative_path] = digest.hexdigest()
                manifest["file_hashes"] = file_hashes
                manifest_bytes = json.dumps(
                    manifest, ensure_ascii=False, indent=2, default=_json_default
                ).encode("utf-8")
                archive.writestr("manifest.json", manifest_bytes)
                file_hashes["manifest.json"] = _sha256_bytes(manifest_bytes)
                checksum_lines = [
                    f"{digest}  {path}" for path, digest in sorted(file_hashes.items())
                ]
                archive.writestr("checksums.sha256", "\n".join(checksum_lines) + "\n")

            object_key = f"exports/{job.id}/{_safe_filename(job.package_name)}"
            upload_file(object_key, zip_path, "application/zip")
            job.object_key = object_key
            job.file_size = zip_path.stat().st_size
            job.manifest = manifest
            job.progress = {
                "documents": len(documents),
                "chunks": len(chunks),
                "vectors": int(manifest.get("index", {}).get("vector_count", 0)),
            }
            job.status = TransferStatus.COMPLETED
            job.finished_at = datetime.now(UTC)
            db.commit()
        except Exception as exc:
            db.rollback()
            job = db.get(KnowledgeBaseTransferJob, job_id)
            if job:
                self._fail_job(db, job, str(exc))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _build_manifest(
        self,
        knowledge_base: KnowledgeBase,
        documents: list[Document],
        chunks: list[DocumentChunk],
        tables: list[DocumentTable],
        embedding_profile: ModelProfile,
        mode: str,
        db: Session,
    ) -> dict:
        rerank = (
            db.get(ModelProfile, knowledge_base.rerank_profile_id)
            if knowledge_base.rerank_profile_id
            else None
        )
        enrichment = (
            db.get(ModelProfile, knowledge_base.enrichment_profile_id)
            if knowledge_base.enrichment_profile_id
            else None
        )
        llm = (
            db.get(ModelProfile, knowledge_base.default_llm_profile_id)
            if knowledge_base.default_llm_profile_id
            else None
        )
        return {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "package_id": str(uuid.uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
            "app_version": settings.app_version,
            "mode": mode,
            "knowledge_base": {
                "source_id": knowledge_base.id,
                "name": knowledge_base.name,
                "description": knowledge_base.description,
                "settings": dict(knowledge_base.settings or {}),
                "models": {
                    "embedding": _model_descriptor(embedding_profile),
                    "enrichment": _model_descriptor(enrichment),
                    "rerank": _model_descriptor(rerank),
                    "llm": _model_descriptor(llm),
                },
            },
            "counts": {
                "documents": len(documents),
                "chunks": len(chunks),
                "tables": len(tables),
            },
            "index": {
                "version": "v2" if mode == "full" else None,
                "dimensions": embedding_profile.dimensions,
                "vector_count": 0,
                "bm25_document_count": 0,
            },
            "files": {
                "documents": "documents.jsonl",
                "chunks": "chunks.jsonl",
                "tables": "tables.jsonl",
                "vectors": "index/milvus_vectors.jsonl" if mode == "full" else None,
                "bm25": "index/bm25_documents.jsonl" if mode == "full" else None,
            },
        }

    @staticmethod
    def _chunk_record(chunk: DocumentChunk) -> dict:
        return {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "kb_id": chunk.kb_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "section_path": chunk.section_path,
            "token_count": chunk.token_count,
            "metadata_json": dict(chunk.metadata_json or {}),
            "granularity_level": chunk.granularity_level,
            "granule_type": chunk.granule_type,
            "parent_chunk_id": chunk.parent_chunk_id,
            "context_prefix": chunk.context_prefix,
            "summary": chunk.summary,
            "keywords": list(chunk.keywords or []),
            "hypothetical_questions": list(chunk.hypothetical_questions or []),
            "qa_pairs": list(chunk.qa_pairs or []),
            "enrichment_status": chunk.enrichment_status,
            "enrichment_meta": dict(chunk.enrichment_meta or {}),
            "created_at": chunk.created_at,
        }

    @staticmethod
    @staticmethod
    def _table_record(table: DocumentTable) -> dict:
        return {
            "id": table.id,
            "document_id": table.document_id,
            "kb_id": table.kb_id,
            "page_start": table.page_start,
            "page_end": table.page_end,
            "bbox": list(table.bbox or []),
            "layout_block_id": table.layout_block_id,
            "caption": table.caption,
            "cells": list(table.cells or []),
            "headers": list(table.headers or []),
            "rows": list(table.rows or []),
            "markdown": table.markdown,
            "description": table.description,
            "description_meta": dict(table.description_meta or {}),
            "extraction_engine": table.extraction_engine,
            "quality_score": table.quality_score,
            "quality_json": dict(table.quality_json or {}),
            "metadata_json": dict(table.metadata_json or {}),
            "created_at": table.created_at,
        }

    @staticmethod
    def _bm25_records(
        documents: list[Document], chunks: list[DocumentChunk]
    ) -> list[dict]:
        sources = {document.id: document.filename for document in documents}
        return [
            {
                "chunk_id": chunk.id,
                "doc_id": chunk.document_id,
                "source": sources.get(chunk.document_id, ""),
                "section": chunk.section_path,
                "content": (
                    f"{chunk.context_prefix}\n{chunk.content}"
                    if chunk.context_prefix
                    else chunk.content
                ),
                "keywords": " ".join(chunk.keywords or []),
            }
            for chunk in chunks
            if chunk.granularity_level != "L3"
        ]

    def inspect_package(self, package_path: str | Path, package_name: str = "") -> dict:
        path = Path(package_path)
        if path.stat().st_size > settings.kb_package_max_bytes:
            raise ValidationError("知识库包超过大小限制")
        with zipfile.ZipFile(path) as archive:
            names = [info.filename for info in archive.infolist() if not info.is_dir()]
            if len(names) != len(set(names)):
                raise ValidationError("知识库包包含重复文件")
            if sum(info.file_size for info in archive.infolist()) > settings.kb_package_max_bytes:
                raise ValidationError("知识库包解压后超过大小限制")
            for name in names:
                if name.startswith("/") or name.startswith("\\") or ".." in Path(name).parts:
                    raise ValidationError(f"知识库包包含非法路径：{name}")
            if "manifest.json" not in names or "checksums.sha256" not in names:
                raise ValidationError("知识库包缺少 manifest 或校验文件")
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            if int(manifest.get("schema_version", 0)) not in {1, PACKAGE_SCHEMA_VERSION}:
                raise ValidationError("不支持的知识库包版本")
            if manifest.get("mode") not in {"documents", "full"}:
                raise ValidationError("知识库包模式无效")
            checksums = self._parse_checksums(archive.read("checksums.sha256").decode("utf-8"))
            unlisted = set(names) - set(checksums) - {"checksums.sha256"}
            if unlisted:
                raise ValidationError(
                    f"知识库包包含未校验文件：{', '.join(sorted(unlisted))}"
                )
            for name, expected in checksums.items():
                if name not in names:
                    raise ValidationError(f"校验清单引用不存在文件：{name}")
                if _sha256_zip_member(archive, name) != expected:
                    raise ValidationError(f"文件校验失败：{name}")
            for name, expected in dict(manifest.get("file_hashes") or {}).items():
                if name not in names or _sha256_zip_member(archive, name) != expected:
                    raise ValidationError(f"manifest 文件校验失败：{name}")
            required = ["documents.jsonl", "chunks.jsonl"]
            if manifest["mode"] == "full":
                required.extend(
                    ["index/milvus_vectors.jsonl", "index/bm25_documents.jsonl"]
                )
            for name in required:
                if name not in names:
                    raise ValidationError(f"知识库包缺少文件：{name}")
            return {
                "valid": True,
                "package_name": package_name or path.name,
                "manifest": manifest,
                "warnings": [],
            }

    @staticmethod
    def _parse_checksums(value: str) -> dict[str, str]:
        checksums: dict[str, str] = {}
        for line in value.splitlines():
            if not line.strip():
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
                raise ValidationError("checksums.sha256 格式无效")
            checksums[parts[1]] = parts[0]
        return checksums

    def create_import_job(
        self,
        db: Session,
        *,
        object_key: str,
        package_name: str,
        created_by: str,
        request_options: dict,
    ) -> KnowledgeBaseTransferJob:
        mode = str(request_options.get("mode", "documents"))
        index_policy = str(request_options.get("index_policy", "strict"))
        if mode == "documents":
            index_policy = "rebuild"
        job = KnowledgeBaseTransferJob(
            id=str(uuid.uuid4()),
            kind=TransferJobKind.IMPORT,
            mode=mode,
            index_policy=index_policy,
            status=TransferStatus.QUEUED,
            package_name=package_name,
            object_key=object_key,
            progress={"phase": "queued", "request": request_options},
            created_by=created_by,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    async def import_knowledge_base(self, db: Session, job_id: str) -> None:
        job = db.get(KnowledgeBaseTransferJob, job_id)
        if job is None or not job.object_key:
            return
        job.status = TransferStatus.RUNNING
        job.started_at = datetime.now(UTC)
        job.progress = {**dict(job.progress or {}), "phase": "downloading"}
        db.commit()
        temp_dir = Path(tempfile.mkdtemp(prefix="zhiyuan-import-"))
        imported_kb: KnowledgeBase | None = None
        uploaded_keys: list[str] = []
        try:
            package_path = temp_dir / "package.zip"
            self._download_object(job.object_key, package_path)
            inspection = self.inspect_package(package_path, job.package_name)
            manifest = inspection["manifest"]
            job.manifest = manifest
            job.mode = str(manifest["mode"])
            db.commit()
            options = dict((job.progress or {}).get("request") or {})
            if str(options.get("index_policy", "strict")) not in {"strict", "rebuild"}:
                raise ValidationError("index_policy 必须为 strict 或 rebuild")
            embedding = self._profile(db, options.get("embedding_profile_id"), ModelKind.EMBEDDING)
            if not embedding.dimensions:
                raise ValidationError("目标 Embedding 模型必须配置 dimensions")
            enrichment = self._profile(
                db, options.get("enrichment_profile_id"), ModelKind.ENRICHMENT
            )
            rerank = self._optional_profile(db, options.get("rerank_profile_id"), ModelKind.RERANK)
            llm = self._optional_profile(db, options.get("llm_profile_id"), ModelKind.LLM)
            index_policy = str(options.get("index_policy", "strict"))
            if manifest["mode"] == "documents":
                index_policy = "rebuild"
            if index_policy == "strict":
                self._validate_embedding_compatibility(manifest, embedding)

            with zipfile.ZipFile(package_path) as archive:
                document_records = _read_jsonl(archive, "documents.jsonl")
                chunk_records = _read_jsonl(archive, "chunks.jsonl")
                table_records = (
                    _read_jsonl(archive, "tables.jsonl")
                    if "tables.jsonl" in archive.namelist()
                    else []
                )
                expected_counts = dict(manifest.get("counts") or {})
                if len(document_records) != int(expected_counts.get("documents", 0)):
                    raise ValidationError("知识库包文档数量与 manifest 不一致")
                if len(chunk_records) != int(expected_counts.get("chunks", 0)):
                    raise ValidationError("知识库包 chunk 数量与 manifest 不一致")
                if "tables" in expected_counts and len(table_records) != int(expected_counts.get("tables", 0)):
                    raise ValidationError("知识库包表格数量与 manifest 不一致")
                table_map = {
                    str(record.get("id")): str(uuid.uuid4())
                    for record in table_records
                    if record.get("id")
                }
                imported_kb = self._create_knowledge_base(
                    db, manifest, options, embedding, enrichment, rerank, llm
                )
                document_map = self._create_documents(
                    db,
                    archive,
                    imported_kb,
                    document_records,
                    uploaded_keys,
                )
                chunk_map = self._create_chunks(
                    db, imported_kb, chunk_records, document_map, table_map
                )
                self._create_tables(
                    db, imported_kb, table_records, document_map, table_map
                )
                db.commit()
                job.result_knowledge_base_id = imported_kb.id
                job.progress = {
                    **dict(job.progress or {}),
                    "phase": "indexing",
                    "documents": len(document_map),
                    "chunks": len(chunk_map),
                }
                db.commit()

                if index_policy == "strict":
                    vector_records = _read_jsonl(archive, "index/milvus_vectors.jsonl")
                    self._restore_vectors(
                        db,
                        imported_kb,
                        document_map,
                        chunk_map,
                        vector_records,
                        embedding,
                    )
                else:
                    await self._rebuild_vectors(db, imported_kb, embedding)

            job.status = TransferStatus.COMPLETED
            job.finished_at = datetime.now(UTC)
            job.progress = {**dict(job.progress or {}), "phase": "completed"}
            job.errors = []
            db.commit()
        except Exception as exc:
            db.rollback()
            if imported_kb is not None:
                self._cleanup_import(db, imported_kb.id, uploaded_keys)
            job = db.get(KnowledgeBaseTransferJob, job_id)
            if job:
                self._fail_job(db, job, str(exc))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _download_object(object_key: str, target: Path) -> None:
        response = get_object_stream(object_key)
        try:
            with target.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        finally:
            response.close()
            response.release_conn()

    @staticmethod
    def _profile(db: Session, profile_id: object, kind: str) -> ModelProfile:
        profile = db.get(ModelProfile, str(profile_id)) if profile_id else None
        if profile is None or not profile.is_active or profile.kind != kind:
            raise ValidationError(f"目标 {kind} 模型不可用")
        return profile

    @staticmethod
    def _optional_profile(
        db: Session, profile_id: object, kind: str
    ) -> ModelProfile | None:
        if not profile_id:
            return None
        return KnowledgeBaseTransferService._profile(db, profile_id, kind)

    @staticmethod
    def _validate_embedding_compatibility(manifest: dict, profile: ModelProfile) -> None:
        expected = dict((manifest.get("knowledge_base", {}).get("models") or {}).get("embedding") or {})
        actual = _model_descriptor(profile) or {}
        for field in ("provider", "model_name", "dimensions"):
            if expected.get(field) != actual.get(field):
                raise ValidationError(
                    "严格索引恢复要求 Embedding 的 provider/model_name/dimensions 完全一致；"
                    "如需更换模型，请使用 index_policy=rebuild"
                )

    @staticmethod
    def _create_knowledge_base(
        db: Session,
        manifest: dict,
        options: dict,
        embedding: ModelProfile,
        enrichment: ModelProfile,
        rerank: ModelProfile | None,
        llm: ModelProfile | None,
    ) -> KnowledgeBase:
        metadata = dict(manifest.get("knowledge_base") or {})
        requested_name = str(options.get("name") or metadata.get("name") or "导入知识库").strip()
        knowledge_base = KnowledgeBase(
            id=str(uuid.uuid4()),
            name=_unique_kb_name(db, requested_name),
            description=str(metadata.get("description") or ""),
            embedding_profile_id=embedding.id,
            enrichment_profile_id=enrichment.id,
            rerank_profile_id=rerank.id if rerank else None,
            default_llm_profile_id=llm.id if llm else None,
            settings={
                **{
                    key: value
                    for key, value in dict(metadata.get("settings") or {}).items()
                    if key != "index_migration"
                },
                "index_version": "v2",
                "imported_from": {
                    "source_knowledge_base_id": metadata.get("source_id"),
                    "package_schema_version": manifest.get("schema_version"),
                },
            },
        )
        db.add(knowledge_base)
        db.flush()
        return knowledge_base

    def _create_documents(
        self,
        db: Session,
        archive: zipfile.ZipFile,
        knowledge_base: KnowledgeBase,
        records: list[dict],
        uploaded_keys: list[str],
    ) -> dict[str, str]:
        document_map: dict[str, str] = {}
        for record in records:
            old_id = str(record.get("id") or "")
            if not old_id:
                raise ValidationError("文档记录缺少 id")
            filename = _safe_filename(str(record.get("filename") or "document"))
            original_path = str(record.get("original_path") or "")
            if original_path not in archive.namelist():
                raise ValidationError(f"知识库包缺少原始文件：{original_path}")
            object_key = build_object_key(filename, prefix="documents")
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp:
                    temp_path = Path(temp.name)
                    with archive.open(original_path) as source:
                        shutil.copyfileobj(source, temp, length=1024 * 1024)
                upload_file(
                    object_key,
                    temp_path,
                    str(record.get("content_type") or "application/octet-stream"),
                )
                size_bytes = temp_path.stat().st_size
            finally:
                if temp_path:
                    temp_path.unlink(missing_ok=True)
            uploaded_keys.append(object_key)
            new_id = str(uuid.uuid4())
            document_map[old_id] = new_id
            db.add(
                Document(
                    id=new_id,
                    kb_id=knowledge_base.id,
                    filename=filename,
                    object_key=object_key,
                    content_type=str(record.get("content_type") or "application/octet-stream"),
                    size_bytes=size_bytes,
                    status=DocumentStatus.PROCESSING,
                    extracted_text=str(record.get("extracted_text") or ""),
                    tags=list(record.get("tags") or []),
                    metadata_json=dict(record.get("metadata_json") or {}),
                    chunk_mode=str(record.get("chunk_mode") or "structure"),
                    chunk_size=int(record.get("chunk_size") or 512),
                    chunk_overlap=int(record.get("chunk_overlap") or 64),
                )
            )
        return document_map

    @staticmethod
    def _create_chunks(
        db: Session,
        knowledge_base: KnowledgeBase,
        records: list[dict],
        document_map: dict[str, str],
        table_map: dict[str, str] | None = None,
    ) -> dict[str, str]:
        chunk_map = {str(record.get("id")): str(uuid.uuid4()) for record in records}
        table_map = table_map or {}
        for record in records:
            old_id = str(record.get("id"))
            old_document_id = str(record.get("document_id"))
            if old_document_id not in document_map:
                raise ValidationError("Chunk 引用了不存在的文档")
            old_parent = record.get("parent_chunk_id")
            parent_id = chunk_map.get(str(old_parent)) if old_parent else None
            metadata_json = dict(record.get("metadata_json") or {})
            if metadata_json.get("parent_chunk_ids"):
                metadata_json["parent_chunk_ids"] = [
                    chunk_map.get(str(item), str(item))
                    for item in metadata_json["parent_chunk_ids"]
                ]
            if metadata_json.get("parent_chunk_id"):
                metadata_json["parent_chunk_id"] = chunk_map.get(
                    str(metadata_json["parent_chunk_id"])
                )
            if metadata_json.get("table_ids"):
                metadata_json["table_ids"] = [
                    table_map.get(str(item), str(item))
                    for item in metadata_json["table_ids"]
                ]
            enrichment_meta = dict(record.get("enrichment_meta") or {})
            db.add(
                DocumentChunk(
                    id=chunk_map[old_id],
                    document_id=document_map[old_document_id],
                    kb_id=knowledge_base.id,
                    chunk_index=int(record.get("chunk_index") or 0),
                    content=str(record.get("content") or ""),
                    section_path=str(record.get("section_path") or "正文"),
                    token_count=int(record.get("token_count") or 0),
                    metadata_json=metadata_json,
                    granularity_level=str(record.get("granularity_level") or "L2"),
                    granule_type=str(record.get("granule_type") or "paragraph"),
                    parent_chunk_id=parent_id,
                    context_prefix=str(record.get("context_prefix") or ""),
                    summary=str(record.get("summary") or ""),
                    keywords=list(record.get("keywords") or []),
                    hypothetical_questions=list(record.get("hypothetical_questions") or []),
                    qa_pairs=list(record.get("qa_pairs") or []),
                    enrichment_status=str(record.get("enrichment_status") or "pending"),
                    enrichment_meta=enrichment_meta,
                )
            )
        db.flush()
        return chunk_map

    @staticmethod
    def _create_tables(
        db: Session,
        knowledge_base: KnowledgeBase,
        records: list[dict],
        document_map: dict[str, str],
        table_map: dict[str, str],
    ) -> None:
        for record in records:
            old_id = str(record.get("id") or "")
            old_document_id = str(record.get("document_id") or "")
            if not old_id or old_id not in table_map:
                raise ValidationError("表格记录缺少有效 id")
            if old_document_id not in document_map:
                raise ValidationError("表格引用了不存在的文档")
            metadata_json = dict(record.get("metadata_json") or {})
            db.add(
                DocumentTable(
                    id=table_map[old_id],
                    document_id=document_map[old_document_id],
                    kb_id=knowledge_base.id,
                    page_start=int(record.get("page_start") or 1),
                    page_end=int(record.get("page_end") or record.get("page_start") or 1),
                    bbox=list(record.get("bbox") or []),
                    layout_block_id=str(record.get("layout_block_id") or ""),
                    caption=str(record.get("caption") or ""),
                    cells=list(record.get("cells") or []),
                    headers=list(record.get("headers") or []),
                    rows=list(record.get("rows") or []),
                    markdown=str(record.get("markdown") or ""),
                    description=str(record.get("description") or ""),
                    description_meta=dict(record.get("description_meta") or {}),
                    extraction_engine=str(record.get("extraction_engine") or ""),
                    quality_score=float(record.get("quality_score") or 0),
                    quality_json=dict(record.get("quality_json") or {}),
                    metadata_json=metadata_json,
                )
            )
        db.flush()

    def _restore_vectors(
        self,
        db: Session,
        knowledge_base: KnowledgeBase,
        document_map: dict[str, str],
        chunk_map: dict[str, str],
        records: list[dict],
        embedding: ModelProfile,
    ) -> None:
        by_document: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            old_doc_id = str(record.get("doc_id"))
            if old_doc_id not in document_map:
                raise ValidationError("索引快照引用了不存在的文档")
            if str(record.get("chunk_id")) not in chunk_map:
                raise ValidationError("索引快照引用了不存在的 chunk")
            by_document[old_doc_id].append(record)
        if document_map and not by_document:
            raise ValidationError("完整包不包含向量快照")
        covered_chunks = {
            str(row.get("chunk_id")) for row in records if row.get("chunk_id")
        }
        missing_chunks = set(chunk_map) - covered_chunks
        if missing_chunks:
            raise ValidationError("完整包缺少部分 chunk 的向量表示")
        for old_doc_id, mapped_doc_id in document_map.items():
            document = db.get(Document, mapped_doc_id)
            rows = by_document.get(old_doc_id, [])
            if not rows or not any(row.get("representation_type") == "original" for row in rows):
                raise ValidationError("完整包缺少文档原块向量")
            if any(len(row.get("embedding") or []) != (embedding.dimensions or 0) for row in rows):
                raise ValidationError("完整包向量维度与目标 Embedding 配置不一致")
            representations = [
                IndexedRepresentation(
                    chunk_id=chunk_map[str(row["chunk_id"])],
                    document_id=mapped_doc_id,
                    kb_id=knowledge_base.id,
                    section=str(row.get("section") or "正文"),
                    representation_type=str(row.get("representation_type") or "original"),
                    indexed_text=str(row.get("indexed_text") or ""),
                    question_index=int(row.get("question_index") or 0),
                )
                for row in rows
            ]
            vectors = [list(row.get("embedding") or []) for row in rows]
            self.vector_store.replace_document(
                knowledge_base.id,
                mapped_doc_id,
                document.filename if document else "",
                representations,
                vectors,
                embedding.dimensions or 0,
                index_version="v2",
            )
        self._rebuild_bm25(db, knowledge_base)
        self._mark_documents_ready(db, knowledge_base.id)

    async def _rebuild_vectors(
        self, db: Session, knowledge_base: KnowledgeBase, embedding: ModelProfile
    ) -> None:
        ingestion = IngestionService()
        documents = list(db.scalars(select(Document).where(Document.kb_id == knowledge_base.id)))
        for document in documents:
            chunks = list(
                db.scalars(
                    select(DocumentChunk)
                    .where(DocumentChunk.document_id == document.id)
                    .order_by(DocumentChunk.chunk_index.asc())
                )
            )
            l2_chunks = [chunk for chunk in chunks if chunk.granularity_level == "L2"]
            if not chunks:
                continue
            original_vectors = await ingestion._embed_texts(
                embedding, [ingestion._contextual_text(chunk) for chunk in l2_chunks]
            )
            representations, vectors = await ingestion._build_representations(
                embedding, document, chunks, original_vectors
            )
            indexed = ingestion._indexed_chunks(document, chunks)
            ingestion._write_v2_indexes(
                document, embedding, indexed, representations, vectors
            )
        self._mark_documents_ready(db, knowledge_base.id)

    def _rebuild_bm25(self, db: Session, knowledge_base: KnowledgeBase) -> None:
        documents = {
            document.id: document
            for document in db.scalars(
                select(Document).where(Document.kb_id == knowledge_base.id)
            )
        }
        for document_id, document in documents.items():
            chunks = list(
                db.scalars(
                    select(DocumentChunk)
                    .where(DocumentChunk.document_id == document_id)
                    .order_by(DocumentChunk.chunk_index.asc())
                )
            )
            indexed = [
                IndexedChunk(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    kb_id=knowledge_base.id,
                    content=(
                        f"{chunk.context_prefix}\n{chunk.content}"
                        if chunk.context_prefix
                        else chunk.content
                    ),
                    section=chunk.section_path,
                    summary=chunk.summary,
                    keywords=list(chunk.keywords or []),
                    hypothetical_questions=list(chunk.hypothetical_questions or []),
                    metadata={"source": document.filename},
                )
                for chunk in chunks
                if chunk.granularity_level != "L3"
            ]
            self.bm25_store.replace_document(
                knowledge_base.id, document.id, indexed, index_version="v2"
            )

    @staticmethod
    def _mark_documents_ready(db: Session, kb_id: str) -> None:
        for document in db.scalars(select(Document).where(Document.kb_id == kb_id)):
            document.status = DocumentStatus.READY
            document.error = None
        db.commit()

    def _cleanup_import(
        self, db: Session, kb_id: str, uploaded_keys: list[str]
    ) -> None:
        try:
            self.vector_store.drop_knowledge_base(kb_id, index_version="v2")
            self.bm25_store.delete_knowledge_base(kb_id, index_version="v2")
        except Exception:
            logger.warning("failed to clean imported indexes", exc_info=True)
        for object_key in uploaded_keys:
            delete_object(object_key)
        knowledge_base = db.get(KnowledgeBase, kb_id)
        if knowledge_base is not None:
            db.delete(knowledge_base)
            db.commit()

    @staticmethod
    def _fail_job(db: Session, job: KnowledgeBaseTransferJob, error: str) -> None:
        job.status = TransferStatus.FAILED
        job.errors = [*list(job.errors or []), {"error": error[:2000]}]
        job.finished_at = datetime.now(UTC)
        db.commit()


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_zip_member(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(archive: zipfile.ZipFile, name: str) -> list[dict]:
    if name not in archive.namelist():
        raise ValidationError(f"知识库包缺少文件：{name}")
    records: list[dict] = []
    for line in archive.read(name).decode("utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValidationError(f"{name} 包含非法记录")
            records.append(item)
    return records


def _unique_kb_name(db: Session, requested: str) -> str:
    base = requested.strip() or "导入知识库"
    if db.scalar(select(KnowledgeBase.id).where(KnowledgeBase.name == base)) is None:
        return base
    index = 2
    while True:
        candidate = f"{base}（导入 {index}）"
        if db.scalar(select(KnowledgeBase.id).where(KnowledgeBase.name == candidate)) is None:
            return candidate
        index += 1
