import logging
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.chunking.modes import split_document, validate_chunk_mode
from app.core.config import settings
from app.core.errors import NotFoundError, ValidationError
from app.core.storage import download_bytes
from app.db.models import (
    Document,
    DocumentChunk,
    DocumentStatus,
    IngestionJob,
    JobStatus,
    KnowledgeBase,
    ModelKind,
    ModelProfile,
)
from app.indexing.bm25_store import BM25Store
from app.indexing.model_gateway import ModelGateway
from app.indexing.types import IndexedChunk
from app.indexing.vector_store import MilvusVectorStore
from app.parsing.metadata import build_document_metadata
from app.parsing.registry import parse_document

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self):
        self.gateway = ModelGateway()
        self.vector_store = MilvusVectorStore()
        self.bm25_store = BM25Store()

    async def ingest(self, db: Session, document_id: str, job_id: str) -> None:
        document = db.get(Document, document_id)
        job = db.get(IngestionJob, job_id)
        if document is None or job is None:
            raise NotFoundError("入库任务不存在")
        knowledge_base = db.get(KnowledgeBase, document.kb_id)
        if knowledge_base is None:
            raise NotFoundError("知识库不存在")
        profile = db.get(ModelProfile, knowledge_base.embedding_profile_id)
        if profile is None or profile.kind != ModelKind.EMBEDDING or not profile.is_active:
            raise ValidationError("知识库 Embedding 模型不可用")
        if not profile.dimensions:
            raise ValidationError("Embedding 模型必须配置 dimensions")

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        job.attempts += 1
        document.status = DocumentStatus.PROCESSING
        document.error = None
        db.commit()

        temp_path: str | None = None
        try:
            suffix = Path(document.filename).suffix.lower()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(download_bytes(document.object_key))

            parsed = parse_document(temp_path)
            parsed_metadata = build_document_metadata(temp_path, parsed)
            document.metadata_json = {**(document.metadata_json or {}), **parsed_metadata}
            try:
                chunk_mode = validate_chunk_mode(document.chunk_mode)
            except ValueError:
                chunk_mode = "structure"
                document.chunk_mode = chunk_mode
            chunks = await split_document(
                parsed,
                mode=chunk_mode,
                chunk_size=document.chunk_size,
                chunk_overlap=document.chunk_overlap,
                embed_fn=lambda texts: self.gateway.embed_texts(profile, texts),
                semantic_threshold=float(
                    (knowledge_base.settings or {}).get("semantic_threshold", 0.72)
                ),
            )

            db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
            db.commit()

            chunk_rows: list[DocumentChunk] = []
            for chunk in chunks:
                chunk_rows.append(
                    DocumentChunk(
                        id=str(uuid.uuid4()),
                        document_id=document.id,
                        kb_id=document.kb_id,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        section_path=chunk.section_path,
                        token_count=chunk.token_count,
                        metadata_json=chunk.metadata,
                    )
                )
            db.add_all(chunk_rows)
            db.commit()

            vectors: list[list[float]] = []
            batch_size = settings.embedding_batch_size
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                vectors.extend(
                    await self.gateway.embed_texts(profile, [chunk.content for chunk in batch])
                )

            indexed = [
                IndexedChunk(
                    chunk_id=row.id,
                    document_id=document.id,
                    kb_id=document.kb_id,
                    content=row.content,
                    section=row.section_path,
                    metadata={
                        "source": document.filename,
                        "chunk_mode": document.chunk_mode,
                    },
                )
                for row in chunk_rows
            ]
            self.vector_store.replace_document(
                document.kb_id,
                document.id,
                document.filename,
                indexed,
                vectors,
                profile.dimensions,
            )
            self.bm25_store.replace_document(document.kb_id, document.id, indexed)

            document.extracted_text = parsed.full_text
            document.status = DocumentStatus.READY
            document.error = None
            job.status = JobStatus.SUCCEEDED
            job.error = None
            job.finished_at = datetime.now(UTC)
            job.result = {
                "title": parsed.title,
                "chunks": len(chunks),
                "characters": parsed.total_chars,
                "format": parsed.metadata.get("format", ""),
                "chunk_mode": document.chunk_mode,
                "metadata": document.metadata_json,
                "language": document.metadata_json.get("language", ""),
                "sha256": document.metadata_json.get("sha256", ""),
            }
            db.commit()
        except Exception as exc:
            db.rollback()
            try:
                self.vector_store.delete_document(document.kb_id, document.id)
                self.bm25_store.delete_document(document.kb_id, document.id)
            except Exception:
                logger.warning("failed to clean partial indexes", exc_info=True)
            db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
            document = db.get(Document, document_id)
            job = db.get(IngestionJob, job_id)
            if document:
                document.status = DocumentStatus.FAILED
                document.error = str(exc)[:4000]
            if job:
                job.status = JobStatus.FAILED
                job.error = str(exc)[:4000]
                job.finished_at = datetime.now(UTC)
            db.commit()
            logger.exception("document ingestion failed document_id=%s", document_id)
            raise
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    logger.warning("failed to remove temporary file %s", temp_path)

    async def reindex(self, db: Session, document: Document, job: IngestionJob) -> None:
        await self.ingest(db, document.id, job.id)

    def pending_documents(self, db: Session) -> list[str]:
        return list(
            db.scalars(
                select(Document.id).where(Document.status == DocumentStatus.PENDING).limit(100)
            )
        )



