import logging
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.chunking.modes import split_document, validate_chunk_mode
from app.chunking.multigranularity import (
    MAX_L3_PER_L2,
    GranuleSpec,
    build_l1_specs,
    build_l3_specs,
    build_section_units,
)
from app.chunking.semantic_chunker import count_tokens
from app.core.config import settings
from app.core.errors import NotFoundError, ValidationError
from app.core.storage import download_bytes
from app.db.models import (
    Document,
    DocumentChunk,
    DocumentStatus,
    DocumentTable,
    IngestionJob,
    JobStatus,
    KnowledgeBase,
    ModelKind,
    ModelProfile,
)
from app.indexing.bm25_store import BM25Store
from app.indexing.chunk_enricher import PROMPT_VERSION, ChunkEnricher
from app.indexing.model_gateway import ModelGateway
from app.indexing.qa_generator import QA_PROMPT_VERSION, QAGenerator, QASource
from app.indexing.types import IndexedChunk, IndexedRepresentation
from app.indexing.vector_store import MilvusVectorStore
from app.parsing.base import ParsedTable
from app.parsing.metadata import build_document_metadata
from app.parsing.pdf_layout import build_layout_analyzer
from app.parsing.pdf_ocr import build_ocr_engines
from app.parsing.pdf_processor import PDFProcessingPipeline
from app.parsing.pdf_tables import build_table_manager
from app.parsing.registry import parse_document
from app.retrieval.context_window import build_context_prefix, with_context_prefix
from app.services.parsing_engine_service import engine_configs_for_run

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self):
        self.gateway = ModelGateway()
        self.enricher = ChunkEnricher(self.gateway)
        self.qa_generator = QAGenerator(self.gateway)
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
        enrichment_profile = self._enrichment_profile(db, knowledge_base)

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

            pdf_pipeline_meta: dict = {}
            if suffix == ".pdf":
                engine_configs = engine_configs_for_run(db)
                pdf_result = await PDFProcessingPipeline(
                    ocr_engines=build_ocr_engines(engine_configs.get("ocr", [])),
                    layout_analyzer=build_layout_analyzer(engine_configs.get("layout", [])),
                    table_manager=build_table_manager(engine_configs.get("table", [])),
                    enrichment_profile=enrichment_profile,
                ).process(temp_path)
                parsed = pdf_result.parsed
                pdf_pipeline_meta = pdf_result.pipeline_meta
            else:
                parsed = parse_document(temp_path)
            parsed_metadata = build_document_metadata(temp_path, parsed)
            document.metadata_json = {
                **(document.metadata_json or {}),
                **parsed_metadata,
                **({"pdf_pipeline": pdf_pipeline_meta} if pdf_pipeline_meta else {}),
            }
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
            db.execute(delete(DocumentTable).where(DocumentTable.document_id == document.id))
            db.commit()

            table_by_id = {table.id: table for table in parsed.tables}
            chunk_rows: list[DocumentChunk] = []
            tables_by_chunk: dict[str, list[ParsedTable]] = {}
            for chunk in chunks:
                metadata_json = dict(chunk.metadata)
                table_ids = [str(item) for item in metadata_json.get("table_ids") or []]
                chunk_tables = [table_by_id[item] for item in table_ids if item in table_by_id]
                chunk_rows.append(
                    DocumentChunk(
                        id=str(uuid.uuid4()),
                        document_id=document.id,
                        kb_id=document.kb_id,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        section_path=chunk.section_path,
                        token_count=chunk.token_count,
                        metadata_json=metadata_json,
                        granule_type="table" if chunk_tables else "paragraph",
                    )
                )
            db.add_all(chunk_rows)
            for table in parsed.tables:
                db.add(
                    DocumentTable(
                        id=table.id,
                        document_id=document.id,
                        kb_id=document.kb_id,
                        page_start=table.page_start,
                        page_end=table.page_end,
                        bbox=table.bbox,
                        layout_block_id=table.layout_block_id,
                        caption=table.caption,
                        cells=table.cells,
                        headers=table.headers,
                        rows=table.rows,
                        markdown=table.markdown,
                        description=table.description,
                        description_meta=table.description_meta,
                        extraction_engine=table.extraction_engine,
                        quality_score=table.quality_score,
                        quality_json=table.quality_json,
                        metadata_json=table.metadata,
                    )
                )
            db.commit()
            for row in chunk_rows:
                table_ids = [str(item) for item in (row.metadata_json or {}).get("table_ids") or []]
                if table_ids:
                    tables_by_chunk.setdefault(row.id, []).extend(
                        table_by_id[item] for item in table_ids if item in table_by_id
                    )

            enrichment_results = await self.enricher.enrich(
                enrichment_profile,
                [(row.id, row.content) for row in chunk_rows],
            )
            for row in chunk_rows:
                enrichment = enrichment_results[row.id]
                row.summary = enrichment.summary
                row.keywords = enrichment.keywords
                row.hypothetical_questions = enrichment.hypothetical_questions
                row.enrichment_status = enrichment.status
                row.enrichment_meta = dict(enrichment.meta)
                if row.id in tables_by_chunk:
                    table = tables_by_chunk[row.id][0]
                    row.summary = table.description or enrichment.summary
                    row.enrichment_meta = {
                        **dict(row.enrichment_meta or {}),
                        "table_id": table.id,
                        "table_description": table.description,
                    }
            db.commit()

            l2_rows = list(chunk_rows)
            granule_rows = await self._build_granularity_rows(
                enrichment_profile,
                document,
                l2_rows,
                start_index=len(l2_rows),
                tables_by_chunk=tables_by_chunk,
            )
            db.add_all(granule_rows)
            db.commit()
            all_rows = [*l2_rows, *granule_rows]
            self._apply_context_prefixes(document, all_rows)
            db.commit()
            qa_stats = await self._apply_qa_pairs(enrichment_profile, all_rows)
            db.commit()

            original_vectors = await self._embed_texts(
                profile, [self._contextual_text(row) for row in l2_rows]
            )
            indexed = self._indexed_chunks(document, all_rows)
            representations, representation_vectors = await self._build_representations(
                profile, document, all_rows, original_vectors
            )
            db.commit()

            active_version = self._index_version(knowledge_base)
            self._write_indexes(
                knowledge_base,
                document,
                profile,
                indexed,
                original_vectors,
                representations,
                representation_vectors,
                active_version,
            )

            document.extracted_text = parsed.full_text
            document.status = DocumentStatus.READY
            document.error = None
            job.status = JobStatus.SUCCEEDED
            job.error = None
            job.finished_at = datetime.now(UTC)
            job.result = {
                "title": parsed.title,
                "chunks": len(all_rows),
                "l1_chunks": sum(1 for row in all_rows if row.granularity_level == "L1"),
                "l2_chunks": len(l2_rows),
                "l3_chunks": sum(1 for row in all_rows if row.granularity_level == "L3"),
                "characters": parsed.total_chars,
                "format": parsed.metadata.get("format", ""),
                "chunk_mode": document.chunk_mode,
                "metadata": document.metadata_json,
                "language": document.metadata_json.get("language", ""),
                "sha256": document.metadata_json.get("sha256", ""),
                "enrichment_succeeded": sum(
                    1 for row in all_rows if row.enrichment_status == "ready"
                ),
                "enrichment_degraded": sum(
                    1 for row in all_rows if row.enrichment_status == "degraded"
                ),
                "representations": len(representations),
                "qa_pairs": qa_stats["qa_pairs"],
                "qa_by_type": qa_stats["qa_by_type"],
                "qa_degraded_chunks": qa_stats["qa_degraded_chunks"],
                "qa_prompt_version": qa_stats["qa_prompt_version"],
                "enrichment_model": (
                    enrichment_profile.model_name if enrichment_profile else "rules"
                ),
                "tables": len(parsed.tables),
                "pdf_pipeline": pdf_pipeline_meta,
            }
            db.commit()
        except Exception as exc:
            db.rollback()
            try:
                self.vector_store.delete_document(
                    document.kb_id, document.id, index_version="v1"
                )
                self.vector_store.delete_document(
                    document.kb_id, document.id, index_version="v2"
                )
                self.bm25_store.delete_document(
                    document.kb_id, document.id, index_version="v1"
                )
                self.bm25_store.delete_document(
                    document.kb_id, document.id, index_version="v2"
                )
            except Exception:
                logger.warning("failed to clean partial indexes", exc_info=True)
            db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
            db.execute(delete(DocumentTable).where(DocumentTable.document_id == document_id))
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

    async def rebuild_knowledge_base_index(self, db: Session, kb_id: str) -> dict:
        knowledge_base = db.get(KnowledgeBase, kb_id)
        if knowledge_base is None:
            raise NotFoundError("知识库不存在")
        profile = db.get(ModelProfile, knowledge_base.embedding_profile_id)
        if profile is None or profile.kind != ModelKind.EMBEDDING or not profile.is_active:
            raise ValidationError("知识库 Embedding 模型不可用")
        if not profile.dimensions:
            raise ValidationError("Embedding 模型必须配置 dimensions")
        enrichment_profile = self._enrichment_profile(db, knowledge_base)
        settings_data = dict(knowledge_base.settings or {})
        migration = {
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "processed_documents": 0,
            "total_documents": 0,
            "errors": [],
        }
        settings_data["index_migration"] = migration
        knowledge_base.settings = settings_data
        db.commit()

        documents = list(
            db.scalars(
                select(Document)
                .where(
                    Document.kb_id == kb_id,
                    Document.status == DocumentStatus.READY,
                )
                .order_by(Document.created_at.asc())
            )
        )
        migration["total_documents"] = len(documents)
        knowledge_base.settings = {**settings_data, "index_migration": migration}
        db.commit()

        for document in documents:
            try:
                rows = list(
                    db.scalars(
                        select(DocumentChunk)
                        .where(
                            DocumentChunk.document_id == document.id,
                            DocumentChunk.granularity_level == "L2",
                        )
                        .order_by(DocumentChunk.chunk_index.asc())
                    )
                )
                if not rows:
                    migration["processed_documents"] += 1
                    continue
                if any(
                    row.enrichment_status == "pending"
                    or (row.enrichment_meta or {}).get("prompt_version") != PROMPT_VERSION
                    for row in rows
                ):
                    enrichment_results = await self.enricher.enrich(
                        enrichment_profile,
                        [(row.id, row.content) for row in rows],
                    )
                    for row in rows:
                        enrichment = enrichment_results[row.id]
                        row.summary = enrichment.summary
                        row.keywords = enrichment.keywords
                        row.hypothetical_questions = enrichment.hypothetical_questions
                        row.enrichment_status = enrichment.status
                        row.enrichment_meta = dict(enrichment.meta)
                    db.commit()

                db.execute(
                    delete(DocumentChunk).where(
                        DocumentChunk.document_id == document.id,
                        DocumentChunk.granularity_level.in_(["L1", "L3"]),
                    )
                )
                db.flush()
                granule_rows = await self._build_granularity_rows(
                    enrichment_profile,
                    document,
                    rows,
                    start_index=max((row.chunk_index for row in rows), default=-1) + 1,
                )
                db.add_all(granule_rows)
                db.commit()
                all_rows = [*rows, *granule_rows]
                self._apply_context_prefixes(document, all_rows)
                db.commit()
                await self._apply_qa_pairs(enrichment_profile, all_rows)
                db.commit()

                indexed = self._indexed_chunks(document, all_rows)
                original_vectors = await self._embed_texts(
                    profile, [self._contextual_text(row) for row in rows]
                )
                representations, representation_vectors = await self._build_representations(
                    profile, document, all_rows, original_vectors
                )
                self._write_v2_indexes(
                    document, profile, indexed, representations, representation_vectors
                )
                migration["processed_documents"] += 1
                knowledge_base.settings = {**settings_data, "index_migration": migration}
                db.commit()
            except Exception as exc:
                db.rollback()
                logger.exception("knowledge base backfill failed document_id=%s", document.id)
                migration["errors"].append(
                    {"document_id": document.id, "error": str(exc)[:1000]}
                )
                migration["processed_documents"] += 1
                knowledge_base = db.get(KnowledgeBase, kb_id)
                if knowledge_base:
                    settings_data = dict(knowledge_base.settings or {})
                    knowledge_base.settings = {**settings_data, "index_migration": migration}
                    db.commit()

        knowledge_base = db.get(KnowledgeBase, kb_id)
        if knowledge_base is None:
            raise NotFoundError("知识库不存在")
        settings_data = dict(knowledge_base.settings or {})
        if migration["errors"]:
            migration["status"] = "failed"
            migration["finished_at"] = datetime.now(UTC).isoformat()
        else:
            migration["status"] = "completed"
            migration["finished_at"] = datetime.now(UTC).isoformat()
            settings_data["index_version"] = "v2"
        settings_data["index_migration"] = migration
        knowledge_base.settings = settings_data
        db.commit()
        return migration

    def pending_documents(self, db: Session) -> list[str]:
        return list(
            db.scalars(
                select(Document.id).where(Document.status == DocumentStatus.PENDING).limit(100)
            )
        )

    @staticmethod
    def _enrichment_profile(
        db: Session, knowledge_base: KnowledgeBase
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
                and profile.kind in {ModelKind.ENRICHMENT, ModelKind.LLM}
            ):
                return profile
        return None

    @staticmethod
    def _index_version(knowledge_base: KnowledgeBase) -> str:
        version = str((knowledge_base.settings or {}).get("index_version", "v1"))
        return version if version in {"v1", "v2"} else "v1"

    async def _embed_texts(self, profile: ModelProfile, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = max(1, settings.embedding_batch_size)
        for start in range(0, len(texts), batch_size):
            vectors.extend(
                await self.gateway.embed_texts(profile, texts[start : start + batch_size])
            )
        return vectors

    async def _build_granularity_rows(
        self,
        enrichment_profile: ModelProfile | None,
        document: Document,
        l2_rows: list[DocumentChunk],
        *,
        start_index: int,
        tables_by_chunk: dict[str, list[ParsedTable]] | None = None,
    ) -> list[DocumentChunk]:
        units = build_section_units(l2_rows)
        summaries = await self.enricher.summarize_sections(enrichment_profile, units)
        specs: list[GranuleSpec] = [
            *build_l1_specs(units, summaries),
            *build_l3_specs(l2_rows),
            *self._table_row_specs(l2_rows, tables_by_chunk or {}),
        ]
        rows: list[DocumentChunk] = []
        for offset, spec in enumerate(specs):
            rows.append(
                DocumentChunk(
                    id=str(uuid.uuid4()),
                    document_id=document.id,
                    kb_id=document.kb_id,
                    chunk_index=start_index + offset,
                    content=spec.content,
                    section_path=spec.section_path,
                    token_count=spec.token_count,
                    metadata_json={
                        **spec.metadata,
                        "chunk_mode": document.chunk_mode,
                    },
                    granularity_level=spec.granularity_level,
                    granule_type=spec.granule_type,
                    parent_chunk_id=spec.parent_chunk_id,
                    summary=spec.summary,
                    keywords=spec.keywords,
                    hypothetical_questions=[],
                    enrichment_status=spec.enrichment_status,
                    enrichment_meta=spec.enrichment_meta,
                )
            )
        return rows

    @staticmethod
    def _table_row_specs(
        l2_rows: list[DocumentChunk],
        tables_by_chunk: dict[str, list[ParsedTable]],
    ) -> list[GranuleSpec]:
        specs: list[GranuleSpec] = []
        for row in l2_rows:
            for table in tables_by_chunk.get(row.id, []):
                headers = (
                    table.headers[0]
                    if table.headers and isinstance(table.headers[0], list)
                    else table.headers
                )
                header_text = " | ".join(str(value).strip() for value in headers if str(value).strip())
                for row_index, values in enumerate(table.rows[:MAX_L3_PER_L2], 1):
                    row_values = values if isinstance(values, list) else [values]
                    content = f"{table.caption or '表格'} | {header_text} | " + " | ".join(
                        str(value).strip() for value in row_values
                    )
                    content = content.strip()
                    if not content:
                        continue
                    specs.append(
                        GranuleSpec(
                            content=content,
                            section_path=row.section_path,
                            token_count=count_tokens(content),
                            granularity_level="L3",
                            granule_type="table_row",
                            parent_chunk_id=row.id,
                            metadata={
                                "granularity_level": "L3",
                                "parent_chunk_id": row.id,
                                "parent_chunk_index": row.chunk_index,
                                "pages": [table.page_start, table.page_end],
                                "element_type": "table_row",
                                "selection_reason": "structured_table_row",
                                "table_ids": [table.id],
                                "table_row_index": row_index,
                            },
                            enrichment_status="ready",
                            enrichment_meta={
                                "generated_by": "layout_table",
                                "granularity_level": "L3",
                                "table_id": table.id,
                            },
                        )
                    )
        return specs

    def _indexed_chunks(
        self, document: Document, rows: list[DocumentChunk]
    ) -> list[IndexedChunk]:
        return [
            IndexedChunk(
                chunk_id=row.id,
                document_id=document.id,
                kb_id=document.kb_id,
                content=self._contextual_text(row),
                section=row.section_path,
                summary=row.summary,
                keywords=list(row.keywords or []),
                hypothetical_questions=list(row.hypothetical_questions or []),
                metadata={
                    "source": document.filename,
                    "chunk_mode": document.chunk_mode,
                    "granularity_level": getattr(row, "granularity_level", "L2"),
                    "granule_type": getattr(row, "granule_type", "paragraph"),
                    "parent_chunk_id": getattr(row, "parent_chunk_id", None),
                    "context_prefix": getattr(row, "context_prefix", ""),
                    "pages": list((row.metadata_json or {}).get("pages") or []),
                    "bboxes": list((row.metadata_json or {}).get("bboxes") or []),
                    "block_types": list((row.metadata_json or {}).get("block_types") or []),
                    "ocr_engines": list((row.metadata_json or {}).get("ocr_engines") or []),
                    "ocr_confidence": (row.metadata_json or {}).get("ocr_confidence"),
                    "table_ids": list((row.metadata_json or {}).get("table_ids") or []),
                },
            )
            for row in rows
        ]

    async def _build_representations(
        self,
        profile: ModelProfile,
        document: Document,
        rows: list[DocumentChunk],
        original_vectors: list[list[float]],
    ) -> tuple[list[IndexedRepresentation], list[list[float]]]:
        l2_rows = [
            row for row in rows if getattr(row, "granularity_level", "L2") == "L2"
        ]
        if len(l2_rows) != len(original_vectors):
            raise ValueError("L2 chunk 数量与原始向量数量不一致")
        representations = [
            IndexedRepresentation(
                chunk_id=row.id,
                document_id=document.id,
                kb_id=document.kb_id,
                section=row.section_path,
                representation_type="original",
                indexed_text=self._contextual_text(row),
            )
            for row in l2_rows
        ]
        representation_vectors = list(original_vectors)
        extra_items: list[tuple[DocumentChunk, str, int, str]] = []
        for row in rows:
            level = getattr(row, "granularity_level", "L2")
            if level == "L1":
                extra_items.append((row, "l1_summary", 0, self._contextual_text(row)))
            elif level == "L3":
                extra_items.append((row, "l3_atomic", 0, self._contextual_text(row)))
            elif row.summary:
                extra_items.append((row, "summary", 0, row.summary))
            questions = [
                str(item.get("question"))
                for item in (getattr(row, "qa_pairs", None) or [])
                if isinstance(item, dict) and item.get("question")
            ] or list(row.hypothetical_questions or [])
            for question_index, question in enumerate(questions, 1):
                if question:
                    extra_items.append((row, "question", question_index, str(question)))
        batch_size = max(1, settings.embedding_batch_size)
        for start in range(0, len(extra_items), batch_size):
            batch = extra_items[start : start + batch_size]
            try:
                extra_vectors = await self.gateway.embed_texts(
                    profile, [item[3] for item in batch]
                )
            except Exception:
                logger.warning(
                    "representation embedding failed document_id=%s batch_start=%s",
                    document.id,
                    start,
                    exc_info=True,
                )
                for row, _, _, _ in batch:
                    self._mark_degraded(row, "representation_embedding_failed")
                continue
            try:
                embedded_items = list(zip(batch, extra_vectors, strict=True))
            except ValueError:
                logger.warning(
                    "representation embedding count mismatch document_id=%s batch_start=%s",
                    document.id,
                    start,
                    exc_info=True,
                )
                for row, _, _, _ in batch:
                    self._mark_degraded(row, "representation_embedding_failed")
                continue
            for (row, representation_type, question_index, indexed_text), vector in embedded_items:
                representations.append(
                    IndexedRepresentation(
                        chunk_id=row.id,
                        document_id=document.id,
                        kb_id=document.kb_id,
                        section=row.section_path,
                        representation_type=representation_type,
                        indexed_text=indexed_text,
                        question_index=question_index,
                    )
                )
                representation_vectors.append(vector)
        for row in rows:
            meta = dict(row.enrichment_meta or {})
            meta["representation_count"] = sum(
                1 for item in representations if item.chunk_id == row.id
            )
            meta["question_count"] = len(row.hypothetical_questions or [])
            meta["qa_count"] = len(getattr(row, "qa_pairs", None) or [])
            row.enrichment_meta = meta
        return representations, representation_vectors

    @staticmethod
    def _mark_degraded(row: DocumentChunk, reason: str) -> None:
        meta = dict(row.enrichment_meta or {})
        reasons = list(meta.get("degraded_reasons") or [])
        if reason not in reasons:
            reasons.append(reason)
        meta["degraded_reasons"] = reasons
        row.enrichment_meta = meta
        row.enrichment_status = "degraded"

    @staticmethod
    def _contextual_text(row: DocumentChunk) -> str:
        return with_context_prefix(getattr(row, "context_prefix", "") or "", row.content)

    def _apply_context_prefixes(
        self, document: Document, rows: list[DocumentChunk]
    ) -> None:
        document_title = str(
            (document.metadata_json or {}).get("title") or document.filename
        )
        section_summaries = {
            row.section_path: row.summary
            for row in rows
            if getattr(row, "granularity_level", "L2") == "L1" and row.summary
        }
        rows_by_id = {row.id: row for row in rows}
        summary_chars = max(20, settings.rag_context_prefix_summary_chars)
        for row in rows:
            level = getattr(row, "granularity_level", "L2")
            summary = ""
            summary_label = "上级摘要"
            if level == "L2":
                summary = section_summaries.get(row.section_path, "")
            elif level == "L3" and row.parent_chunk_id:
                parent = rows_by_id.get(row.parent_chunk_id)
                summary = parent.summary if parent else ""
                summary_label = "父段摘要"
            row.context_prefix = build_context_prefix(
                document_title,
                row.section_path,
                summary=summary,
                summary_label=summary_label,
                summary_chars=summary_chars,
            )

    async def _apply_qa_pairs(
        self,
        profile: ModelProfile | None,
        rows: list[DocumentChunk],
    ) -> dict:
        sources = [
            QASource(
                chunk_id=row.id,
                content=row.content,
                contextual_text=self._contextual_text(row),
                section_path=row.section_path,
                keywords=list(row.keywords or []),
                summary=row.summary or "",
                granularity_level=row.granularity_level,
            )
            for row in rows
        ]
        results = await self.qa_generator.generate(profile, sources)
        qa_by_type = {question_type: 0 for question_type in ("detail", "comparison", "summary")}
        qa_pairs = 0
        degraded_chunks = 0
        for row in rows:
            result = results[row.id]
            row.qa_pairs = [
                {
                    "question": pair.question,
                    "answer": pair.answer,
                    "question_type": pair.question_type,
                }
                for pair in result.pairs
            ]
            row.hypothetical_questions = [pair.question for pair in result.pairs]
            meta = dict(row.enrichment_meta or {})
            meta.update(
                {
                    "qa_generated_by": result.meta.get("generated_by"),
                    "qa_model_profile_id": result.meta.get("model_profile_id"),
                    "qa_model_name": result.meta.get("model_name"),
                    "qa_prompt_version": result.meta.get("prompt_version"),
                    "qa_count": len(result.pairs),
                    "qa_types": list(result.meta.get("qa_types") or []),
                    "qa_degraded_reasons": list(result.meta.get("degraded_reasons") or []),
                }
            )
            row.enrichment_meta = meta
            if result.status == "degraded":
                row.enrichment_status = "degraded"
                degraded_chunks += 1
            qa_pairs += len(result.pairs)
            for pair in result.pairs:
                qa_by_type[pair.question_type] = qa_by_type.get(pair.question_type, 0) + 1
        return {
            "qa_pairs": qa_pairs,
            "qa_by_type": qa_by_type,
            "qa_degraded_chunks": degraded_chunks,
            "qa_prompt_version": QA_PROMPT_VERSION,
        }

    def _write_indexes(
        self,
        knowledge_base: KnowledgeBase,
        document: Document,
        profile: ModelProfile,
        indexed: list[IndexedChunk],
        original_vectors: list[list[float]],
        representations: list[IndexedRepresentation],
        representation_vectors: list[list[float]],
        active_version: str,
    ) -> None:
        legacy_indexed = [
            chunk for chunk in indexed if chunk.metadata.get("granularity_level") == "L2"
        ]
        if active_version == "v1":
            self.vector_store.replace_document(
                document.kb_id,
                document.id,
                document.filename,
                None,
                original_vectors,
                profile.dimensions or 0,
                index_version="v1",
                chunks=legacy_indexed,
            )
            self.bm25_store.replace_document(
                document.kb_id, document.id, legacy_indexed, index_version="v1"
            )
            try:
                self._write_v2_indexes(
                    document, profile, indexed, representations, representation_vectors
                )
            except Exception:
                logger.warning(
                    "v2 shadow index write failed for document %s", document.id, exc_info=True
                )
            return
        self._write_v2_indexes(
            document, profile, indexed, representations, representation_vectors
        )

    def _write_v2_indexes(
        self,
        document: Document,
        profile: ModelProfile,
        indexed: list[IndexedChunk],
        representations: list[IndexedRepresentation],
        representation_vectors: list[list[float]],
    ) -> None:
        self.vector_store.replace_document(
            document.kb_id,
            document.id,
            document.filename,
            representations,
            representation_vectors,
            profile.dimensions or 0,
            index_version="v2",
        )
        self.bm25_store.replace_document(
            document.kb_id,
            document.id,
            [
                chunk
                for chunk in indexed
                if chunk.metadata.get("granularity_level") != "L3"
            ],
            index_version="v2",
        )



