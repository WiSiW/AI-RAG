import hashlib
import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.core.errors import ConflictError, ValidationError
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
from app.db.session import Base
from app.services import knowledge_base_transfer_service as transfer_module
from app.services.knowledge_base_transfer_service import KnowledgeBaseTransferService


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        yield Path(directory)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_package(path: Path, *, mode: str = "documents", include_table: bool = False) -> None:
    document_id = "old-doc"
    l2_id = "old-l2"
    l3_id = "old-l3"
    documents = [
        {
            "id": document_id,
            "filename": "manual.md",
            "content_type": "text/markdown",
            "size_bytes": 10,
            "tags": ["demo"],
            "metadata_json": {"title": "手册"},
            "chunk_mode": "structure",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "extracted_text": "安装前必须关闭电源。",
            "original_path": f"originals/{document_id}/manual.md",
        }
    ]
    chunks = [
        {
            "id": l2_id,
            "document_id": document_id,
            "kb_id": "old-kb",
            "chunk_index": 0,
            "content": "安装前必须关闭电源。",
            "section_path": "安装",
            "token_count": 512,
            "granularity_level": "L2",
            "granule_type": "paragraph",
            "metadata_json": {"table_ids": ["old-table"]} if include_table else {},
            "context_prefix": "[文档: 手册 / 章节: 安装]",
            "summary": "安装电源操作。",
            "keywords": ["安装", "电源"],
            "hypothetical_questions": ["安装前需要关闭什么？"],
            "qa_pairs": [
                {"question": "安装前需要关闭什么？", "answer": "必须关闭电源。", "question_type": "detail"},
                {"question": "电源与安装有什么关系？", "answer": "安装前必须关闭电源。", "question_type": "comparison"},
                {"question": "安装注意事项是什么？", "answer": "安装前必须关闭电源。", "question_type": "summary"},
            ],
            "enrichment_status": "ready",
            "enrichment_meta": {"qa_count": 3},
        },
        {
            "id": l3_id,
            "document_id": document_id,
            "kb_id": "old-kb",
            "chunk_index": 1,
            "content": "安装前必须关闭电源。",
            "section_path": "安装",
            "token_count": 10,
            "granularity_level": "L3",
            "granule_type": "sentence",
            "parent_chunk_id": l2_id,
            "metadata_json": {"parent_chunk_id": l2_id},
            "context_prefix": "[文档: 手册 / 章节: 安装]",
            "enrichment_status": "ready",
        },
    ]
    table_records = []
    if include_table:
        table_records.append(
            {
                "id": "old-table",
                "document_id": document_id,
                "kb_id": "old-kb",
                "page_start": 1,
                "page_end": 1,
                "bbox": [0.1, 0.2, 0.9, 0.8],
                "layout_block_id": "layout-table-1",
                "caption": "收入表",
                "cells": [{"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "text": "产品"}],
                "headers": ["产品", "收入"],
                "rows": [["产品A", "500"]],
                "markdown": "| 产品 | 收入 |\n| --- | --- |\n| 产品A | 500 |",
                "description": "产品A收入为500万美元。",
                "description_meta": {"generated_by": "rules"},
                "extraction_engine": "pdfplumber",
                "quality_score": 0.9,
                "quality_json": {"rows": 1},
                "metadata_json": {},
            }
        )
    payloads = {
        "documents.jsonl": "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in documents).encode(),
        "chunks.jsonl": "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in chunks).encode(),
        f"originals/{document_id}/manual.md": b"manual data",
        **(
            {
                "tables.jsonl": "".join(
                    json.dumps(item, ensure_ascii=False) + "\n" for item in table_records
                ).encode()
            }
            if include_table
            else {}
        ),
    }
    manifest = {
        "schema_version": 2 if include_table else 1,
        "package_id": str(uuid.uuid4()),
        "created_at": "2026-10-29T00:00:00+00:00",
        "app_version": "test",
        "mode": mode,
        "knowledge_base": {
            "source_id": "old-kb",
            "name": "待导入知识库",
            "description": "",
            "settings": {"chunk_mode": "structure"},
            "models": {
                "embedding": {
                    "name": "embedding",
                    "kind": "embedding",
                    "provider": "openai-compatible",
                    "model_name": "embedding",
                    "dimensions": 3,
                }
            },
        },
        "counts": {"documents": 1, "chunks": 2, **({"tables": 1} if include_table else {})},
        "index": {"version": "v2" if mode == "full" else None, "dimensions": 3},
        "file_hashes": {name: _sha(data) for name, data in payloads.items()},
    }
    if mode == "full":
        payloads["index/milvus_vectors.jsonl"] = (
            json.dumps(
                {
                    "chunk_id": l2_id,
                    "doc_id": document_id,
                    "source": "manual.md",
                    "section": "安装",
                    "representation_type": "original",
                    "question_index": 0,
                    "indexed_text": "安装前必须关闭电源。",
                    "embedding": [1.0, 0.0, 0.0],
                },
                ensure_ascii=False,
            )
            + "\n"
        ).encode()
        payloads["index/bm25_documents.jsonl"] = b"{}\n"
        manifest["file_hashes"] = {name: _sha(data) for name, data in payloads.items()}
        manifest["index"]["vector_count"] = 1
        manifest["index"]["bm25_document_count"] = 1
    manifest_bytes = json.dumps(manifest, ensure_ascii=False).encode()
    checksum_lines = [
        *(f"{_sha(data)}  {name}" for name, data in payloads.items()),
        f"{_sha(manifest_bytes)}  manifest.json",
    ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr("checksums.sha256", "\n".join(checksum_lines) + "\n")


def test_package_inspection_and_tamper_detection(temp_dir: Path):
    package = temp_dir / "package.zip"
    build_package(package)
    service = KnowledgeBaseTransferService()

    inspection = service.inspect_package(package)

    assert inspection["valid"] is True
    assert inspection["manifest"]["counts"]["chunks"] == 2

    tampered = temp_dir / "tampered.zip"
    with zipfile.ZipFile(package) as source, zipfile.ZipFile(tampered, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "documents.jsonl":
                data += b"tampered"
            target.writestr(info, data)
    with pytest.raises(ValidationError):
        service.inspect_package(tampered)


def test_package_rejects_path_traversal(temp_dir: Path):
    package = temp_dir / "evil.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("../evil.txt", "bad")
        archive.writestr("manifest.json", "{}")
        archive.writestr("checksums.sha256", "")
    with pytest.raises(ValidationError):
        KnowledgeBaseTransferService().inspect_package(package)


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "index_policy"), [("documents", "rebuild"), ("full", "strict")])
async def test_package_import_remaps_ids_and_relations(
    temp_dir: Path, monkeypatch, mode: str, index_policy: str
):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    package = temp_dir / "package.zip"
    build_package(package, mode=mode, include_table=True)
    try:
        embedding = ModelProfile(
            id="target-embedding",
            name="embedding",
            kind=ModelKind.EMBEDDING,
            provider="openai-compatible",
            base_url="https://example.com/v1",
            model_name="embedding",
            dimensions=3,
        )
        enrichment = ModelProfile(
            id="target-enrichment",
            name="enrichment",
            kind=ModelKind.ENRICHMENT,
            base_url="https://example.com/v1",
            model_name="enrichment",
        )
        db.add_all(
            [
                embedding,
                enrichment,
                KnowledgeBase(
                    id="existing-kb",
                    name="待导入知识库",
                    description="",
                    embedding_profile_id=embedding.id,
                    enrichment_profile_id=enrichment.id,
                    settings={},
                ),
            ]
        )
        db.commit()
        service = KnowledgeBaseTransferService()
        job = service.create_import_job(
            db,
            object_key="imports/test.zip",
            package_name="package.zip",
            created_by="admin",
            request_options={
                "embedding_profile_id": embedding.id,
                "enrichment_profile_id": enrichment.id,
                "index_policy": index_policy,
                "name": "待导入知识库",
                "mode": mode,
            },
        )
        monkeypatch.setattr(
            service,
            "_download_object",
            lambda object_key, target: shutil.copyfile(package, target),
        )
        monkeypatch.setattr(transfer_module, "upload_file", lambda *args, **kwargs: None)
        monkeypatch.setattr(transfer_module, "delete_object", lambda *args, **kwargs: None)

        calls: list[str] = []

        async def fake_rebuild_vectors(db_session, knowledge_base, profile):
            calls.append("rebuild")
            service._mark_documents_ready(db_session, knowledge_base.id)

        def fake_restore_vectors(db_session, knowledge_base, *args, **kwargs):
            calls.append("strict")
            service._mark_documents_ready(db_session, knowledge_base.id)

        monkeypatch.setattr(service, "_rebuild_vectors", fake_rebuild_vectors)
        monkeypatch.setattr(service, "_restore_vectors", fake_restore_vectors)
        await service.import_knowledge_base(db, job.id)
        db.refresh(job)

        assert job.status == TransferStatus.COMPLETED, job.errors
        assert calls == (["rebuild"] if index_policy == "rebuild" else ["strict"])
        imported = db.get(KnowledgeBase, job.result_knowledge_base_id)
        assert imported is not None
        assert imported.name == "待导入知识库（导入 2）"
        documents = list(db.scalars(select(Document).where(Document.kb_id == imported.id)))
        chunks = list(db.scalars(select(DocumentChunk).where(DocumentChunk.kb_id == imported.id)))
        assert len(documents) == 1
        assert documents[0].id != "old-doc"
        assert documents[0].status == DocumentStatus.READY
        assert len(chunks) == 2
        tables = list(db.scalars(select(DocumentTable).where(DocumentTable.kb_id == imported.id)))
        assert len(tables) == 1
        assert tables[0].caption == "收入表"
        assert tables[0].id != "old-table"
        l3 = next(chunk for chunk in chunks if chunk.granularity_level == "L3")
        l2 = next(chunk for chunk in chunks if chunk.granularity_level == "L2")
        assert l3.parent_chunk_id == l2.id
        assert l3.metadata_json["parent_chunk_id"] == l2.id
        assert {chunk.id for chunk in chunks}.isdisjoint({"old-l2", "old-l3"})
        assert l2.qa_pairs[0]["question"] == "安装前需要关闭什么？"
        assert l2.metadata_json["table_ids"] == [tables[0].id]
    finally:
        db.close()
        engine.dispose()


def test_active_export_locks_knowledge_base():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        embedding = ModelProfile(
            id="embedding",
            name="embedding",
            kind=ModelKind.EMBEDDING,
            base_url="https://example.com/v1",
            model_name="embedding",
            dimensions=3,
        )
        kb = KnowledgeBase(
            id="kb",
            name="kb",
            description="",
            embedding_profile_id=embedding.id,
            settings={},
        )
        db.add_all([embedding, kb])
        db.add(
            KnowledgeBaseTransferJob(
                id="job",
                kind=TransferJobKind.EXPORT,
                mode="documents",
                status=TransferStatus.RUNNING,
                source_knowledge_base_id=kb.id,
            )
        )
        db.commit()

        with pytest.raises(ConflictError):
            KnowledgeBaseTransferService.ensure_kb_writable(db, kb.id)
    finally:
        db.close()
        engine.dispose()


def test_embedding_compatibility_rejects_mismatch():
    service = KnowledgeBaseTransferService()
    manifest = {
        "knowledge_base": {
            "models": {
                "embedding": {
                    "provider": "openai-compatible",
                    "model_name": "model-a",
                    "dimensions": 1024,
                }
            }
        }
    }
    profile = ModelProfile(
        id="profile",
        name="profile",
        kind=ModelKind.EMBEDDING,
        provider="openai-compatible",
        base_url="https://example.com/v1",
        model_name="model-b",
        dimensions=768,
    )
    with pytest.raises(ValidationError):
        service._validate_embedding_compatibility(manifest, profile)
