import uuid

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api import documents as documents_api
from app.db.models import Document, DocumentStatus, IngestionJob
from app.db.session import Base, SessionLocal, engine
from app.main import app


def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "test-password"},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_document_reference_serialization_and_bulk_reparse(monkeypatch):
    Base.metadata.drop_all(bind=engine)
    with TestClient(app) as client:
        headers = auth_headers(client)
        embedding = client.post(
            "/api/v1/model-profiles",
            headers=headers,
            json={
                "name": "引用测试 Embedding",
                "kind": "embedding",
                "base_url": "https://example.com/v1",
                "model_name": "embedding",
                "dimensions": 3,
            },
        )
        enrichment = client.post(
            "/api/v1/model-profiles",
            headers=headers,
            json={
                "name": "引用测试增强",
                "kind": "enrichment",
                "base_url": "https://example.com/v1",
                "model_name": "enrichment",
            },
        )
        kb = client.post(
            "/api/v1/knowledge-bases",
            headers=headers,
            json={
                "name": "引用测试知识库",
                "embedding_profile_id": embedding.json()["id"],
                "enrichment_profile_id": enrichment.json()["id"],
            },
        )

        db = SessionLocal()
        try:
            document = Document(
                id=str(uuid.uuid4()),
                kb_id=kb.json()["id"],
                filename="demo.md",
                object_key="documents/demo.md",
                content_type="text/markdown",
                size_bytes=10,
                status=DocumentStatus.READY,
                extracted_text="正文内容。",
                references_json=[
                    {
                        "label": "[1]",
                        "text": "参考条目",
                        "source_type": "bibliography",
                        "page_start": 1,
                        "page_end": 1,
                        "section_path": "参考文献",
                        "metadata": {},
                    }
                ],
                metadata_json={
                    "reference_extraction": {
                        "status": "ready",
                        "generated_by": "rules",
                        "reference_count": 1,
                    }
                },
            )
            db.add(document)
            db.add(
                Document(
                    id=str(uuid.uuid4()),
                    kb_id=kb.json()["id"],
                    filename="processing.md",
                    object_key="documents/processing.md",
                    content_type="text/markdown",
                    size_bytes=10,
                    status=DocumentStatus.PROCESSING,
                    extracted_text="",
                )
            )
            db.commit()
            document_id = document.id
        finally:
            db.close()

        monkeypatch.setattr(documents_api.celery_app, "send_task", lambda *args, **kwargs: None)
        detail = client.get(f"/api/v1/documents/{document_id}", headers=headers)
        assert detail.status_code == 200, detail.text
        payload = detail.json()
        assert payload["reference_count"] == 1
        assert payload["references"][0]["text"] == "参考条目"
        assert payload["reference_extraction"]["status"] == "ready"

        listing = client.get(
            f"/api/v1/knowledge-bases/{kb.json()['id']}/documents",
            headers=headers,
        )
        listed = next(item for item in listing.json() if item["id"] == document_id)
        assert listed["reference_count"] == 1

        content = client.get(f"/api/v1/documents/{document_id}/content", headers=headers)
        assert content.status_code == 200, content.text
        assert content.json()["text"] == "正文内容。"
        assert content.json()["references"][0]["label"] == "[1]"

        queued = client.post(
            f"/api/v1/knowledge-bases/{kb.json()['id']}/documents/reparse-references",
            headers=headers,
        )
        assert queued.status_code == 200, queued.text
        assert queued.json()["queued"] == 1
        assert len(queued.json()["job_ids"]) == 1

        db = SessionLocal()
        try:
            document = db.get(Document, document_id)
            jobs = list(db.scalars(select(IngestionJob).where(IngestionJob.document_id == document_id)))
            assert document is not None
            assert document.status == DocumentStatus.PENDING
            assert len(jobs) == 1
        finally:
            db.close()
    Base.metadata.drop_all(bind=engine)

