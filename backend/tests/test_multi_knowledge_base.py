import uuid

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select

from app.api.public_api import _select_kbs
from app.core.errors import AppError, ValidationError
from app.db.models import (
    ConversationKnowledgeBase,
    KnowledgeBase,
    ModelKind,
    ModelProfile,
    PublishedApi,
)
from app.db.session import Base, SessionLocal, engine
from app.indexing.types import RetrievedChunk
from app.main import app
from app.memory.session_manager import ConversationMemory
from app.retrieval.hybrid_retriever import weighted_rrf_fuse
from app.schemas import ChatRequest, PublicSearchRequest
from app.services.chat_service import ChatService


def retrieved(chunk_id: str, kb_id: str, score: float = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        kb_id=kb_id,
        content=f"content-{chunk_id}",
        section="正文",
        source=f"{chunk_id}.md",
        metadata={"score": score},
    )


def reset_database() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def create_profile(db, name: str, kind: str, dimensions: int | None = None) -> ModelProfile:
    profile = ModelProfile(
        id=str(uuid.uuid4()),
        name=name,
        kind=kind,
        base_url="https://example.com/v1",
        model_name=name,
        dimensions=dimensions,
    )
    db.add(profile)
    db.flush()
    return profile


def create_kb(db, name: str, embedding_id: str, llm_id: str | None = None) -> KnowledgeBase:
    knowledge_base = KnowledgeBase(
        id=str(uuid.uuid4()),
        name=name,
        description="",
        embedding_profile_id=embedding_id,
        default_llm_profile_id=llm_id,
        settings={
            "candidate_top_k": 20,
            "final_top_k": 5,
            "rerank_threshold": 0.3,
        },
    )
    db.add(knowledge_base)
    db.flush()
    return knowledge_base


def test_weighted_rrf_fuses_ranked_lists_from_multiple_knowledge_bases():
    first_kb = retrieved("a", "kb-1")
    shared_kb = retrieved("b", "kb-2")
    second_kb = retrieved("c", "kb-2")
    result = weighted_rrf_fuse(
        [([first_kb, shared_kb], 1.0), ([shared_kb, second_kb], 1.0)],
        top_k=3,
    )

    assert [item.chunk_id for item in result] == ["b", "a", "c"]
    assert result[0].rrf_score > result[1].rrf_score


def test_chat_scope_normalizes_legacy_and_multi_knowledge_base_fields():
    request = ChatRequest(
        knowledge_base_ids=["kb-1", "kb-1", "kb-2"],
        query="跨库问题",
    )
    assert request.knowledge_base_ids == ["kb-1", "kb-2"]
    assert request.knowledge_base_id is None

    legacy = ChatRequest(knowledge_base_id="kb-1", query="单库问题")
    assert legacy.knowledge_base_ids == ["kb-1"]
    assert legacy.knowledge_base_id == "kb-1"

    with pytest.raises(PydanticValidationError):
        ChatRequest(
            knowledge_base_id="kb-1",
            knowledge_base_ids=["kb-2"],
            query="冲突范围",
        )


def test_conversation_scope_is_persisted_and_cannot_change():
    reset_database()
    db = SessionLocal()
    try:
        embedding = create_profile(db, "embedding", ModelKind.EMBEDDING, 3)
        llm = create_profile(db, "llm", ModelKind.LLM)
        kb_one = create_kb(db, "知识库一", embedding.id, llm.id)
        kb_two = create_kb(db, "知识库二", embedding.id, llm.id)
        db.commit()

        memory = ConversationMemory()
        conversation = memory.get_or_create(
            db,
            kb_ids=[kb_one.id, kb_two.id],
            conversation_id=None,
            llm_profile_id=llm.id,
            first_query="跨库会话",
        )
        links = list(
            db.scalars(
                select(ConversationKnowledgeBase)
                .where(ConversationKnowledgeBase.conversation_id == conversation.id)
                .order_by(ConversationKnowledgeBase.position)
            )
        )
        assert [link.knowledge_base_id for link in links] == [kb_one.id, kb_two.id]

        with pytest.raises(ValidationError):
            memory.get_or_create(
                db,
                kb_ids=[kb_one.id],
                conversation_id=conversation.id,
                llm_profile_id=llm.id,
                first_query="更改范围",
            )
    finally:
        db.close()
        reset_database()


@pytest.mark.asyncio
async def test_multi_kb_retrieval_reuses_embedding_query_per_model(monkeypatch):
    reset_database()
    db = SessionLocal()
    try:
        embedding = create_profile(db, "shared-embedding", ModelKind.EMBEDDING, 3)
        kb_one = create_kb(db, "知识库一", embedding.id)
        kb_two = create_kb(db, "知识库二", embedding.id)
        db.commit()

        service = ChatService()
        embedding_calls: list[str] = []

        async def fake_embed(profile, texts):
            embedding_calls.append(profile.id)
            return [[1.0, 0.0, 0.0] for _ in texts]

        async def fake_routes(*, kb_id, **kwargs):
            if kb_id == kb_one.id:
                return {
                    "bm25": [retrieved("one-bm25", kb_id)],
                    "original": [retrieved("one-vector", kb_id)],
                    "summary": [],
                    "questions": [],
                }
            return {
                "bm25": [retrieved("two-bm25", kb_id)],
                "original": [retrieved("two-vector", kb_id)],
                "summary": [],
                "questions": [],
            }

        monkeypatch.setattr(service.gateway, "embed_texts", fake_embed)
        monkeypatch.setattr(service.retriever, "retrieve_routes", fake_routes)

        outcome = await service._retrieve(
            db,
            [kb_one, kb_two],
            "跨库查询",
            rerank=False,
            enforce_context_budget=False,
        )

        assert embedding_calls == [embedding.id]
        assert {chunk.kb_id for chunk in outcome.chunks} == {kb_one.id, kb_two.id}
        assert {chunk.metadata["kb_name"] for chunk in outcome.chunks} == {"知识库一", "知识库二"}
    finally:
        db.close()
        reset_database()


def test_public_scope_defaults_to_all_authorized_and_rejects_unbound_ids():
    reset_database()
    db = SessionLocal()
    try:
        embedding = create_profile(db, "embedding", ModelKind.EMBEDDING, 3)
        kb_one = create_kb(db, "知识库一", embedding.id)
        kb_two = create_kb(db, "知识库二", embedding.id)
        publication = PublishedApi(
            id=str(uuid.uuid4()),
            name="多库接口",
            slug="multi-kb",
            kind="search",
            knowledge_base_ids=[kb_one.id, kb_two.id],
            api_key_hash="hash",
            api_key_prefix="prefix",
        )
        db.add(publication)
        db.commit()

        assert _select_kbs(db, publication, []) == [kb_one.id, kb_two.id]
        assert _select_kbs(db, publication, [kb_one.id]) == [kb_one.id]
        with pytest.raises(AppError):
            _select_kbs(db, publication, ["not-authorized"])

        request = PublicSearchRequest(query="问题")
        assert request.knowledge_base_ids == []
    finally:
        db.close()
        reset_database()


def test_conversation_api_returns_all_knowledge_base_names():
    reset_database()
    with TestClient(app) as client:
        login = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "test-password"},
        )
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        embedding = client.post(
            "/api/v1/model-profiles",
            headers=headers,
            json={
                "name": "embedding",
                "kind": "embedding",
                "base_url": "https://example.com/v1",
                "model_name": "embedding-test",
                "dimensions": 3,
            },
        ).json()
        llm = client.post(
            "/api/v1/model-profiles",
            headers=headers,
            json={
                "name": "llm",
                "kind": "llm",
                "base_url": "https://example.com/v1",
                "model_name": "chat-test",
            },
        ).json()
        enrichment = client.post(
            "/api/v1/model-profiles",
            headers=headers,
            json={
                "name": "enrichment",
                "kind": "enrichment",
                "base_url": "https://example.com/v1",
                "model_name": "enrichment-test",
            },
        ).json()
        kb_ids = [
            client.post(
                "/api/v1/knowledge-bases",
                headers=headers,
                json={
                    "name": name,
                    "embedding_profile_id": embedding["id"],
                    "enrichment_profile_id": enrichment["id"],
                    "default_llm_profile_id": llm["id"],
                },
            ).json()["id"]
            for name in ("知识库一", "知识库二")
        ]

        db = SessionLocal()
        try:
            conversation = ConversationMemory().get_or_create(
                db,
                kb_ids=kb_ids,
                conversation_id=None,
                llm_profile_id=llm["id"],
                first_query="跨库会话",
            )
            conversation_id = conversation.id
        finally:
            db.close()

        response = client.get("/api/v1/conversations", headers=headers)
        assert response.status_code == 200
        item = next(item for item in response.json() if item["id"] == conversation_id)
        assert item["knowledge_base_ids"] == kb_ids
        assert item["knowledge_base_names"] == ["知识库一", "知识库二"]
    reset_database()
