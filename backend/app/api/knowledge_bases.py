import uuid

from fastapi import APIRouter, Depends, Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.security import get_current_admin
from app.core.storage import delete_object
from app.db.models import (
    Conversation,
    Document,
    DocumentStatus,
    KnowledgeBase,
    ModelKind,
    ModelProfile,
    PublishedApi,
)
from app.db.session import get_db
from app.indexing.bm25_store import BM25Store
from app.indexing.vector_store import MilvusVectorStore
from app.schemas import KnowledgeBaseCreate, KnowledgeBaseRead, KnowledgeBaseUpdate

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])
vector_store = MilvusVectorStore()
bm25_store = BM25Store()

DEFAULT_KB_SETTINGS = {
    "candidate_top_k": 20,
    "final_top_k": 5,
    "rerank_threshold": 0.3,
    "chunk_mode": "structure",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "semantic_threshold": 0.72,
}


def _profile_or_error(db: Session, profile_id: str | None, kind: str) -> ModelProfile:
    if not profile_id:
        raise ValidationError(f"未配置 {kind} 模型")
    profile = db.get(ModelProfile, profile_id)
    if profile is None:
        raise NotFoundError("模型配置不存在")
    if profile.kind != kind:
        raise ValidationError(f"模型类型必须为 {kind}")
    if not profile.is_active:
        raise ValidationError("模型未启用")
    return profile


def _to_read(kb: KnowledgeBase, document_count: int, embedding_name: str | None) -> KnowledgeBaseRead:
    return KnowledgeBaseRead(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        embedding_profile_id=kb.embedding_profile_id,
        rerank_profile_id=kb.rerank_profile_id,
        default_llm_profile_id=kb.default_llm_profile_id,
        settings=kb.settings,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
        document_count=document_count,
        embedding_profile_name=embedding_name,
    )


@router.get("", response_model=list[KnowledgeBaseRead])
def list_knowledge_bases(
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[KnowledgeBaseRead]:
    rows = db.execute(
        select(KnowledgeBase, func.count(Document.id), ModelProfile.name)
        .outerjoin(Document, Document.kb_id == KnowledgeBase.id)
        .outerjoin(ModelProfile, ModelProfile.id == KnowledgeBase.embedding_profile_id)
        .group_by(KnowledgeBase.id, ModelProfile.name)
        .order_by(KnowledgeBase.created_at.desc())
    ).all()
    return [_to_read(kb, count, name) for kb, count, name in rows]


@router.post("", response_model=KnowledgeBaseRead, status_code=201)
def create_knowledge_base(
    request: KnowledgeBaseCreate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> KnowledgeBaseRead:
    embedding = _profile_or_error(db, request.embedding_profile_id, ModelKind.EMBEDDING)
    if request.rerank_profile_id:
        _profile_or_error(db, request.rerank_profile_id, ModelKind.RERANK)
    if request.default_llm_profile_id:
        _profile_or_error(db, request.default_llm_profile_id, ModelKind.LLM)
    settings_data = {**DEFAULT_KB_SETTINGS, **request.settings}
    if settings_data.get("chunk_mode") not in {
        "structure", "recursive", "fixed", "paragraph", "sentence", "semantic", "parent_child"
    }:
        raise ValidationError("不支持的分块模式")
    kb = KnowledgeBase(
        id=str(uuid.uuid4()),
        name=request.name,
        description=request.description,
        embedding_profile_id=request.embedding_profile_id,
        rerank_profile_id=request.rerank_profile_id,
        default_llm_profile_id=request.default_llm_profile_id,
        settings=settings_data,
    )
    db.add(kb)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError("知识库名称已存在") from exc
    db.refresh(kb)
    return _to_read(kb, 0, embedding.name)


@router.get("/{kb_id}", response_model=KnowledgeBaseRead)
def get_knowledge_base(
    kb_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> KnowledgeBaseRead:
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise NotFoundError("知识库不存在")
    document_count = db.scalar(
        select(func.count(Document.id)).where(Document.kb_id == kb_id)
    ) or 0
    embedding = db.get(ModelProfile, kb.embedding_profile_id)
    return _to_read(kb, document_count, embedding.name if embedding else None)


@router.patch("/{kb_id}", response_model=KnowledgeBaseRead)
def update_knowledge_base(
    kb_id: str,
    request: KnowledgeBaseUpdate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> KnowledgeBaseRead:
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise NotFoundError("知识库不存在")
    fields = request.model_fields_set
    if "name" in fields and request.name is not None:
        kb.name = request.name
    if "description" in fields and request.description is not None:
        kb.description = request.description
    if "rerank_profile_id" in fields:
        if request.rerank_profile_id:
            _profile_or_error(db, request.rerank_profile_id, ModelKind.RERANK)
        kb.rerank_profile_id = request.rerank_profile_id
    if "default_llm_profile_id" in fields:
        if request.default_llm_profile_id:
            _profile_or_error(db, request.default_llm_profile_id, ModelKind.LLM)
        kb.default_llm_profile_id = request.default_llm_profile_id
    if "settings" in fields and request.settings is not None:
        next_settings = {**kb.settings, **request.settings}
        if next_settings.get("chunk_mode") not in {
            "structure", "recursive", "fixed", "paragraph", "sentence", "semantic", "parent_child"
        }:
            raise ValidationError("不支持的分块模式")
        kb.settings = next_settings
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError("知识库名称已存在") from exc
    db.refresh(kb)
    count = db.scalar(select(func.count(Document.id)).where(Document.kb_id == kb.id)) or 0
    embedding = db.get(ModelProfile, kb.embedding_profile_id)
    return _to_read(kb, count, embedding.name if embedding else None)


@router.delete("/{kb_id}", status_code=204)
def delete_knowledge_base(
    kb_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Response:
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise NotFoundError("知识库不存在")
    active_count = db.scalar(
        select(func.count(Document.id)).where(
            Document.kb_id == kb_id,
            Document.status.in_([DocumentStatus.PENDING, DocumentStatus.PROCESSING]),
        )
    )
    if active_count:
        raise ConflictError("知识库仍有待处理文档，请稍后再试")
    for publication in db.scalars(select(PublishedApi)):
        if kb_id in (publication.knowledge_base_ids or []):
            raise ConflictError(f"知识库正被发布接口“{publication.name}”使用，无法删除")
    documents = list(db.scalars(select(Document).where(Document.kb_id == kb_id)))
    for document in documents:
        delete_object(document.object_key)
    vector_store.drop_knowledge_base(kb_id)
    bm25_store.delete_knowledge_base(kb_id)
    for conversation in db.scalars(select(Conversation).where(Conversation.knowledge_base_id == kb_id)):
        db.delete(conversation)
    for document in documents:
        db.delete(document)
    db.delete(kb)
    db.commit()
    return Response(status_code=204)




