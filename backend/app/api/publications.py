import uuid

from fastapi import APIRouter, Depends, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.security import generate_api_key, get_current_admin, hash_api_key
from app.db.models import KnowledgeBase, ModelKind, ModelProfile, PublishedApi
from app.db.session import get_db
from app.schemas import (
    PublishedApiCreate,
    PublishedApiCreated,
    PublishedApiRead,
    PublishedApiUpdate,
)

router = APIRouter(prefix="/publications", tags=["publications"])


def _validate_selection(
    db: Session, knowledge_base_ids: list[str], llm_profile_id: str | None
) -> None:
    unique_ids = list(dict.fromkeys(knowledge_base_ids))
    if not unique_ids:
        raise ValidationError("至少选择一个知识库")
    found = set(db.scalars(select(KnowledgeBase.id).where(KnowledgeBase.id.in_(unique_ids))))
    if found != set(unique_ids):
        raise ValidationError("包含不存在的知识库")
    if llm_profile_id:
        profile = db.get(ModelProfile, llm_profile_id)
        if profile is None or profile.kind != ModelKind.LLM or not profile.is_active:
            raise ValidationError("问答模型不可用")


def _to_read(publication: PublishedApi) -> PublishedApiRead:
    endpoint_kind = "qa" if publication.kind == "qa" else "search"
    return PublishedApiRead(
        id=publication.id,
        name=publication.name,
        slug=publication.slug,
        kind=publication.kind,
        description=publication.description,
        knowledge_base_ids=publication.knowledge_base_ids or [],
        llm_profile_id=publication.llm_profile_id,
        settings=publication.settings or {},
        api_key_prefix=publication.api_key_prefix,
        is_active=publication.is_active,
        last_used_at=publication.last_used_at,
        created_at=publication.created_at,
        updated_at=publication.updated_at,
        endpoint=f"{settings.api_prefix}/public/{endpoint_kind}/{publication.slug}",
    )


@router.get("", response_model=list[PublishedApiRead])
def list_publications(
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[PublishedApiRead]:
    items = db.scalars(select(PublishedApi).order_by(PublishedApi.created_at.desc()))
    return [_to_read(item) for item in items]


@router.post("", response_model=PublishedApiCreated, status_code=201)
def create_publication(
    request: PublishedApiCreate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> PublishedApiCreated:
    _validate_selection(db, request.knowledge_base_ids, request.llm_profile_id)
    api_key = generate_api_key()
    publication = PublishedApi(
        id=str(uuid.uuid4()),
        name=request.name,
        slug=request.slug,
        kind=request.kind,
        description=request.description,
        knowledge_base_ids=list(dict.fromkeys(request.knowledge_base_ids)),
        llm_profile_id=request.llm_profile_id,
        settings={
            "top_k": 5,
            "rerank": True,
            "stream_enabled": True,
            **request.settings,
        },
        api_key_hash=hash_api_key(api_key),
        api_key_prefix=api_key[:12],
        is_active=request.is_active,
    )
    db.add(publication)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError("接口标识已存在") from exc
    db.refresh(publication)
    return PublishedApiCreated(**_to_read(publication).model_dump(), api_key=api_key)


@router.get("/{publication_id}", response_model=PublishedApiRead)
def get_publication(
    publication_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> PublishedApiRead:
    publication = db.get(PublishedApi, publication_id)
    if publication is None:
        raise NotFoundError("发布接口不存在")
    return _to_read(publication)


@router.patch("/{publication_id}", response_model=PublishedApiRead)
def update_publication(
    publication_id: str,
    request: PublishedApiUpdate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> PublishedApiRead:
    publication = db.get(PublishedApi, publication_id)
    if publication is None:
        raise NotFoundError("发布接口不存在")
    fields = request.model_fields_set
    next_llm_id = request.llm_profile_id if "llm_profile_id" in fields else publication.llm_profile_id
    if request.knowledge_base_ids is not None:
        _validate_selection(db, request.knowledge_base_ids, next_llm_id)
        publication.knowledge_base_ids = list(dict.fromkeys(request.knowledge_base_ids))
    elif "llm_profile_id" in fields:
        _validate_selection(db, publication.knowledge_base_ids or [], request.llm_profile_id)
    for field in ("name", "slug", "kind", "description", "llm_profile_id", "is_active"):
        if field in fields:
            setattr(publication, field, getattr(request, field))
    if request.settings is not None:
        publication.settings = {**(publication.settings or {}), **request.settings}
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError("接口标识已存在") from exc
    db.refresh(publication)
    return _to_read(publication)


@router.post("/{publication_id}/rotate-key", response_model=PublishedApiCreated)
def rotate_publication_key(
    publication_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> PublishedApiCreated:
    publication = db.get(PublishedApi, publication_id)
    if publication is None:
        raise NotFoundError("发布接口不存在")
    api_key = generate_api_key()
    publication.api_key_hash = hash_api_key(api_key)
    publication.api_key_prefix = api_key[:12]
    db.commit()
    db.refresh(publication)
    return PublishedApiCreated(**_to_read(publication).model_dump(), api_key=api_key)


@router.delete("/{publication_id}", status_code=204)
def delete_publication(
    publication_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Response:
    publication = db.get(PublishedApi, publication_id)
    if publication is None:
        raise NotFoundError("发布接口不存在")
    db.delete(publication)
    db.commit()
    return Response(status_code=204)
