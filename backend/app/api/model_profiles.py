import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.security import encrypt_secret, get_current_admin
from app.db.models import KnowledgeBase, ModelKind, ModelProfile, PublishedApi
from app.db.session import get_db
from app.indexing.model_gateway import ModelGateway
from app.schemas import (
    ModelProfileCreate,
    ModelProfileRead,
    ModelProfileUpdate,
    ModelTestResult,
)

router = APIRouter(prefix="/model-profiles", tags=["model-profiles"])
gateway = ModelGateway()


def model_to_read(profile: ModelProfile) -> ModelProfileRead:
    data = ModelProfileRead.model_validate(profile)
    return data.model_copy(update={"has_api_key": bool(profile.api_key_encrypted)})


def validate_profile_kind(profile: ModelProfile, kind: str) -> None:
    if profile.kind != kind:
        raise ValidationError(f"模型类型必须为 {kind}")
    if not profile.is_active:
        raise ValidationError("模型未启用")
    if kind == ModelKind.EMBEDDING and not profile.dimensions:
        raise ValidationError("Embedding 模型必须配置 dimensions")


@router.get("", response_model=list[ModelProfileRead])
def list_profiles(
    kind: str | None = Query(default=None),
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[ModelProfileRead]:
    statement = select(ModelProfile).order_by(ModelProfile.created_at.desc())
    if kind:
        statement = statement.where(ModelProfile.kind == kind)
    return [model_to_read(profile) for profile in db.scalars(statement)]


@router.post("", response_model=ModelProfileRead, status_code=201)
def create_profile(
    request: ModelProfileCreate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ModelProfileRead:
    if request.kind == ModelKind.EMBEDDING and not request.dimensions:
        raise ValidationError("Embedding 模型必须配置 dimensions")
    profile = ModelProfile(
        id=str(uuid.uuid4()),
        name=request.name,
        kind=request.kind,
        provider=request.provider,
        base_url=request.base_url.rstrip("/"),
        model_name=request.model_name,
        api_key_encrypted=encrypt_secret(request.api_key),
        dimensions=request.dimensions,
        config=request.config,
        is_active=request.is_active,
    )
    db.add(profile)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError("模型名称已存在") from exc
    db.refresh(profile)
    return model_to_read(profile)


@router.patch("/{profile_id}", response_model=ModelProfileRead)
def update_profile(
    profile_id: str,
    request: ModelProfileUpdate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ModelProfileRead:
    profile = db.get(ModelProfile, profile_id)
    if profile is None:
        raise NotFoundError("模型配置不存在")
    fields = request.model_fields_set
    for field in ("name", "provider", "base_url", "model_name", "dimensions", "config", "is_active"):
        if field in fields:
            value = getattr(request, field)
            if field == "base_url" and value:
                value = value.rstrip("/")
            setattr(profile, field, value)
    if "api_key" in fields and request.api_key:
        profile.api_key_encrypted = encrypt_secret(request.api_key)
    if request.clear_api_key:
        profile.api_key_encrypted = None
    if profile.kind == ModelKind.EMBEDDING and not profile.dimensions:
        raise ValidationError("Embedding 模型必须配置 dimensions")
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError("模型名称已存在") from exc
    db.refresh(profile)
    return model_to_read(profile)


@router.delete("/{profile_id}", status_code=204)
def delete_profile(
    profile_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> None:
    profile = db.get(ModelProfile, profile_id)
    if profile is None:
        raise NotFoundError("模型配置不存在")
    reference_count = db.scalar(
        select(func.count(KnowledgeBase.id)).where(
            or_(
                KnowledgeBase.embedding_profile_id == profile_id,
                KnowledgeBase.rerank_profile_id == profile_id,
                KnowledgeBase.enrichment_profile_id == profile_id,
                KnowledgeBase.default_llm_profile_id == profile_id,
            )
        )
    )
    publication_count = db.scalar(
        select(func.count(PublishedApi.id)).where(PublishedApi.llm_profile_id == profile_id)
    )
    if reference_count or publication_count:
        raise ConflictError("模型正被知识库或发布接口使用，无法删除")
    db.delete(profile)
    db.commit()


@router.post("/{profile_id}/test", response_model=ModelTestResult)
async def test_profile(
    profile_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ModelTestResult:
    profile = db.get(ModelProfile, profile_id)
    if profile is None:
        raise NotFoundError("模型配置不存在")
    if not profile.is_active:
        raise ValidationError("模型未启用")
    message, details = await gateway.test_profile(profile)
    return ModelTestResult(
        ok=True,
        latency_ms=float(details.get("latency_ms", 0)),
        message=message,
        details=details,
    )

