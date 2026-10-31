import uuid

from fastapi import APIRouter, Depends, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.security import get_current_admin
from app.db.models import ParsingEngineConfig
from app.db.session import get_db
from app.parsing.pdf_ocr import sanitize_engine_config
from app.schemas import (
    ParsingEngineAdapterRead,
    ParsingEngineCreate,
    ParsingEngineRead,
    ParsingEngineReorder,
    ParsingEngineTestResult,
    ParsingEngineUpdate,
)
from app.services.parsing_engine_service import (
    adapter_capabilities,
    health_for_config,
    list_engine_configs,
    test_engine,
    validate_adapter,
)

router = APIRouter(prefix="/parsing-engines", tags=["parsing-engines"])


def _read(config: ParsingEngineConfig) -> ParsingEngineRead:
    return ParsingEngineRead(
        id=config.id,
        category=config.category,
        adapter=config.adapter,
        display_name=config.display_name,
        priority=config.priority,
        enabled=config.enabled,
        config=dict(config.config or {}),
        health=health_for_config(config),
        created_at=config.created_at,
        updated_at=config.updated_at,
    )


@router.get("/adapters", response_model=list[ParsingEngineAdapterRead])
def list_adapters(_: str = Depends(get_current_admin)) -> list[ParsingEngineAdapterRead]:
    return [ParsingEngineAdapterRead(**item) for item in adapter_capabilities()]


@router.get("", response_model=list[ParsingEngineRead])
def list_engines(
    category: str | None = None,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[ParsingEngineRead]:
    if category and category not in {"ocr", "table", "layout"}:
        raise ValidationError("category 必须为 ocr、table 或 layout")
    return [_read(item) for item in list_engine_configs(db, category)]


@router.post("", response_model=ParsingEngineRead, status_code=201)
def create_engine(
    payload: ParsingEngineCreate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ParsingEngineRead:
    try:
        validate_adapter(payload.category, payload.adapter)
        config_data = sanitize_engine_config(payload.config)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    existing = db.scalar(
        select(ParsingEngineConfig).where(
            ParsingEngineConfig.category == payload.category,
            ParsingEngineConfig.display_name == payload.display_name,
        )
    )
    if existing is not None:
        raise ConflictError("同类别下已存在同名引擎")
    config = ParsingEngineConfig(
        id=str(uuid.uuid4()),
        category=payload.category,
        adapter=payload.adapter,
        display_name=payload.display_name,
        priority=payload.priority,
        enabled=payload.enabled,
        config=config_data,
    )
    db.add(config)
    db.commit()
    db.refresh(config)
    return _read(config)


@router.patch("/{engine_id}", response_model=ParsingEngineRead)
def update_engine(
    engine_id: str,
    payload: ParsingEngineUpdate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ParsingEngineRead:
    config = db.get(ParsingEngineConfig, engine_id)
    if config is None:
        raise NotFoundError("解析引擎不存在")
    data = payload.model_dump(exclude_unset=True)
    if "config" in data and data["config"] is not None:
        try:
            data["config"] = sanitize_engine_config(data["config"])
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
    for key, value in data.items():
        if value is not None:
            setattr(config, key, value)
    db.commit()
    db.refresh(config)
    return _read(config)


@router.delete("/{engine_id}", status_code=204)
def delete_engine(
    engine_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Response:
    config = db.get(ParsingEngineConfig, engine_id)
    if config is None:
        raise NotFoundError("解析引擎不存在")
    total = db.scalar(select(ParsingEngineConfig.id).limit(2))
    if total is not None and len(list(db.scalars(select(ParsingEngineConfig.id)))) <= 1:
        raise ConflictError("至少保留一个解析引擎配置；可以停用而不删除")
    db.delete(config)
    db.commit()
    return Response(status_code=204)


@router.post("/reorder", response_model=list[ParsingEngineRead])
def reorder_engines(
    payload: ParsingEngineReorder,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[ParsingEngineRead]:
    items = list(
        db.scalars(
            select(ParsingEngineConfig).where(ParsingEngineConfig.category == payload.category)
        )
    )
    by_id = {item.id: item for item in items}
    if any(engine_id not in by_id for engine_id in payload.engine_ids):
        raise ValidationError("排序列表包含不属于该类别的引擎")
    ordered_ids = list(dict.fromkeys(payload.engine_ids))
    ordered_ids.extend(item.id for item in items if item.id not in ordered_ids)
    for index, engine_id in enumerate(ordered_ids, 1):
        by_id[engine_id].priority = index * 10
    db.commit()
    return [_read(item) for item in list_engine_configs(db, payload.category)]


@router.post("/{engine_id}/test", response_model=ParsingEngineTestResult)
async def test_engine_endpoint(
    engine_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ParsingEngineTestResult:
    config = db.get(ParsingEngineConfig, engine_id)
    if config is None:
        raise NotFoundError("解析引擎不存在")
    ok, message, latency_ms, details = await test_engine(config)
    return ParsingEngineTestResult(ok=ok, message=message, latency_ms=latency_ms, details=details)