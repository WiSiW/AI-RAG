import time
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import ParsingEngineConfig
from app.parsing.pdf_layout import build_layout_engine, layout_adapter_capabilities
from app.parsing.pdf_ocr import build_ocr_engine, ocr_adapter_capabilities
from app.parsing.pdf_tables import build_table_engine, table_adapter_capabilities

DEFAULT_ENGINES = [
    {
        "category": "ocr",
        "adapter": "http_ocr",
        "display_name": "RapidOCR（视觉服务）",
        "priority": 10,
        "enabled": True,
        "config": {
            "base_url": settings.resolved_pdf_process_service_url,
            "engine": "rapidocr",
            "transport": "multipart",
            "recognize_path": "/ocr",
            "health_path": "/health",
            "language": "chi_sim+eng",
            "api_key_env": settings.pdf_process_api_key_env,
        },
    },
    {
        "category": "ocr",
        "adapter": "http_ocr",
        "display_name": "Tesseract（视觉服务）",
        "priority": 20,
        "enabled": True,
        "config": {
            "base_url": settings.resolved_pdf_process_service_url,
            "engine": "tesseract",
            "transport": "multipart",
            "recognize_path": "/ocr",
            "health_path": "/health",
            "language": "chi_sim+eng",
            "api_key_env": settings.pdf_process_api_key_env,
        },
    },
    {
        "category": "table",
        "adapter": "pdfplumber",
        "display_name": "pdfplumber",
        "priority": 10,
        "enabled": True,
        "config": {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
        },
    },
    {
        "category": "table",
        "adapter": "camelot",
        "display_name": "Camelot",
        "priority": 20,
        "enabled": True,
        "config": {"flavor": "lattice", "fallback_flavor": "stream"},
    },
    {
        "category": "table",
        "adapter": "tabula",
        "display_name": "Tabula",
        "priority": 30,
        "enabled": True,
        "config": {"lattice": True},
    },
    {
        "category": "layout",
        "adapter": "layout_http",
        "display_name": "LayoutParser 版面服务",
        "priority": 10,
        "enabled": bool(settings.resolved_pdf_process_service_url),
        "config": {
            "base_url": settings.resolved_pdf_process_service_url,
            "api_key_env": settings.pdf_process_api_key_env,
        },
    },
]


CAPABILITIES = {
    "ocr": ocr_adapter_capabilities,
    "table": table_adapter_capabilities,
    "layout": layout_adapter_capabilities,
}


def ensure_default_engines(db: Session) -> None:
    if db.scalar(select(ParsingEngineConfig.id).limit(1)) is not None:
        return
    for item in DEFAULT_ENGINES:
        db.add(
            ParsingEngineConfig(
                id=str(uuid.uuid4()),
                category=item["category"],
                adapter=item["adapter"],
                display_name=item["display_name"],
                priority=item["priority"],
                enabled=item["enabled"],
                config=dict(item["config"]),
            )
        )
    db.commit()


def migrate_legacy_ocr_engines(db: Session) -> None:
    process_url = settings.resolved_pdf_process_service_url
    changed = False
    for item in db.scalars(
        select(ParsingEngineConfig).where(
            ParsingEngineConfig.category == "ocr",
            ParsingEngineConfig.adapter.in_(["rapidocr", "tesseract"]),
        )
    ):
        remote_engine = item.adapter
        config = dict(item.config or {})
        base_url = str(config.get("base_url") or process_url)
        if "-layout" in base_url:
            base_url = base_url.replace("-layout", "-process", 1)
        config.update(
            {
                "base_url": base_url,
                "engine": remote_engine,
                "transport": "multipart",
                "recognize_path": str(config.get("recognize_path") or "/ocr"),
                "health_path": str(config.get("health_path") or "/health"),
                "api_key_env": str(
                    config.get("api_key_env") or settings.pdf_process_api_key_env
                ),
            }
        )
        item.adapter = "http_ocr"
        item.config = config
        changed = True
    for item in db.scalars(
        select(ParsingEngineConfig).where(
            ParsingEngineConfig.category == "ocr",
            ParsingEngineConfig.adapter == "http_ocr",
        )
    ):
        config = dict(item.config or {})
        current_url = str(config.get("base_url") or "")
        if "-layout" in current_url:
            config["base_url"] = current_url.replace("-layout", "-process", 1)
            changed = True
        if not config.get("api_key_env"):
            config["api_key_env"] = settings.pdf_process_api_key_env
            changed = True
        item.config = config
    if process_url:
        for item in db.scalars(
            select(ParsingEngineConfig).where(
                ParsingEngineConfig.category == "layout",
                ParsingEngineConfig.adapter == "layout_http",
            )
        ):
            config = dict(item.config or {})
            current_url = str(config.get("base_url") or "")
            if not current_url:
                config["base_url"] = process_url
                changed = True
            elif "-layout" in current_url:
                config["base_url"] = current_url.replace("-layout", "-process", 1)
                changed = True
            if not config.get("api_key_env"):
                config["api_key_env"] = settings.pdf_process_api_key_env
                changed = True
            item.config = config
    if changed:
        db.commit()


def list_engine_configs(db: Session, category: str | None = None) -> list[ParsingEngineConfig]:
    ensure_default_engines(db)
    migrate_legacy_ocr_engines(db)
    statement = select(ParsingEngineConfig)
    if category:
        statement = statement.where(ParsingEngineConfig.category == category)
    statement = statement.order_by(
        ParsingEngineConfig.category.asc(),
        ParsingEngineConfig.priority.asc(),
        ParsingEngineConfig.created_at.asc(),
    )
    return list(db.scalars(statement))


def engine_configs_for_run(db: Session) -> dict[str, list[dict[str, Any]]]:
    grouped = {"ocr": [], "table": [], "layout": []}
    for item in list_engine_configs(db):
        if not item.enabled:
            continue
        grouped.setdefault(item.category, []).append(
            {
                "id": item.id,
                "category": item.category,
                "adapter": item.adapter,
                "display_name": item.display_name,
                "priority": item.priority,
                "enabled": item.enabled,
                "config": dict(item.config or {}),
            }
        )
    return grouped


def adapter_capabilities() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for category in ("ocr", "table", "layout"):
        for item in CAPABILITIES[category]():
            result.append({"category": category, **item, "config_schema": {}})
    return result


def validate_adapter(category: str, adapter: str) -> None:
    allowed = {item["adapter"] for item in CAPABILITIES.get(category, lambda: [])()}
    if adapter not in allowed:
        raise ValueError(f"不支持的 {category} 适配器：{adapter}")


async def test_engine(config: ParsingEngineConfig) -> tuple[bool, str, float, dict[str, Any]]:
    payload = {
        "id": config.id,
        "adapter": config.adapter,
        "display_name": config.display_name,
        "config": dict(config.config or {}),
    }
    started = time.perf_counter()
    if config.category == "ocr":
        engine = build_ocr_engine(payload)
    elif config.category == "table":
        engine = build_table_engine(payload)
    elif config.category == "layout":
        engine = build_layout_engine(payload)
    else:
        return False, "未知引擎类别", 0, {}
    if engine is None:
        return False, "适配器不存在", 0, {}
    ok, message, details = await engine.test()
    return ok, message, round((time.perf_counter() - started) * 1000, 2), details


def health_for_config(config: ParsingEngineConfig) -> dict[str, Any]:
    payload = {
        "id": config.id,
        "adapter": config.adapter,
        "display_name": config.display_name,
        "config": dict(config.config or {}),
    }
    if config.category == "ocr":
        engine = build_ocr_engine(payload)
    elif config.category == "table":
        engine = build_table_engine(payload)
    elif config.category == "layout":
        engine = build_layout_engine(payload)
    else:
        engine = None
    if engine is None:
        return {"available": False, "detail": "适配器不存在"}
    available, detail = engine.available()
    return {"available": available, "detail": detail}