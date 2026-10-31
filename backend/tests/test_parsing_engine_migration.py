from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.models import ParsingEngineConfig
from app.db.session import Base
from app.services.parsing_engine_service import migrate_legacy_ocr_engines


def test_legacy_local_ocr_configs_migrate_idempotently(monkeypatch):
    monkeypatch.setattr(settings, "pdf_process_service_url", "http://pdf-process:8020")
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        rapid = ParsingEngineConfig(
            id="rapid-id",
            category="ocr",
            adapter="rapidocr",
            display_name="RapidOCR",
            priority=7,
            enabled=False,
            config={"language": "chi_sim+eng"},
        )
        tesseract = ParsingEngineConfig(
            id="tess-id",
            category="ocr",
            adapter="tesseract",
            display_name="Tesseract",
            priority=8,
            enabled=True,
            config={"language": "chi_sim+eng", "psm": 6},
        )
        existing_http = ParsingEngineConfig(
            id="http-id",
            category="ocr",
            adapter="http_ocr",
            display_name="Existing HTTP OCR",
            priority=9,
            enabled=True,
            config={
                "base_url": "http://pdf-" + "layout:8020",
                "engine": "tesseract",
                "transport": "multipart",
            },
        )
        db.add_all([rapid, tesseract, existing_http])
        db.commit()

        migrate_legacy_ocr_engines(db)
        migrate_legacy_ocr_engines(db)
        rows = list(
            db.scalars(select(ParsingEngineConfig).order_by(ParsingEngineConfig.priority))
        )

        assert len(rows) == 3
        assert [item.id for item in rows] == ["rapid-id", "tess-id", "http-id"]
        assert {item.adapter for item in rows} == {"http_ocr"}
        assert [item.config["engine"] for item in rows] == ["rapidocr", "tesseract", "tesseract"]
        assert all(item.config["transport"] == "multipart" for item in rows)
        assert all(item.config["base_url"] == "http://pdf-process:8020" for item in rows)
        assert rows[0].enabled is False
        assert rows[1].enabled is True
    finally:
        db.close()
        engine.dispose()