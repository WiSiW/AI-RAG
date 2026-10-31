import base64
import json

import httpx
import pytest

from app.core.config import settings
from app.parsing.pdf_ocr import HTTPOCREngine, OCREngineSpec


def _spec(base_url: str, **config):
    return OCREngineSpec(
        id="remote-rapid",
        adapter="http_ocr",
        display_name="Remote RapidOCR",
        config={"base_url": base_url, "engine": "rapidocr", **config},
    )


@pytest.mark.asyncio
async def test_http_ocr_multipart_request_and_response(monkeypatch):
    monkeypatch.setenv("TEST_OCR_TOKEN", "secret-token")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers.get("content-type", "")
        captured["authorization"] = request.headers.get("authorization", "")
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={
                "engine_id": "rapidocr",
                "engine_used": "rapidocr",
                "page_number": 2,
                "blocks": [
                    {"text": "识别文本", "bbox": [0.1, 0.2, 0.8, 0.3], "confidence": 0.91}
                ],
                "full_text": "识别文本",
                "mean_confidence": 0.91,
            },
        )

    engine = HTTPOCREngine(
        _spec(
            "http://ocr.test",
            transport="multipart",
            api_key_env="TEST_OCR_TOKEN",
        ),
        http_transport=httpx.MockTransport(handler),
    )
    result = await engine.recognize(
        b"fake-png",
        page_number=2,
        width=1000,
        height=1400,
        language="chi_sim+eng",
    )

    assert captured["content_type"].startswith("multipart/form-data; boundary=")
    assert captured["authorization"] == "Bearer secret-token"
    assert b'name="engine"' in captured["body"]
    assert b"rapidocr" in captured["body"]
    assert b'name="image"; filename="page.png"' in captured["body"]
    assert result.full_text == "识别文本"
    assert result.blocks[0].confidence == pytest.approx(0.91)
    assert result.metadata["remote_engine"] == "rapidocr"


@pytest.mark.asyncio
async def test_http_ocr_legacy_json_transport_still_works():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"full_text": "兼容文本", "blocks": [], "mean_confidence": 0.8},
        )

    engine = HTTPOCREngine(
        _spec("http://legacy.test"),
        http_transport=httpx.MockTransport(handler),
    )
    engine.spec.config.pop("engine")
    result = await engine.recognize(
        b"legacy-image",
        page_number=1,
        width=800,
        height=1000,
    )

    assert captured["image_base64"] == base64.b64encode(b"legacy-image").decode("ascii")
    assert "engine" not in captured
    assert result.full_text == "兼容文本"


@pytest.mark.asyncio
async def test_http_ocr_connection_failures_open_circuit(monkeypatch):
    monkeypatch.setattr(settings, "pdf_ocr_circuit_failure_threshold", 2)
    monkeypatch.setattr(settings, "pdf_ocr_circuit_cooldown_seconds", 60)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("service down", request=request)

    engine = HTTPOCREngine(
        _spec("http://circuit.test", transport="multipart"),
        http_transport=httpx.MockTransport(handler),
    )
    for _ in range(2):
        with pytest.raises(RuntimeError, match="OCR 服务连接失败"):
            await engine.recognize(b"image", page_number=1, width=100, height=100)

    available, detail = engine.available()
    assert available is False
    assert "熔断" in detail