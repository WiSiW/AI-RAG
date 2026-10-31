import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient


def _load_service_module():
    path = Path(__file__).resolve().parents[2] / "pdf_process_service" / "app.py"
    spec = importlib.util.spec_from_file_location("pdf_process_service_app", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_ocr_multipart_endpoint_and_engine_validation(monkeypatch):
    module = _load_service_module()
    monkeypatch.setattr(
        module,
        "_ocr_health",
        lambda engine: {"available": True, "detail": engine},
    )
    monkeypatch.setattr(
        module,
        "_run_rapidocr",
        lambda image_bytes, width, height: [
            module.OCRBlock(text="远程识别", bbox=[0.1, 0.2, 0.8, 0.3], confidence=0.9)
        ],
    )
    client = TestClient(module.app)

    response = client.post(
        "/ocr",
        files={"image": ("page.png", b"fake-image", "image/png")},
        data={
            "engine": "rapidocr",
            "page_number": "3",
            "width": "1000",
            "height": "1400",
            "language": "chi_sim+eng",
            "layout_hint": "paragraph,table",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["engine_used"] == "rapidocr"
    assert payload["full_text"] == "远程识别"
    assert payload["blocks"][0]["confidence"] == 0.9

    invalid = client.post(
        "/ocr",
        files={"image": ("page.png", b"fake-image", "image/png")},
        data={"engine": "unknown"},
    )
    assert invalid.status_code == 422


def test_health_is_ok_when_ocr_available_even_without_layout_model(monkeypatch):
    module = _load_service_module()
    monkeypatch.setattr(
        module,
        "_ocr_health",
        lambda engine: {"available": engine == "rapidocr", "detail": engine},
    )

    def missing_layout():
        raise RuntimeError("weights missing")

    monkeypatch.setattr(module, "_load_layout_model", missing_layout)
    response = TestClient(module.app).get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "pdf-process"
    assert payload["components"]["ocr"]["rapidocr"]["available"] is True
    assert payload["components"]["layout"]["available"] is False

def test_ocr_endpoint_rejects_empty_oversized_and_unavailable_engines(monkeypatch):
    module = _load_service_module()
    monkeypatch.setattr(module, "MAX_IMAGE_MB", 1)
    monkeypatch.setattr(
        module,
        "_ocr_health",
        lambda engine: {"available": False, "detail": "engine unavailable"},
    )
    client = TestClient(module.app)

    empty = client.post(
        "/ocr",
        files={"image": ("page.png", b"", "image/png")},
        data={"engine": "rapidocr"},
    )
    assert empty.status_code == 400

    oversized = client.post(
        "/ocr",
        files={"image": ("page.png", b"x" * (1024 * 1024 + 1), "image/png")},
        data={"engine": "rapidocr"},
    )
    assert oversized.status_code == 413

    unavailable = client.post(
        "/ocr",
        files={"image": ("page.png", b"image", "image/png")},
        data={"engine": "tesseract"},
    )
    assert unavailable.status_code == 503

def test_ocr_endpoint_enforces_optional_bearer_token(monkeypatch):
    module = _load_service_module()
    monkeypatch.setattr(module, "SERVICE_API_KEY", "process-secret")
    monkeypatch.setattr(
        module,
        "_ocr_health",
        lambda engine: {"available": True, "detail": engine},
    )
    monkeypatch.setattr(
        module,
        "_run_rapidocr",
        lambda image_bytes, width, height: [
            module.OCRBlock(text="鉴权成功", bbox=[0.1, 0.1, 0.5, 0.2], confidence=0.8)
        ],
    )
    client = TestClient(module.app)
    files = {"image": ("page.png", b"image", "image/png")}
    data = {"engine": "rapidocr"}

    assert client.post("/ocr", files=files, data=data).status_code == 401
    response = client.post(
        "/ocr",
        files=files,
        data=data,
        headers={"Authorization": "Bearer process-secret"},
    )
    assert response.status_code == 200
    assert response.json()["full_text"] == "鉴权成功"