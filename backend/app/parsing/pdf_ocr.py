import base64
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.core.config import settings
from app.parsing.pdf_types import OCRPageResult, OCRTextBlock, normalize_bbox

_CIRCUIT_FAILURES: dict[str, int] = {}
_CIRCUIT_OPEN_UNTIL: dict[str, float] = {}


@dataclass
class OCREngineSpec:
    id: str
    adapter: str
    display_name: str
    config: dict[str, Any]


class OCREngine(Protocol):
    spec: OCREngineSpec

    def available(self) -> tuple[bool, str]: ...

    async def test(self) -> tuple[bool, str, dict[str, Any]]: ...

    async def recognize(
        self,
        image_bytes: bytes,
        *,
        page_number: int,
        width: float,
        height: float,
        language: str = "chi_sim+eng",
        layout_hint: str = "",
    ) -> OCRPageResult: ...


def _confidence(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score > 1:
        score /= 100
    return max(0.0, min(1.0, score))


def _mean_confidence(blocks: list[OCRTextBlock]) -> float | None:
    scores = [block.confidence for block in blocks if block.confidence is not None]
    return sum(scores) / len(scores) if scores else None


def _circuit_open(base_url: str) -> bool:
    open_until = _CIRCUIT_OPEN_UNTIL.get(base_url)
    if open_until is None:
        return False
    if open_until <= time.monotonic():
        _CIRCUIT_OPEN_UNTIL.pop(base_url, None)
        _CIRCUIT_FAILURES.pop(base_url, None)
        return False
    return True


def _record_connection_failure(base_url: str) -> None:
    threshold = max(1, settings.pdf_ocr_circuit_failure_threshold)
    failures = _CIRCUIT_FAILURES.get(base_url, 0) + 1
    _CIRCUIT_FAILURES[base_url] = failures
    if failures >= threshold:
        _CIRCUIT_OPEN_UNTIL[base_url] = time.monotonic() + max(
            1.0, settings.pdf_ocr_circuit_cooldown_seconds
        )


def _record_connection_success(base_url: str) -> None:
    _CIRCUIT_FAILURES.pop(base_url, None)
    _CIRCUIT_OPEN_UNTIL.pop(base_url, None)


class HTTPOCREngine:
    def __init__(
        self,
        spec: OCREngineSpec,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.spec = spec
        self._http_transport = http_transport

    def _base_url(self) -> str:
        return str(
            self.spec.config.get("base_url") or settings.resolved_pdf_process_service_url
        ).rstrip("/")

    def _headers(self) -> dict[str, str]:
        env_name = str(self.spec.config.get("api_key_env") or "")
        token = os.getenv(env_name, "") if env_name else ""
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _timeout(self) -> float:
        try:
            return float(
                self.spec.config.get("timeout_seconds") or settings.pdf_ocr_timeout_seconds
            )
        except (TypeError, ValueError):
            return settings.pdf_ocr_timeout_seconds

    def _transport(self) -> str:
        return str(self.spec.config.get("transport") or "json").lower()

    def _remote_engine(self) -> str:
        return str(self.spec.config.get("engine") or "").strip().lower()

    def available(self) -> tuple[bool, str]:
        base_url = self._base_url()
        if not base_url:
            return False, "未配置 base_url"
        if _circuit_open(base_url):
            return False, "OCR 服务连接熔断中"
        return True, base_url

    async def test(self) -> tuple[bool, str, dict[str, Any]]:
        available, detail = self.available()
        if not available:
            return False, detail, {}
        started = time.perf_counter()
        path = str(self.spec.config.get("health_path") or "/health")
        remote_engine = self._remote_engine()
        params = {"engine": remote_engine} if remote_engine else None
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout(),
                transport=self._http_transport,
            ) as client:
                response = await client.get(
                    f"{self._base_url()}{path}",
                    headers=self._headers(),
                    params=params,
                )
                response.raise_for_status()
                payload = response.json() if response.content else {}
            return True, "PDF 处理服务 OCR 可用", {
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "health": payload,
            }
        except Exception as exc:
            return False, str(exc), {
                "latency_ms": round((time.perf_counter() - started) * 1000, 2)
            }

    async def recognize(
        self,
        image_bytes: bytes,
        *,
        page_number: int,
        width: float,
        height: float,
        language: str = "chi_sim+eng",
        layout_hint: str = "",
    ) -> OCRPageResult:
        available, detail = self.available()
        if not available:
            raise RuntimeError(detail)
        max_bytes = max(1, settings.pdf_ocr_max_image_mb) * 1024 * 1024
        if not image_bytes:
            raise ValueError("OCR 页面图像为空")
        if len(image_bytes) > max_bytes:
            raise ValueError(f"OCR 页面图像不能超过 {settings.pdf_ocr_max_image_mb} MB")

        started = time.perf_counter()
        path = str(self.spec.config.get("recognize_path") or "/ocr")
        remote_engine = self._remote_engine()
        payload = {
            "page_number": page_number,
            "width": width,
            "height": height,
            "language": language,
            "layout_hint": layout_hint,
        }
        if remote_engine:
            payload["engine"] = remote_engine
        transport = self._transport()
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout(),
                transport=self._http_transport,
            ) as client:
                if transport == "multipart":
                    if not remote_engine:
                        raise ValueError("multipart OCR 配置必须指定 engine")
                    response = await client.post(
                        f"{self._base_url()}{path}",
                        headers=self._headers(),
                        data={key: str(value) for key, value in payload.items()},
                        files={"image": ("page.png", image_bytes, "image/png")},
                    )
                elif transport == "json":
                    response = await client.post(
                        f"{self._base_url()}{path}",
                        headers=self._headers(),
                        json={
                            **payload,
                            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                        },
                    )
                else:
                    raise ValueError(f"不支持的 OCR 传输方式：{transport}")
                response.raise_for_status()
                data = response.json()
            _record_connection_success(self._base_url())
        except httpx.RequestError as exc:
            _record_connection_failure(self._base_url())
            raise RuntimeError(f"OCR 服务连接失败：{exc}") from exc

        blocks = []
        for item in data.get("blocks") or []:
            blocks.append(
                OCRTextBlock(
                    text=str(item.get("text") or "").strip(),
                    bbox=normalize_bbox(item.get("bbox"), width, height),
                    confidence=_confidence(item.get("confidence")),
                    block_type=str(item.get("block_type") or "paragraph"),
                )
            )
        text = str(data.get("full_text") or "\n".join(block.text for block in blocks))
        return OCRPageResult(
            engine_id=self.spec.id,
            page_number=page_number,
            blocks=blocks,
            full_text=text,
            mean_confidence=_confidence(data.get("mean_confidence")) or _mean_confidence(blocks),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            warnings=[str(item) for item in data.get("warnings") or []],
            metadata={
                "remote_engine": str(data.get("engine_used") or remote_engine or ""),
                "transport": transport,
            },
        )


_OCR_ADAPTERS = {"http_ocr": HTTPOCREngine}


def build_ocr_engine(config: dict[str, Any]) -> OCREngine | None:
    adapter = str(config.get("adapter") or "")
    engine_class = _OCR_ADAPTERS.get(adapter)
    if engine_class is None:
        return None
    spec = OCREngineSpec(
        id=str(config.get("id") or adapter),
        adapter=adapter,
        display_name=str(config.get("display_name") or adapter),
        config=dict(config.get("config") or {}),
    )
    return engine_class(spec)


def build_ocr_engines(configs: list[dict[str, Any]]) -> list[OCREngine]:
    engines: list[OCREngine] = []
    for config in sorted(
        configs, key=lambda item: (int(item.get("priority", 100)), str(item.get("id", "")))
    ):
        engine = build_ocr_engine(config)
        if engine is not None:
            engines.append(engine)
    return engines


def ocr_adapter_capabilities() -> list[dict[str, Any]]:
    return [
        {
            "adapter": "http_ocr",
            "label": "HTTP OCR",
            "description": "调用 PDF 处理服务的 RapidOCR 或 Tesseract 接口",
        }
    ]


def sanitize_engine_config(config: dict[str, Any]) -> dict[str, Any]:
    forbidden = re.compile(r"(api[_-]?key|password|secret|token)$", re.IGNORECASE)

    def validate(value: Any) -> Any:
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                if forbidden.search(str(key)):
                    raise ValueError(f"引擎配置不能保存明文密钥字段：{key}，请使用 api_key_env")
                result[str(key)] = validate(item)
            return result
        if isinstance(value, list):
            return [validate(item) for item in value]
        return value

    return validate(config)