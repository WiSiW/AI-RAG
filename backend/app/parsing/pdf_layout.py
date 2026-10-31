import base64
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings
from app.parsing.pdf_types import LayoutRegion, PageLayout, normalize_bbox


@dataclass
class LayoutEngineSpec:
    id: str
    adapter: str
    display_name: str
    config: dict[str, Any]


class LayoutAnalyzer:
    def __init__(self, spec: LayoutEngineSpec | None = None):
        self.spec = spec

    def available(self) -> tuple[bool, str]:
        return False, "未配置版面模型"

    async def test(self) -> tuple[bool, str, dict[str, Any]]:
        available, detail = self.available()
        return available, detail, {}

    async def analyze(
        self,
        image_bytes: bytes,
        *,
        page_number: int,
        width: float,
        height: float,
        native_elements: list[dict[str, Any]],
    ) -> PageLayout:
        return geometry_layout(
            page_number=page_number,
            width=width,
            height=height,
            native_elements=native_elements,
            warning="版面模型不可用，使用几何规则",
        )


class HTTPLayoutAnalyzer(LayoutAnalyzer):
    def __init__(self, spec: LayoutEngineSpec):
        super().__init__(spec)
        self._disabled = False
    def _base_url(self) -> str:
        if self.spec is None:
            return ""
        return str(self.spec.config.get("base_url") or settings.resolved_pdf_process_service_url).rstrip("/")

    def _headers(self) -> dict[str, str]:
        env_name = str((self.spec.config if self.spec else {}).get("api_key_env") or "")
        token = os.getenv(env_name, "") if env_name else ""
        return {"Authorization": f"Bearer {token}"} if token else ""

    def available(self) -> tuple[bool, str]:
        if self._disabled:
            return False, "版面服务本任务中已降级"
        return (True, self._base_url()) if self._base_url() else (False, "未配置版面服务地址")

    async def test(self) -> tuple[bool, str, dict[str, Any]]:
        available, detail = self.available()
        if not available:
            return False, detail, {}
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=settings.pdf_layout_timeout_seconds) as client:
                response = await client.get(f"{self._base_url()}/health", headers=self._headers())
                response.raise_for_status()
            return True, "版面服务可用", {"latency_ms": round((time.perf_counter() - started) * 1000, 2)}
        except Exception as exc:
            return False, str(exc), {"latency_ms": round((time.perf_counter() - started) * 1000, 2)}

    async def analyze(
        self,
        image_bytes: bytes,
        *,
        page_number: int,
        width: float,
        height: float,
        native_elements: list[dict[str, Any]],
    ) -> PageLayout:
        available, detail = self.available()
        if not available:
            return geometry_layout(
                page_number=page_number,
                width=width,
                height=height,
                native_elements=native_elements,
                warning=detail,
            )
        payload = {
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "page_number": page_number,
            "width": width,
            "height": height,
        }
        try:
            async with httpx.AsyncClient(timeout=settings.pdf_layout_timeout_seconds) as client:
                response = await client.post(
                    f"{self._base_url()}/analyze",
                    json=payload,
                    headers=self._headers(),
                )
                response.raise_for_status()
                data = response.json()
            regions = [
                LayoutRegion(
                    id=str(item.get("id") or f"p{page_number}-r{index}"),
                    type=str(item.get("type") or "paragraph"),
                    bbox=normalize_bbox(item.get("bbox"), width, height),
                    confidence=float(item.get("confidence") or 0),
                    reading_order=int(item.get("reading_order") or index),
                    metadata=dict(item.get("metadata") or {}),
                )
                for index, item in enumerate(data.get("regions") or [])
            ]
            if regions:
                return PageLayout(
                    page_number=page_number,
                    width=width,
                    height=height,
                    regions=regions,
                    source="layout_service",
                )
        except Exception:
            self._disabled = True
        return geometry_layout(
            page_number=page_number,
            width=width,
            height=height,
            native_elements=native_elements,
            warning="版面服务调用失败，使用几何规则",
        )


def build_layout_engine(config: dict[str, Any]) -> LayoutAnalyzer:
    spec = LayoutEngineSpec(
        id=str(config.get("id") or "layout"),
        adapter=str(config.get("adapter") or "layout_http"),
        display_name=str(config.get("display_name") or "版面分析"),
        config=dict(config.get("config") or {}),
    )
    if spec.adapter == "layout_http":
        return HTTPLayoutAnalyzer(spec)
    return LayoutAnalyzer(spec)


def build_layout_analyzer(configs: list[dict[str, Any]]) -> LayoutAnalyzer:
    enabled = [item for item in configs if item.get("enabled", True)]
    if not enabled:
        return LayoutAnalyzer()
    config = sorted(enabled, key=lambda item: (int(item.get("priority", 100)), str(item.get("id", ""))))[0]
    return build_layout_engine(config)


def geometry_layout(
    *,
    page_number: int,
    width: float,
    height: float,
    native_elements: list[dict[str, Any]],
    warning: str = "",
) -> PageLayout:
    regions: list[LayoutRegion] = []
    for index, element in enumerate(native_elements):
        bbox = normalize_bbox(element.get("bbox"), width, height)
        if bbox[3] <= 0.065:
            region_type = "header"
        elif bbox[1] >= 0.935:
            region_type = "footer"
        elif element.get("block_type") == "title":
            region_type = "title"
        else:
            region_type = "paragraph"
        regions.append(
            LayoutRegion(
                id=f"p{page_number}-r{index}",
                type=region_type,
                bbox=bbox,
                confidence=float(element.get("confidence") or 0.8),
                reading_order=index,
                metadata={"page": page_number, **dict(element.get("metadata") or {})},
            )
        )
    return PageLayout(
        page_number=page_number,
        width=width,
        height=height,
        regions=regions,
        source="geometry",
        warning=warning,
    )


def layout_adapter_capabilities() -> list[dict[str, Any]]:
    return [
        {
            "adapter": "layout_http",
            "label": "HTTP 版面分析",
            "description": "调用 LayoutParser/Detectron2 独立版面服务",
        }
    ]