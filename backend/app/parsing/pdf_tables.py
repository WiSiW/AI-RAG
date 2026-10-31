import asyncio
import importlib.util
import shutil
import time
from dataclasses import dataclass
from typing import Any, Protocol

from app.parsing.pdf_types import BBox, LayoutRegion, TableCandidate, table_to_markdown


@dataclass
class TableEngineSpec:
    id: str
    adapter: str
    display_name: str
    config: dict[str, Any]


class TableEngine(Protocol):
    spec: TableEngineSpec

    def available(self) -> tuple[bool, str]: ...
    async def test(self) -> tuple[bool, str, dict[str, Any]]: ...
    async def extract(
        self,
        file_path: str,
        *,
        page_number: int,
        page_width: float,
        page_height: float,
        regions: list[LayoutRegion],
    ) -> list[TableCandidate]: ...


def _cell_grid(values: list[list[Any]]) -> tuple[list[dict[str, Any]], list[list[str]]]:
    rows = [[str(value or "").strip() for value in row] for row in values]
    width = max((len(row) for row in rows), default=0)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    cells: list[dict[str, Any]] = []
    for row_index, row in enumerate(normalized):
        for col_index, value in enumerate(row):
            cells.append(
                {
                    "row": row_index,
                    "col": col_index,
                    "rowspan": 1,
                    "colspan": 1,
                    "text": value,
                    "is_header": row_index == 0,
                }
            )
    return cells, normalized


def score_table(candidate: TableCandidate) -> TableCandidate:
    rows = [row for row in candidate.rows if isinstance(row, list)]
    row_count = len(rows)
    col_count = max((len(row) for row in rows), default=0)
    cell_count = row_count * col_count
    nonempty = sum(1 for row in rows for value in row if str(value or "").strip())
    consistent = sum(1 for row in rows if len(row) == col_count)
    completeness = nonempty / max(cell_count, 1)
    consistency = consistent / max(row_count, 1)
    has_header = bool(candidate.headers)
    structure = 1.0 if row_count >= 2 and col_count >= 2 else 0.35
    score = structure * 0.25 + completeness * 0.35 + consistency * 0.25 + (0.15 if has_header else 0)
    if candidate.extraction_engine.startswith("ocr") or candidate.metadata.get("ocr_used"):
        score *= 0.85
    candidate.quality_score = round(max(0.0, min(1.0, score)), 4)
    candidate.quality = {
        "rows": row_count,
        "columns": col_count,
        "nonempty_cells": nonempty,
        "completeness": round(completeness, 4),
        "consistency": round(consistency, 4),
        "has_header": has_header,
    }
    return candidate


def _candidate_from_rows(
    *,
    engine_id: str,
    page_number: int,
    bbox: BBox,
    rows: list[list[Any]],
    layout_block_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> TableCandidate | None:
    if not rows:
        return None
    cells, normalized = _cell_grid(rows)
    headers = normalized[0] if normalized else []
    body = normalized[1:] if len(normalized) > 1 else normalized
    candidate = TableCandidate(
        engine_id=engine_id,
        page_start=page_number,
        page_end=page_number,
        bbox=bbox,
        cells=cells,
        headers=headers,
        rows=body,
        markdown=table_to_markdown(headers, body),
        layout_block_id=layout_block_id,
        metadata=dict(metadata or {}),
    )
    return score_table(candidate)


class PDFPlumberTableEngine:
    def __init__(self, spec: TableEngineSpec):
        self.spec = spec

    def available(self) -> tuple[bool, str]:
        return (True, "pdfplumber") if importlib.util.find_spec("pdfplumber") else (False, "pdfplumber 未安装")

    async def test(self) -> tuple[bool, str, dict[str, Any]]:
        started = time.perf_counter()
        available, detail = self.available()
        return available, detail, {"latency_ms": round((time.perf_counter() - started) * 1000, 2)}

    async def extract(
        self,
        file_path: str,
        *,
        page_number: int,
        page_width: float,
        page_height: float,
        regions: list[LayoutRegion],
    ) -> list[TableCandidate]:
        available, detail = self.available()
        if not available:
            raise RuntimeError(detail)
        return await asyncio.to_thread(
            self._extract_sync, file_path, page_number, page_width, page_height, regions
        )

    def _extract_sync(
        self,
        file_path: str,
        page_number: int,
        page_width: float,
        page_height: float,
        regions: list[LayoutRegion],
    ) -> list[TableCandidate]:
        import pdfplumber

        settings_data = dict(self.spec.config)
        table_settings = {
            "vertical_strategy": str(settings_data.get("vertical_strategy") or "lines"),
            "horizontal_strategy": str(settings_data.get("horizontal_strategy") or "lines"),
            "snap_tolerance": int(settings_data.get("snap_tolerance") or 3),
            "join_tolerance": int(settings_data.get("join_tolerance") or 3),
        }
        candidates: list[TableCandidate] = []
        with pdfplumber.open(file_path) as pdf:
            page = pdf.pages[page_number - 1]
            found = page.find_tables(table_settings=table_settings)
            for index, table in enumerate(found):
                data = table.extract() or []
                bbox_pixels = list(table.bbox)
                from app.parsing.pdf_types import normalize_bbox

                bbox = normalize_bbox(bbox_pixels, page_width, page_height)
                block_id = ""
                for region in regions:
                    if region.type == "table" and _contains(region.bbox, bbox, threshold=0.2):
                        block_id = region.id
                        break
                candidate = _candidate_from_rows(
                    engine_id=self.spec.id,
                    page_number=page_number,
                    bbox=bbox,
                    rows=data,
                    layout_block_id=block_id or f"pdfplumber-p{page_number}-t{index}",
                    metadata={"adapter": "pdfplumber"},
                )
                if candidate:
                    candidates.append(candidate)
        return candidates


class CamelotTableEngine:
    def __init__(self, spec: TableEngineSpec):
        self.spec = spec

    def available(self) -> tuple[bool, str]:
        if not importlib.util.find_spec("camelot"):
            return False, "camelot 未安装"
        if not shutil.which("gs"):
            return False, "Ghostscript 命令不存在"
        return True, "camelot"

    async def test(self) -> tuple[bool, str, dict[str, Any]]:
        started = time.perf_counter()
        available, detail = self.available()
        return available, detail, {"latency_ms": round((time.perf_counter() - started) * 1000, 2)}

    async def extract(
        self,
        file_path: str,
        *,
        page_number: int,
        page_width: float,
        page_height: float,
        regions: list[LayoutRegion],
    ) -> list[TableCandidate]:
        available, detail = self.available()
        if not available:
            raise RuntimeError(detail)
        flavors = [str(self.spec.config.get("flavor") or "lattice")]
        fallback = str(self.spec.config.get("fallback_flavor") or "stream")
        if fallback and fallback not in flavors:
            flavors.append(fallback)
        for flavor in flavors:
            try:
                candidates = await asyncio.to_thread(
                    self._extract_sync,
                    file_path,
                    page_number,
                    page_width,
                    page_height,
                    regions,
                    flavor,
                )
                if candidates:
                    return candidates
            except Exception:
                continue
        return []

    def _extract_sync(
        self,
        file_path: str,
        page_number: int,
        page_width: float,
        page_height: float,
        regions: list[LayoutRegion],
        flavor: str,
    ) -> list[TableCandidate]:
        import camelot

        tables = camelot.read_pdf(file_path, pages=str(page_number), flavor=flavor)
        candidates: list[TableCandidate] = []
        from app.parsing.pdf_types import normalize_bbox

        for _index, table in enumerate(tables):
            rows = table.df.values.tolist()
            raw_bbox = getattr(table, "_bbox", None) or [0, 0, page_width, page_height]
            # Camelot reports [x0, y0, x1, y1] in bottom-left PDF coordinates.
            bbox_pixels = [raw_bbox[0], page_height - raw_bbox[3], raw_bbox[2], page_height - raw_bbox[1]]
            candidate = _candidate_from_rows(
                engine_id=self.spec.id,
                page_number=page_number,
                bbox=normalize_bbox(bbox_pixels, page_width, page_height),
                rows=rows,
                metadata={"adapter": "camelot", "flavor": flavor},
            )
            if candidate:
                candidates.append(candidate)
        return candidates


class TabulaTableEngine:
    def __init__(self, spec: TableEngineSpec):
        self.spec = spec

    def available(self) -> tuple[bool, str]:
        if not importlib.util.find_spec("tabula"):
            return False, "tabula 未安装"
        if not shutil.which("java"):
            return False, "Java 命令不存在"
        return True, "tabula"

    async def test(self) -> tuple[bool, str, dict[str, Any]]:
        started = time.perf_counter()
        available, detail = self.available()
        return available, detail, {"latency_ms": round((time.perf_counter() - started) * 1000, 2)}

    async def extract(
        self,
        file_path: str,
        *,
        page_number: int,
        page_width: float,
        page_height: float,
        regions: list[LayoutRegion],
    ) -> list[TableCandidate]:
        available, detail = self.available()
        if not available:
            raise RuntimeError(detail)
        return await asyncio.to_thread(
            self._extract_sync, file_path, page_number, regions
        )

    def _extract_sync(
        self,
        file_path: str,
        page_number: int,
        regions: list[LayoutRegion],
    ) -> list[TableCandidate]:
        import tabula

        frames = tabula.read_pdf(
            file_path,
            pages=page_number,
            multiple_tables=True,
            lattice=bool(self.spec.config.get("lattice", True)),
            stream=not bool(self.spec.config.get("lattice", True)),
        )
        candidates: list[TableCandidate] = []
        for index, frame in enumerate(frames or []):
            rows = [list(frame.columns)] + frame.fillna("").values.tolist()
            bbox = next((region.bbox for region in regions if region.type == "table"), [0, 0, 1, 1])
            candidate = _candidate_from_rows(
                engine_id=self.spec.id,
                page_number=page_number,
                bbox=bbox,
                rows=rows,
                layout_block_id=f"tabula-p{page_number}-t{index}",
                metadata={"adapter": "tabula"},
            )
            if candidate:
                candidates.append(candidate)
        return candidates


def _contains(outer: BBox, inner: BBox, *, threshold: float = 0.5) -> bool:
    x0 = max(outer[0], inner[0])
    y0 = max(outer[1], inner[1])
    x1 = min(outer[2], inner[2])
    y1 = min(outer[3], inner[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return intersection / area >= threshold


class TableExtractionManager:
    def __init__(self, engines: list[TableEngine], *, min_quality: float = 0.45):
        self.engines = engines
        self.min_quality = min_quality

    async def extract(
        self,
        file_path: str,
        *,
        page_number: int,
        page_width: float,
        page_height: float,
        regions: list[LayoutRegion],
        native_text: bool,
    ) -> tuple[list[TableCandidate], list[dict[str, Any]]]:
        candidates: list[TableCandidate] = []
        trace: list[dict[str, Any]] = []
        for engine in self.engines:
            adapter = engine.spec.adapter
            if not native_text and adapter in {"camelot", "tabula"}:
                trace.append({"engine": engine.spec.id, "skipped": "scanned_page"})
                continue
            available, detail = engine.available()
            if not available:
                trace.append({"engine": engine.spec.id, "skipped": detail})
                continue
            started = time.perf_counter()
            try:
                engine_candidates = await engine.extract(
                    file_path,
                    page_number=page_number,
                    page_width=page_width,
                    page_height=page_height,
                    regions=regions,
                )
            except Exception as exc:
                trace.append({"engine": engine.spec.id, "error": str(exc)[:500]})
                continue
            trace.append(
                {
                    "engine": engine.spec.id,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "tables": len(engine_candidates),
                    "best_score": max((item.quality_score for item in engine_candidates), default=0),
                }
            )
            candidates.extend(engine_candidates)
            if engine_candidates and max(item.quality_score for item in engine_candidates) >= self.min_quality:
                break
        accepted = [candidate for candidate in candidates if candidate.quality_score >= self.min_quality]
        return accepted, trace


def reconstruct_tables_from_ocr(
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    regions: list[LayoutRegion],
    ocr_blocks: list[dict[str, Any]],
    engine_id: str,
) -> list[TableCandidate]:
    table_regions = [region for region in regions if region.type == "table"]
    if not table_regions:
        return []
    from app.parsing.pdf_types import bbox_iou

    candidates: list[TableCandidate] = []
    for region in table_regions:
        inside = [
            block
            for block in ocr_blocks
            if bbox_iou(region.bbox, block.get("bbox") or [0, 0, 1, 1]) > 0.1
        ]
        if not inside:
            continue
        rows: dict[int, list[tuple[float, str]]] = {}
        for block in inside:
            bbox = block.get("bbox") or [0, 0, 1, 1]
            row_key = round(((bbox[1] + bbox[3]) / 2) * 20)
            rows.setdefault(row_key, []).append((bbox[0], str(block.get("text") or "")))
        values = [
            [text for _, text in sorted(items, key=lambda item: item[0])]
            for _, items in sorted(rows.items())
        ]
        candidates.append(
            _candidate_from_rows(
                engine_id=engine_id,
                page_number=page_number,
                bbox=region.bbox,
                rows=values,
                layout_block_id=region.id,
                metadata={"adapter": "ocr_coordinate", "ocr_used": True},
            )
        )
    return [candidate for candidate in candidates if candidate is not None]


_ADAPTERS = {
    "pdfplumber": PDFPlumberTableEngine,
    "camelot": CamelotTableEngine,
    "tabula": TabulaTableEngine,
}


def build_table_engine(config: dict[str, Any]) -> TableEngine | None:
    adapter = str(config.get("adapter") or "")
    engine_class = _ADAPTERS.get(adapter)
    if engine_class is None:
        return None
    spec = TableEngineSpec(
        id=str(config.get("id") or adapter),
        adapter=adapter,
        display_name=str(config.get("display_name") or adapter),
        config=dict(config.get("config") or {}),
    )
    return engine_class(spec)


def build_table_manager(configs: list[dict[str, Any]]) -> TableExtractionManager:
    engines: list[TableEngine] = []
    for config in sorted(configs, key=lambda item: (int(item.get("priority", 100)), str(item.get("id", "")))):
        engine = build_table_engine(config)
        if engine is not None:
            engines.append(engine)
    return TableExtractionManager(engines)


def table_adapter_capabilities() -> list[dict[str, Any]]:
    return [
        {"adapter": "pdfplumber", "label": "pdfplumber", "description": "原生表格和线条结构提取"},
        {"adapter": "camelot", "label": "Camelot", "description": "lattice/stream 表格提取"},
        {"adapter": "tabula", "label": "Tabula", "description": "Java 表格提取后备"},
    ]