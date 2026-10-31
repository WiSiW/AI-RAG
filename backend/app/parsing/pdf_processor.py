import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from app.core.config import settings
from app.core.errors import ValidationError
from app.db.models import ModelProfile
from app.parsing.base import ParsedDocument, ParsedElement, ParsedTable
from app.parsing.pdf_layout import LayoutAnalyzer, geometry_layout
from app.parsing.pdf_ocr import OCREngine
from app.parsing.pdf_quality import evaluate_text_quality
from app.parsing.pdf_table_summary import TableSummaryService
from app.parsing.pdf_tables import TableExtractionManager, reconstruct_tables_from_ocr
from app.parsing.pdf_types import PageLayout, bbox_iou, normalize_bbox, table_to_markdown

logger = logging.getLogger(__name__)


@dataclass
class PDFProcessingResult:
    parsed: ParsedDocument
    pipeline_meta: dict[str, Any]


@dataclass
class TableCandidateWithElements:
    table: ParsedTable
    candidate: Any


class PDFProcessingPipeline:
    def __init__(
        self,
        *,
        ocr_engines: list[OCREngine] | None = None,
        layout_analyzer: LayoutAnalyzer | None = None,
        table_manager: TableExtractionManager | None = None,
        enrichment_profile: ModelProfile | None = None,
        table_summary_service: TableSummaryService | None = None,
        dpi: int | None = None,
    ):
        self.ocr_engines = ocr_engines or []
        self.layout_analyzer = layout_analyzer or LayoutAnalyzer()
        self.table_manager = table_manager
        self.enrichment_profile = enrichment_profile
        self.table_summary_service = table_summary_service or TableSummaryService()
        self.dpi = dpi or settings.pdf_ocr_dpi

    async def process(self, file_path: str) -> PDFProcessingResult:
        started = time.perf_counter()
        document = pymupdf.open(file_path)
        try:
            page_elements: list[ParsedElement] = []
            tables: list[ParsedTable] = []
            page_traces: list[dict[str, Any]] = []
            native_pages = 0
            ocr_pages = 0
            failed_pages: list[int] = []
            engine_usage: dict[str, int] = {}
            layout_sources: dict[str, int] = {}
            warnings: list[str] = []

            for page_number, page in enumerate(document, 1):
                width = float(page.rect.width)
                height = float(page.rect.height)
                native = self._native_page_elements(page, page_number, width, height)
                native_quality = evaluate_text_quality(native["text"])
                needs_ocr = not native_quality.passed
                if self._image_dominant(page) and native_quality.effective_chars < 400:
                    needs_ocr = True

                layout_available, layout_detail = self.layout_analyzer.available()
                page_image = self._render_page(page) if needs_ocr or layout_available else b""
                if page_image:
                    layout = await self.layout_analyzer.analyze(
                        page_image,
                        page_number=page_number,
                        width=width,
                        height=height,
                        native_elements=native["elements"],
                    )
                else:
                    layout = geometry_layout(
                        page_number=page_number,
                        width=width,
                        height=height,
                        native_elements=native["elements"],
                        warning=layout_detail,
                    )
                layout_sources[layout.source] = layout_sources.get(layout.source, 0) + 1

                selected_elements: list[dict[str, Any]] = []
                selected_source = "native"
                selected_quality = native_quality
                attempts: list[dict[str, Any]] = []
                if needs_ocr:
                    ocr_result, ocr_quality, attempts = await self._recognize_chain(
                        page_image,
                        page_number=page_number,
                        width=width,
                        height=height,
                        layout=layout,
                    )
                    if ocr_result is not None and ocr_quality is not None and ocr_quality.passed:
                        selected_elements = [
                            {
                                "text": block.text,
                                "bbox": block.bbox,
                                "block_type": block.block_type,
                                "confidence": block.confidence,
                                "ocr_engine": ocr_result.engine_id,
                            }
                            for block in ocr_result.blocks
                        ]
                        selected_source = "ocr"
                        selected_quality = ocr_quality
                        ocr_pages += 1
                        engine_usage[ocr_result.engine_id] = engine_usage.get(ocr_result.engine_id, 0) + 1
                    elif native_quality.effective_chars:
                        selected_elements = native["elements"]
                        selected_source = "degraded_native"
                        warnings.append(f"第{page_number}页 OCR 未通过质量检查，保留原生文本")
                    else:
                        failed_pages.append(page_number)
                        warnings.append(f"第{page_number}页没有可用文本")
                else:
                    selected_elements = native["elements"]
                    native_pages += 1

                selected_elements = self._filter_layout_noise(selected_elements, layout)
                selected_elements = self._sort_elements(selected_elements, layout)

                page_tables: list[TableCandidateWithElements] = []
                table_trace: list[dict[str, Any]] = []
                if selected_elements:
                    table_candidates, table_trace = await self._extract_tables(
                        file_path,
                        page_number=page_number,
                        width=width,
                        height=height,
                        layout=layout,
                        selected_elements=selected_elements,
                        native_text=selected_source == "native",
                    )
                    for candidate in table_candidates:
                        candidate.metadata["caption"] = self._caption_for_table(candidate, selected_elements)
                        table = self._to_parsed_table(candidate)
                        tables.append(table)
                        page_tables.append(TableCandidateWithElements(table=table, candidate=candidate))
                    warnings.extend(
                        f"第{page_number}页表格引擎 {item.get('engine')} 降级："
                        f"{item.get('error') or item.get('skipped')}"
                        for item in table_trace
                        if item.get("error") or item.get("skipped")
                    )

                if page_tables:
                    selected_elements = self._remove_table_overlap(selected_elements, page_tables)
                    for item in page_tables:
                        selected_elements.append(self._table_element(item.table))
                    selected_elements = self._sort_elements(selected_elements, layout)

                for element in selected_elements:
                    metadata = dict(element.get("metadata") or {})
                    metadata.update(
                        {
                            "page": page_number,
                            "bbox": element.get("bbox") or [],
                            "block_type": element.get("block_type") or "paragraph",
                            "ocr_engine": element.get("ocr_engine") or selected_source,
                            "ocr_confidence": element.get("confidence"),
                            "reading_order": element.get("reading_order", 0),
                        }
                    )
                    page_elements.append(
                        ParsedElement(
                            type=str(element.get("block_type") or "paragraph"),
                            text=str(element.get("text") or "").strip(),
                            level=int(element.get("level") or 0),
                            metadata=metadata,
                            bbox=element.get("bbox"),
                            confidence=element.get("confidence"),
                            source=str(element.get("ocr_engine") or selected_source),
                        )
                    )

                page_traces.append(
                    {
                        "page": page_number,
                        "native_quality": native_quality.as_dict(),
                        "selected_source": selected_source,
                        "selected_quality": selected_quality.as_dict(),
                        "ocr_attempts": attempts,
                        "layout_source": layout.source,
                        "layout_warning": layout.warning,
                        "tables": len(page_tables),
                        "table_trace": table_trace,
                    }
                )

            page_elements = [element for element in page_elements if element.text]
            if not page_elements:
                raise ValidationError("PDF 没有可提取文本；扫描页 OCR 全部失败或未启用可用 OCR 引擎")

            await self.table_summary_service.summarize(self.enrichment_profile, tables)
            full_text = "\n".join(
                element.text for element in page_elements if element.type not in {"header", "footer"}
            )
            degraded = bool(failed_pages) or any(
                trace["selected_source"] == "degraded_native" for trace in page_traces
            )
            pipeline_meta = {
                "status": "degraded" if degraded else "ready",
                "pages_total": len(document),
                "native_pages": native_pages,
                "ocr_pages": ocr_pages,
                "failed_pages": failed_pages,
                "ocr_engine_counts": engine_usage,
                "layout_sources": layout_sources,
                "tables": len(tables),
                "warnings": list(dict.fromkeys(warnings))[:50],
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "ocr_dpi": self.dpi,
            }
            title = str((document.metadata or {}).get("title") or Path(file_path).stem)
            metadata = {
                "format": "pdf",
                "pages": len(document),
                "page_count": len(document),
                "pdf_version": getattr(document, "pdf_version", None),
                "is_encrypted": bool(getattr(document, "is_encrypted", False)),
                "toc_items": len(document.get_toc()),
                "pdf_metadata": {
                    str(key): str(value)
                    for key, value in (document.metadata or {}).items()
                    if value not in (None, "")
                },
                "pdf_pipeline": pipeline_meta,
            }
            return PDFProcessingResult(
                parsed=ParsedDocument(
                    source=Path(file_path).name,
                    title=title,
                    elements=page_elements,
                    full_text=full_text,
                    metadata=metadata,
                    tables=tables,
                    page_traces=page_traces,
                    pipeline_meta=pipeline_meta,
                ),
                pipeline_meta=pipeline_meta,
            )
        finally:
            document.close()

    def _native_page_elements(
        self,
        page: pymupdf.Page,
        page_number: int,
        width: float,
        height: float,
    ) -> dict[str, Any]:
        lines: list[tuple[str, float, list[float]]] = []
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = [span for span in line.get("spans", []) if span.get("text", "").strip()]
                text = "".join(span.get("text", "") for span in spans).strip()
                if not text:
                    continue
                font_size = max((float(span.get("size", 0)) for span in spans), default=0)
                bbox = normalize_bbox(line.get("bbox"), width, height)
                lines.append((text, font_size, bbox))
        body_size = sorted(size for _, size, _ in lines if size > 0)
        median_size = body_size[len(body_size) // 2] if body_size else 10
        elements: list[dict[str, Any]] = []
        for text, font_size, bbox in lines:
            if font_size >= median_size * 1.35:
                level = 1
            elif font_size >= median_size * 1.2:
                level = 2
            elif font_size >= median_size * 1.08:
                level = 3
            else:
                level = 0
            elements.append(
                {
                    "text": text,
                    "bbox": bbox,
                    "block_type": "title" if level else "paragraph",
                    "level": level,
                    "metadata": {"page": page_number},
                }
            )
        return {"elements": elements, "text": "\n".join(item["text"] for item in elements)}

    @staticmethod
    def _image_dominant(page: pymupdf.Page) -> bool:
        page_area = max(float(page.rect.width * page.rect.height), 1.0)
        image_area = 0.0
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") == 1:
                x0, y0, x1, y1 = block.get("bbox", [0, 0, 0, 0])
                image_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)
        return image_area / page_area >= 0.35

    def _render_page(self, page: pymupdf.Page) -> bytes:
        scale = self.dpi / 72
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        return pixmap.tobytes("png")

    async def _recognize_chain(
        self,
        image_bytes: bytes,
        *,
        page_number: int,
        width: float,
        height: float,
        layout: PageLayout,
    ):
        best = None
        attempts: list[dict[str, Any]] = []
        layout_hint = ",".join(sorted({region.type for region in layout.regions}))
        for engine in self.ocr_engines:
            available, detail = engine.available()
            if not available:
                attempts.append(
                    {"engine": engine.spec.id, "adapter": engine.spec.adapter, "skipped": detail}
                )
                continue
            started = time.perf_counter()
            try:
                result = await engine.recognize(
                    image_bytes,
                    page_number=page_number,
                    width=width,
                    height=height,
                    language=str(engine.spec.config.get("language") or "chi_sim+eng"),
                    layout_hint=layout_hint,
                )
            except Exception as exc:
                attempts.append(
                    {
                        "engine": engine.spec.id,
                        "adapter": engine.spec.adapter,
                        "error": str(exc)[:500],
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                )
                continue
            quality = evaluate_text_quality(result.full_text, mean_confidence=result.mean_confidence)
            attempts.append(
                {
                    "engine": engine.spec.id,
                    "adapter": engine.spec.adapter,
                    "quality": quality.as_dict(),
                    "duration_ms": result.duration_ms,
                    "blocks": len(result.blocks),
                }
            )
            if best is None or (quality.passed, quality.score) > (best[1].passed, best[1].score):
                best = (result, quality)
            if quality.passed:
                return result, quality, attempts
        return (best[0], best[1], attempts) if best else (None, None, attempts)

    async def _extract_tables(
        self,
        file_path: str,
        *,
        page_number: int,
        width: float,
        height: float,
        layout: PageLayout,
        selected_elements: list[dict[str, Any]],
        native_text: bool,
    ):
        if self.table_manager is None:
            return [], []
        candidates, trace = await self.table_manager.extract(
            file_path,
            page_number=page_number,
            page_width=width,
            page_height=height,
            regions=layout.regions,
            native_text=native_text,
        )
        if not candidates and not native_text:
            ocr_blocks = [
                {"text": element.get("text"), "bbox": element.get("bbox")}
                for element in selected_elements
            ]
            candidates = reconstruct_tables_from_ocr(
                page_number=page_number,
                page_width=width,
                page_height=height,
                regions=layout.regions,
                ocr_blocks=ocr_blocks,
                engine_id="ocr_table_reconstruction",
            )
        return candidates, trace

    @staticmethod
    def _filter_layout_noise(
        elements: list[dict[str, Any]],
        layout: PageLayout,
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for element in elements:
            bbox = element.get("bbox") or [0, 0, 1, 1]
            region_type = ""
            for region in layout.regions:
                if bbox_iou(bbox, region.bbox) >= 0.45:
                    region_type = region.type
                    break
            if region_type in {"header", "footer", "page_number"}:
                continue
            filtered.append(element)
        return filtered

    @staticmethod
    def _sort_elements(
        elements: list[dict[str, Any]],
        layout: PageLayout,
    ) -> list[dict[str, Any]]:
        if layout.regions:
            for index, element in enumerate(elements):
                for region in layout.regions:
                    if bbox_iou(element.get("bbox") or [0, 0, 1, 1], region.bbox) >= 0.45:
                        element["reading_order"] = region.reading_order
                        break
                else:
                    element["reading_order"] = index
        return sorted(
            elements,
            key=lambda item: (
                int(item.get("reading_order") or 0),
                (item.get("bbox") or [0, 0, 0, 0])[1],
                (item.get("bbox") or [0, 0, 0, 0])[0],
            ),
        )

    @staticmethod
    def _remove_table_overlap(
        elements: list[dict[str, Any]],
        tables: list[TableCandidateWithElements],
    ) -> list[dict[str, Any]]:
        return [
            element
            for element in elements
            if not any(
                bbox_iou(element.get("bbox") or [0, 0, 1, 1], table.candidate.bbox) >= 0.35
                for table in tables
            )
        ]

    def _to_parsed_table(self, candidate) -> ParsedTable:
        caption = self._find_caption(candidate)
        return ParsedTable(
            id=str(uuid.uuid4()),
            page_start=candidate.page_start,
            page_end=candidate.page_end,
            bbox=candidate.bbox,
            layout_block_id=candidate.layout_block_id,
            caption=caption,
            cells=candidate.cells,
            headers=candidate.headers,
            rows=candidate.rows,
            markdown=candidate.markdown or table_to_markdown(candidate.headers, candidate.rows),
            extraction_engine=candidate.engine_id,
            quality_score=candidate.quality_score,
            quality_json=candidate.quality,
            metadata={**candidate.metadata, "page": candidate.page_start},
        )

    @staticmethod
    @staticmethod
    def _caption_for_table(candidate, elements: list[dict[str, Any]]) -> str:
        candidates = []
        for element in elements:
            bbox = element.get("bbox") or [0, 0, 0, 0]
            text = str(element.get("text") or "").strip()
            if not text or bbox[3] > candidate.bbox[1] + 0.01:
                continue
            distance = max(0.0, candidate.bbox[1] - bbox[3])
            if distance <= 0.08:
                candidates.append((distance, text))
        if candidates:
            text = sorted(candidates, key=lambda item: item[0])[0][1]
            if len(text) <= 200:
                return text
        return ""

    @staticmethod
    def _find_caption(candidate) -> str:
        caption = str(candidate.metadata.get("caption") or "").strip()
        if caption:
            return caption
        first_row = candidate.headers if candidate.headers else []
        if first_row and isinstance(first_row[0], list):
            first_row = first_row[0]
        values = [str(value).strip() for value in first_row if str(value).strip()]
        return "表格：" + " / ".join(values[:3]) if values else "表格"

    @staticmethod
    def _table_element(table: ParsedTable) -> dict[str, Any]:
        content = f"{table.caption}\n{table.markdown}".strip()
        return {
            "text": content,
            "bbox": table.bbox,
            "block_type": "table",
            "reading_order": 9999,
            "metadata": {
                "table_id": table.id,
                "layout_block_id": table.layout_block_id,
                "page": table.page_start,
                "bbox": table.bbox,
                "caption": table.caption,
            },
            "ocr_engine": table.extraction_engine,
            "confidence": table.quality_score,
        }