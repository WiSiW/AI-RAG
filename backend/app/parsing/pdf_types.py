from dataclasses import dataclass, field
from typing import Any

BBox = list[float]


@dataclass
class LayoutRegion:
    id: str
    type: str
    bbox: BBox
    confidence: float = 0
    reading_order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageLayout:
    page_number: int
    width: float
    height: float
    regions: list[LayoutRegion] = field(default_factory=list)
    source: str = "geometry"
    warning: str = ""


@dataclass
class OCRTextBlock:
    text: str
    bbox: BBox = field(default_factory=list)
    confidence: float | None = None
    block_type: str = "paragraph"


@dataclass
class OCRPageResult:
    engine_id: str
    page_number: int
    blocks: list[OCRTextBlock] = field(default_factory=list)
    full_text: str = ""
    mean_confidence: float | None = None
    duration_ms: float = 0
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRQuality:
    passed: bool
    score: float
    effective_chars: int
    mean_confidence: float | None
    corruption_ratio: float
    repetition_ratio: float
    language_ratio: float
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 4),
            "effective_chars": self.effective_chars,
            "mean_confidence": self.mean_confidence,
            "corruption_ratio": round(self.corruption_ratio, 4),
            "repetition_ratio": round(self.repetition_ratio, 4),
            "language_ratio": round(self.language_ratio, 4),
            "reasons": list(self.reasons),
        }


@dataclass
class TableCandidate:
    engine_id: str
    page_start: int
    page_end: int
    bbox: BBox
    cells: list[dict[str, Any]]
    headers: list[Any] = field(default_factory=list)
    rows: list[Any] = field(default_factory=list)
    markdown: str = ""
    caption: str = ""
    quality_score: float = 0
    quality: dict[str, Any] = field(default_factory=dict)
    layout_block_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageText:
    page_number: int
    elements: list[dict[str, Any]]
    text: str
    quality: OCRQuality
    source: str = "native"
    engine_id: str = "native"


def normalize_bbox(bbox: Any, width: float, height: float) -> BBox:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return [0.0, 0.0, 1.0, 1.0]
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return [0.0, 0.0, 1.0, 1.0]
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0:
        return [max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)]
    width = max(width, 1.0)
    height = max(height, 1.0)
    return [
        max(0.0, min(1.0, x0 / width)),
        max(0.0, min(1.0, y0 / height)),
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
    ]


def bbox_iou(left: BBox, right: BBox) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if intersection <= 0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _escape_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", "<br>").strip()


def table_to_markdown(headers: list[Any], rows: list[Any]) -> str:
    normalized_headers: list[str]
    if headers and isinstance(headers[0], list):
        normalized_headers = [_escape_cell(value) for value in headers[0]]
    else:
        normalized_headers = [_escape_cell(value) for value in headers]
    width = max(
        [len(normalized_headers), *[len(row) for row in rows if isinstance(row, list)]],
        default=0,
    )
    if width == 0:
        return ""
    normalized_headers = normalized_headers + [""] * (width - len(normalized_headers))
    lines = [
        "| " + " | ".join(normalized_headers) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for row in rows:
        values = [_escape_cell(value) for value in (row if isinstance(row, list) else [])]
        values = values + [""] * (width - len(values))
        lines.append("| " + " | ".join(values[:width]) + " |")
    return "\n".join(lines)