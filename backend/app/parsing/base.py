from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedElement:
    type: str
    text: str
    level: int = 0
    metadata: dict = field(default_factory=dict)
    bbox: list[float] | None = None
    confidence: float | None = None
    source: str = ""


@dataclass
class ParsedTable:
    id: str
    page_start: int
    page_end: int
    bbox: list[float] = field(default_factory=list)
    layout_block_id: str = ""
    caption: str = ""
    cells: list[dict] = field(default_factory=list)
    headers: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    markdown: str = ""
    description: str = ""
    description_meta: dict = field(default_factory=dict)
    extraction_engine: str = ""
    quality_score: float = 0
    quality_json: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    source: str
    title: str
    elements: list[ParsedElement]
    full_text: str
    metadata: dict = field(default_factory=dict)
    tables: list[ParsedTable] = field(default_factory=list)
    page_traces: list[dict] = field(default_factory=list)
    pipeline_meta: dict = field(default_factory=dict)

    @property
    def total_chars(self) -> int:
        return sum(len(element.text) for element in self.elements)


class BaseParser:
    extensions: tuple[str, ...] = ()

    def parse(self, file_path: str) -> ParsedDocument:
        raise NotImplementedError

    @staticmethod
    def source_name(file_path: str) -> str:
        return Path(file_path).name
