from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedElement:
    type: str
    text: str
    level: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    source: str
    title: str
    elements: list[ParsedElement]
    full_text: str
    metadata: dict = field(default_factory=dict)

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
