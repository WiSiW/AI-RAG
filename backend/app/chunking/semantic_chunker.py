import re
from dataclasses import dataclass, field

from app.parsing.base import ParsedDocument, ParsedElement

TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]")


def token_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def count_tokens(text: str) -> int:
    return len(token_spans(text))


@dataclass
class ChunkData:
    content: str
    section_path: str
    token_count: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class SemanticChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        if chunk_size < 32:
            raise ValueError("chunk_size 不能小于 32")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, parsed_doc: ParsedDocument) -> list[ChunkData]:
        sections = self._group_sections(parsed_doc.elements)
        chunks: list[ChunkData] = []
        for section_path, elements in sections:
            section_chunks = self._split_section(section_path, elements)
            for chunk in section_chunks:
                chunk.chunk_index = len(chunks)
                chunks.append(chunk)
        return chunks

    def _group_sections(
        self, elements: list[ParsedElement]
    ) -> list[tuple[str, list[ParsedElement]]]:
        sections: list[tuple[str, list[ParsedElement]]] = []
        stack: list[str] = []
        buffer: list[ParsedElement] = []

        def flush() -> None:
            if not buffer:
                return
            path = " > ".join(stack) if stack else "正文"
            sections.append((path, list(buffer)))
            buffer.clear()

        for element in elements:
            if element.type == "title":
                flush()
                while stack and len(stack) >= element.level:
                    stack.pop()
                stack.append(element.text)
                buffer.append(element)
            else:
                buffer.append(element)
        flush()
        return sections

    def _split_section(
        self, section_path: str, elements: list[ParsedElement]
    ) -> list[ChunkData]:
        units = [self._render_element(element) for element in elements]
        pages = sorted(
            {
                int(element.metadata["page"])
                for element in elements
                if element.metadata.get("page") is not None
            }
        )
        full_text = "\n".join(text for text in units if text).strip()
        if not full_text:
            return []
        if count_tokens(full_text) <= self.chunk_size:
            return [
                ChunkData(
                    content=full_text,
                    section_path=section_path,
                    token_count=count_tokens(full_text),
                    chunk_index=0,
                    metadata={"pages": pages} if pages else {},
                )
            ]

        chunks: list[ChunkData] = []
        current_units: list[str] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current_units, current_tokens
            if not current_units:
                return
            content = "\n".join(current_units).strip()
            if content:
                chunks.append(
                    ChunkData(
                        content=content,
                        section_path=section_path,
                        token_count=count_tokens(content),
                        chunk_index=0,
                        metadata={"pages": pages} if pages else {},
                    )
                )
            current_units = []
            current_tokens = 0

        for unit in units:
            if not unit:
                continue
            unit_tokens = count_tokens(unit)
            if unit_tokens > self.chunk_size:
                flush()
                for piece in self._split_long_text(unit):
                    chunks.append(
                        ChunkData(
                            content=piece,
                            section_path=section_path,
                            token_count=count_tokens(piece),
                            chunk_index=0,
                            metadata={"pages": pages, "split": "token_window"} if pages else {"split": "token_window"},
                        )
                    )
                continue
            if current_units and current_tokens + unit_tokens > self.chunk_size:
                previous = "\n".join(current_units)
                flush()
                overlap_limit = min(self.chunk_overlap, self.chunk_size - unit_tokens)
                overlap = self._tail_tokens(previous, overlap_limit)
                if overlap:
                    current_units = [overlap]
                    current_tokens = count_tokens(overlap)
            current_units.append(unit)
            current_tokens += unit_tokens
        flush()
        return chunks

    @staticmethod
    def _render_element(element: ParsedElement) -> str:
        if element.type == "title":
            return f"【{element.text}】"
        return element.text.strip()

    def _split_long_text(self, text: str) -> list[str]:
        spans = token_spans(text)
        if not spans:
            return []
        step = self.chunk_size - self.chunk_overlap
        pieces: list[str] = []
        start_token = 0
        while start_token < len(spans):
            end_token = min(start_token + self.chunk_size, len(spans))
            start_char = spans[start_token][0]
            end_char = spans[end_token - 1][1]
            pieces.append(text[start_char:end_char].strip())
            if end_token >= len(spans):
                break
            start_token += step
        return [piece for piece in pieces if piece]

    @staticmethod
    def _tail_tokens(text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        spans = token_spans(text)
        if not spans:
            return ""
        start_char = spans[max(0, len(spans) - limit)][0]
        return text[start_char:].strip()
