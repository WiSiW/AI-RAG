from typing import Any

from app.chunking.semantic_chunker import ChunkData, count_tokens, token_spans
from app.parsing.base import ParsedDocument, ParsedElement, ParsedTable


class LayoutAwareChunker:
    """PDF-only chunker that treats tables as atomic semantic blocks."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        if chunk_size < 32:
            raise ValueError("chunk_size 不能小于 32")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tables: dict[str, ParsedTable] = {}

    def split(self, parsed_doc: ParsedDocument) -> list[ChunkData]:
        self.tables = {table.id: table for table in parsed_doc.tables}
        chunks: list[ChunkData] = []
        for section_path, elements in self._group_sections(parsed_doc.elements):
            chunks.extend(self._split_section(section_path, elements))
        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
        return chunks

    @staticmethod
    def _group_sections(elements: list[ParsedElement]) -> list[tuple[str, list[ParsedElement]]]:
        sections: list[tuple[str, list[ParsedElement]]] = []
        stack: list[str] = []
        buffer: list[ParsedElement] = []

        def flush() -> None:
            if buffer:
                sections.append((" > ".join(stack) if stack else "正文", list(buffer)))
                buffer.clear()

        for element in elements:
            if element.type == "title":
                flush()
                level = max(1, element.level)
                while stack and len(stack) >= level:
                    stack.pop()
                stack.append(element.text)
                buffer.append(element)
            else:
                buffer.append(element)
        flush()
        return sections

    def _split_section(self, section_path: str, elements: list[ParsedElement]) -> list[ChunkData]:
        chunks: list[ChunkData] = []
        current: list[ParsedElement] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current, current_tokens
            if not current:
                return
            content = "\n".join(self._render_element(element) for element in current).strip()
            if content:
                chunks.append(self._make_chunk(section_path, current, content))
            current = []
            current_tokens = 0

        for element in elements:
            if element.type == "table":
                table_id = str(element.metadata.get("table_id") or "")
                table = self.tables.get(table_id)
                table_content = element.text.strip()
                table_tokens = count_tokens(table_content)
                if current and (table_tokens > self.chunk_size or current_tokens + table_tokens > self.chunk_size):
                    flush()
                if table_tokens > self.chunk_size and table is not None:
                    compressed = self._compressed_table_text(table)
                    chunks.append(self._make_chunk(section_path, [element], compressed))
                    continue
                current.append(element)
                current_tokens += table_tokens
                flush()
                continue

            text = self._render_element(element)
            tokens = count_tokens(text)
            if tokens > self.chunk_size:
                flush()
                for piece in self._split_long_text(text):
                    chunks.append(self._make_chunk(section_path, [element], piece))
                continue
            if current and current_tokens + tokens > self.chunk_size:
                flush()
            current.append(element)
            current_tokens += tokens
        flush()
        return chunks

    def _make_chunk(self, section_path: str, elements: list[ParsedElement], content: str) -> ChunkData:
        table_ids = [
            str(element.metadata.get("table_id"))
            for element in elements
            if element.type == "table" and element.metadata.get("table_id")
        ]
        pages = sorted(
            {
                int(element.metadata.get("page"))
                for element in elements
                if element.metadata.get("page") is not None
            }
        )
        bboxes = [
            element.bbox or element.metadata.get("bbox")
            for element in elements
            if element.bbox or element.metadata.get("bbox")
        ]
        block_types = list(dict.fromkeys(element.type for element in elements))
        ocr_engines = list(
            dict.fromkeys(
                str(element.metadata.get("ocr_engine"))
                for element in elements
                if element.metadata.get("ocr_engine")
            )
        )
        confidences = [float(element.confidence) for element in elements if element.confidence is not None]
        metadata: dict[str, Any] = {
            "chunk_mode": "pdf_layout",
            "split": "layout_aware",
            "pages": pages,
            "bboxes": bboxes,
            "layout_block_ids": [
                str(element.metadata.get("layout_block_id") or "")
                for element in elements
                if element.metadata.get("layout_block_id")
            ],
            "block_types": block_types,
            "ocr_engines": ocr_engines,
            "ocr_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
            "table_ids": table_ids,
        }
        return ChunkData(
            content=content,
            section_path=section_path,
            token_count=count_tokens(content),
            chunk_index=0,
            metadata=metadata,
        )

    def _compressed_table_text(self, table: ParsedTable) -> str:
        headers = table.headers[0] if table.headers and isinstance(table.headers[0], list) else table.headers
        header_text = " | ".join(str(value).strip() for value in headers if str(value).strip())
        return (
            f"{table.caption or '表格'}\n"
            f"表头：{header_text or '未识别'}\n"
            f"数据规模：{len(table.rows)} 行\n"
            f"表格说明：{table.description or '完整表格已独立存储'}"
        )

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
        start = 0
        while start < len(spans):
            end = min(start + self.chunk_size, len(spans))
            pieces.append(text[spans[start][0] : spans[end - 1][1]].strip())
            if end >= len(spans):
                break
            start += step
        return [piece for piece in pieces if piece]


def should_use_layout_chunker(parsed_doc: ParsedDocument, mode: str) -> bool:
    return mode == "structure" and parsed_doc.metadata.get("format") == "pdf"