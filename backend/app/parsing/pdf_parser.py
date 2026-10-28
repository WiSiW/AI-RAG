from pathlib import Path
from statistics import median

import pymupdf

from app.parsing.base import BaseParser, ParsedDocument, ParsedElement


class PDFParser(BaseParser):
    extensions = (".pdf",)

    def parse(self, file_path: str) -> ParsedDocument:
        document = pymupdf.open(file_path)
        try:
            lines: list[tuple[str, float, int]] = []
            for page_number, page in enumerate(document, 1):
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
                        lines.append((text, font_size, page_number))

            body_size = median([size for _, size, _ in lines if size > 0]) if lines else 10
            elements: list[ParsedElement] = []
            for text, font_size, page_number in lines:
                if font_size >= body_size * 1.35:
                    level = 1
                elif font_size >= body_size * 1.2:
                    level = 2
                elif font_size >= body_size * 1.08:
                    level = 3
                else:
                    level = 0
                elements.append(
                    ParsedElement(
                        type="title" if level else "paragraph",
                        text=text,
                        level=level,
                        metadata={"page": page_number},
                    )
                )

            raw_metadata = {
                str(key): str(value)
                for key, value in (document.metadata or {}).items()
                if value not in (None, "")
            }
            title = Path(file_path).stem
            if raw_metadata.get("title"):
                title = raw_metadata["title"]
            full_text = "\n".join(element.text for element in elements)
            metadata = {
                "format": "pdf",
                "pages": len(document),
                "page_count": len(document),
                "pdf_version": getattr(document, "pdf_version", None),
                "is_encrypted": bool(getattr(document, "is_encrypted", False)),
                "toc_items": len(document.get_toc()),
                "author": raw_metadata.get("author", ""),
                "subject": raw_metadata.get("subject", ""),
                "keywords": raw_metadata.get("keywords", ""),
                "creator": raw_metadata.get("creator", ""),
                "producer": raw_metadata.get("producer", ""),
                "creation_date": raw_metadata.get("creationDate", ""),
                "modified_date": raw_metadata.get("modDate", ""),
                "pdf_metadata": raw_metadata,
            }
            return ParsedDocument(
                source=Path(file_path).name,
                title=title,
                elements=elements,
                full_text=full_text,
                metadata=metadata,
            )
        finally:
            document.close()


