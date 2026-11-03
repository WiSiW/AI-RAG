import zipfile
from pathlib import Path
from xml.etree import ElementTree

from docx import Document as DocxDocument
from docx.document import Document as DocxDocumentType
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.parsing.base import BaseParser, ParsedDocument, ParsedElement, ParsedReference


def _iter_blocks(document: DocxDocumentType):
    if hasattr(document, "iter_inner_content"):
        yield from document.iter_inner_content()
        return
    parent = document.element.body
    for child in parent.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, document)
        elif child.tag.endswith("}tbl"):
            yield Table(child, document)


def _extract_docx_notes(file_path: str) -> list[ParsedReference]:
    references: list[ParsedReference] = []
    try:
        with zipfile.ZipFile(file_path) as archive:
            names = set(archive.namelist())
            for member, source_type in (
                ("word/footnotes.xml", "footnote"),
                ("word/endnotes.xml", "endnote"),
            ):
                if member not in names:
                    continue
                root = ElementTree.fromstring(archive.read(member))
                for note in root:
                    local_name = note.tag.rsplit("}", 1)[-1]
                    if local_name not in {"footnote", "endnote"}:
                        continue
                    note_id = next(
                        (
                            value
                            for key, value in note.attrib.items()
                            if key.rsplit("}", 1)[-1] == "id"
                        ),
                        "",
                    )
                    if note_id in {"-1", "0"}:
                        continue
                    text = " ".join(
                        part.strip()
                        for part in note.itertext()
                        if part and part.strip()
                    )
                    if not text:
                        continue
                    references.append(
                        ParsedReference(
                            label=f"{source_type}-{note_id}" if note_id else "",
                            text=text,
                            source_type=source_type,
                            metadata={"format": "docx", "note_id": note_id},
                        )
                    )
    except (OSError, zipfile.BadZipFile, ElementTree.ParseError):
        return []
    return references


class DocxParser(BaseParser):
    extensions = (".docx",)

    def parse(self, file_path: str) -> ParsedDocument:
        document = DocxDocument(file_path)
        elements: list[ParsedElement] = []

        for block in _iter_blocks(document):
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if not text:
                    continue
                style_name = (block.style.name if block.style else "").lower()
                level = 0
                if style_name.startswith("heading"):
                    suffix = style_name.replace("heading", "").strip()
                    level = int(suffix) if suffix.isdigit() else 1
                elif style_name == "title":
                    level = 1
                elements.append(
                    ParsedElement(
                        type="title" if level else "paragraph",
                        text=text,
                        level=level,
                        metadata={"style": style_name},
                    )
                )
            elif isinstance(block, Table):
                rows = []
                for row in block.rows:
                    cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    rows.append(" | ".join(cells))
                if rows:
                    elements.append(ParsedElement(type="table", text="\n".join(rows)))

        full_text = "\n".join(element.text for element in elements)
        props = document.core_properties
        metadata = {
            "format": "docx",
            "author": props.author or "",
            "last_modified_by": props.last_modified_by or "",
            "subject": props.subject or "",
            "keywords": props.keywords or "",
            "category": props.category or "",
            "comments": props.comments or "",
            "identifier": props.identifier or "",
            "language": props.language or "",
            "revision": props.revision,
            "created": props.created.isoformat() if props.created else "",
            "modified": props.modified.isoformat() if props.modified else "",
            "last_printed": props.last_printed.isoformat() if props.last_printed else "",
            "paragraph_count": sum(1 for item in elements if item.type != "table"),
            "table_count": sum(1 for item in elements if item.type == "table"),
            "section_count": len(document.sections),
        }
        return ParsedDocument(
            source=Path(file_path).name,
            title=props.title or Path(file_path).stem,
            elements=elements,
            full_text=full_text,
            metadata=metadata,
            references=_extract_docx_notes(file_path),
        )
