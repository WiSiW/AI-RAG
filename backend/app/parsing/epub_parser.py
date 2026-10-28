from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub

from app.parsing.base import BaseParser, ParsedDocument, ParsedElement

BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table"]


class EPUBParser(BaseParser):
    extensions = (".epub",)

    def parse(self, file_path: str) -> ParsedDocument:
        book = epub.read_epub(file_path, options={"ignore_ncx": True})
        elements: list[ParsedElement] = []
        spine_ids = [item_id for item_id, _ in getattr(book, "spine", [])]
        documents = []

        if spine_ids:
            for item_id in spine_ids:
                item = book.get_item_with_id(item_id)
                if item is not None and item.get_type() == ITEM_DOCUMENT:
                    documents.append(item)
        if not documents:
            documents = list(book.get_items_of_type(ITEM_DOCUMENT))

        for item in documents:
            soup = BeautifulSoup(item.get_content(), "lxml")
            body = soup.body or soup
            for tag in body.find_all(BLOCK_TAGS):
                if tag.find_parent(["li", "pre", "table"]) is not None and tag.name not in {
                    "pre",
                    "table",
                }:
                    continue
                text = tag.get_text(" ", strip=True)
                if not text:
                    continue
                if tag.name.startswith("h"):
                    level = int(tag.name[1])
                    elements.append(ParsedElement(type="title", text=text, level=level))
                elif tag.name == "pre":
                    elements.append(ParsedElement(type="code", text=text))
                elif tag.name == "table":
                    rows = []
                    for row in tag.find_all("tr"):
                        cells = [
                            cell.get_text(" ", strip=True)
                            for cell in row.find_all(["th", "td"])
                        ]
                        if cells:
                            rows.append(" | ".join(cells))
                    if rows:
                        elements.append(ParsedElement(type="table", text="\n".join(rows)))
                elif tag.name == "li":
                    elements.append(ParsedElement(type="list", text=text))
                else:
                    elements.append(ParsedElement(type="paragraph", text=text))

        def dc_values(name: str) -> list[str]:
            return [str(value[0]) for value in book.get_metadata("DC", name) if value and value[0]]

        title = Path(file_path).stem
        titles = dc_values("title")
        if titles:
            title = titles[0]
        full_text = "\n".join(element.text for element in elements)
        metadata = {
            "format": "epub",
            "sections": len(documents),
            "authors": dc_values("creator"),
            "language": (dc_values("language") or [""])[0],
            "publisher": (dc_values("publisher") or [""])[0],
            "identifier": (dc_values("identifier") or [""])[0],
            "date": (dc_values("date") or [""])[0],
            "description": (dc_values("description") or [""])[0],
            "subjects": dc_values("subject"),
            "chapter_count": len(documents),
        }
        return ParsedDocument(
            source=Path(file_path).name,
            title=title,
            elements=elements,
            full_text=full_text,
            metadata=metadata,
        )

