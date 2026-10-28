import re
from pathlib import Path

from app.parsing.base import BaseParser, ParsedDocument, ParsedElement

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
CODE_PATTERN = re.compile(r"^```([\w+-]*)\s*$")
TABLE_ROW_PATTERN = re.compile(r"^\s*\|.*\|\s*$")
LIST_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


class MarkdownParser(BaseParser):
    extensions = (".md", ".markdown")

    def parse(self, file_path: str) -> ParsedDocument:
        raw = Path(file_path).read_text(encoding="utf-8", errors="replace")
        lines = raw.splitlines()
        front_matter: dict[str, str] = {}
        if lines and lines[0].strip() == "---":
            end = next(
                (index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"),
                None,
            )
            if end is not None:
                for line in lines[1:end]:
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    front_matter[key.strip()] = value.strip().strip('"').strip("'")
                lines = lines[end + 1 :]
        elements: list[ParsedElement] = []
        i = 0

        while i < len(lines):
            line = lines[i]
            heading = HEADING_PATTERN.match(line)
            if heading:
                elements.append(
                    ParsedElement(
                        type="title",
                        text=heading.group(2).strip(),
                        level=len(heading.group(1)),
                    )
                )
                i += 1
                continue

            code = CODE_PATTERN.match(line)
            if code:
                language = code.group(1)
                i += 1
                code_lines: list[str] = []
                while i < len(lines) and not lines[i].startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                elements.append(
                    ParsedElement(
                        type="code",
                        text="\n".join(code_lines),
                        metadata={"language": language},
                    )
                )
                i += 1 if i < len(lines) else 0
                continue

            if TABLE_ROW_PATTERN.match(line):
                table_lines: list[str] = []
                while i < len(lines) and TABLE_ROW_PATTERN.match(lines[i]):
                    table_lines.append(lines[i].strip())
                    i += 1
                elements.append(ParsedElement(type="table", text="\n".join(table_lines)))
                continue

            if LIST_PATTERN.match(line):
                list_lines: list[str] = []
                while i < len(lines) and LIST_PATTERN.match(lines[i]):
                    list_lines.append(lines[i].strip())
                    i += 1
                elements.append(ParsedElement(type="list", text="\n".join(list_lines)))
                continue

            if line.strip():
                paragraph_lines = [line.strip()]
                i += 1
                while i < len(lines) and lines[i].strip():
                    if (
                        HEADING_PATTERN.match(lines[i])
                        or CODE_PATTERN.match(lines[i])
                        or TABLE_ROW_PATTERN.match(lines[i])
                        or LIST_PATTERN.match(lines[i])
                    ):
                        break
                    paragraph_lines.append(lines[i].strip())
                    i += 1
                elements.append(ParsedElement(type="paragraph", text=" ".join(paragraph_lines)))
                continue

            i += 1

        title = front_matter.get("title") or next(
            (element.text for element in elements if element.type == "title" and element.level == 1),
            Path(file_path).stem,
        )
        metadata = {
            "format": "markdown",
            "front_matter": front_matter,
            "author": front_matter.get("author", ""),
            "description": front_matter.get("description", ""),
            "tags": front_matter.get("tags", ""),
            "heading_count": sum(1 for element in elements if element.type == "title"),
            "code_block_count": sum(1 for element in elements if element.type == "code"),
            "table_count": sum(1 for element in elements if element.type == "table"),
            "list_count": sum(1 for element in elements if element.type == "list"),
            "word_count": len(re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9_]+", raw)),
        }
        return ParsedDocument(
            source=Path(file_path).name,
            title=title,
            elements=elements,
            full_text=raw,
            metadata=metadata,
        )


