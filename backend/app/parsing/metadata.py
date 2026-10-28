import hashlib
import mimetypes
import re
from datetime import UTC, datetime
from pathlib import Path

from app.parsing.base import ParsedDocument


def sha256_file(file_path: str, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as source:
        while chunk := source.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def detect_language(text: str) -> str:
    sample = text[:20000]
    chinese = len(re.findall(r"[\u3400-\u9fff]", sample))
    latin = len(re.findall(r"[A-Za-z]", sample))
    if chinese == 0 and latin == 0:
        return "unknown"
    if chinese >= latin / 2:
        return "zh"
    return "en"


def build_document_metadata(file_path: str, parsed: ParsedDocument) -> dict:
    path = Path(file_path)
    stat = path.stat()
    mime_type, _ = mimetypes.guess_type(path.name)
    metadata = {
        "format": parsed.metadata.get("format", path.suffix.lower().lstrip(".")),
        "title": parsed.title,
        "extension": path.suffix.lower(),
        "mime_type": mime_type or "application/octet-stream",
        "file_size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        "sha256": sha256_file(file_path),
        "language": detect_language(parsed.full_text),
        "characters": parsed.total_chars,
        "elements": len(parsed.elements),
        "headings": [
            element.text
            for element in parsed.elements
            if element.type == "title" and element.text.strip()
        ][:200],
        **parsed.metadata,
    }
    return metadata
