from pathlib import Path

from app.core.errors import ValidationError
from app.parsing.base import ParsedDocument
from app.parsing.docx_parser import DocxParser
from app.parsing.epub_parser import EPUBParser
from app.parsing.markdown_parser import MarkdownParser
from app.parsing.pdf_parser import PDFParser

PARSERS = {
    ".pdf": PDFParser,
    ".docx": DocxParser,
    ".md": MarkdownParser,
    ".markdown": MarkdownParser,
    ".epub": EPUBParser,
}
SUPPORTED_EXTENSIONS = tuple(sorted(PARSERS))


def parse_document(file_path: str) -> ParsedDocument:
    extension = Path(file_path).suffix.lower()
    parser_class = PARSERS.get(extension)
    if parser_class is None:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise ValidationError(f"不支持的文件格式 {extension or '(无扩展名)'}，支持：{supported}")
    return parser_class().parse(file_path)
