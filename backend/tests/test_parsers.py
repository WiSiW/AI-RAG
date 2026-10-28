from pathlib import Path

from ebooklib import epub

from app.parsing.markdown_parser import MarkdownParser
from app.parsing.registry import parse_document


def test_markdown_parser_preserves_structure(tmp_path: Path):
    path = tmp_path / "demo.md"
    path.write_text(
        "# 标题\n\n正文内容。\n\n## 小节\n\n```python\nprint('ok')\n```\n",
        encoding="utf-8",
    )
    parsed = MarkdownParser().parse(str(path))
    assert parsed.title == "标题"
    assert [element.type for element in parsed.elements] == ["title", "paragraph", "title", "code"]
    assert parsed.elements[-1].metadata["language"] == "python"


def test_epub_parser_reads_spine(tmp_path: Path):
    book = epub.EpubBook()
    book.set_identifier("test-epub")
    book.set_title("测试电子书")
    book.set_language("zh")
    chapter = epub.EpubHtml(title="第一章", file_name="chapter.xhtml", lang="zh")
    chapter.content = "<h1>第一章</h1><p>这是 EPUB 正文。</p><h2>小节</h2><ul><li>要点一</li></ul>"
    book.add_item(chapter)
    book.toc = (chapter,)
    book.spine = ["nav", chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    path = tmp_path / "demo.epub"
    epub.write_epub(str(path), book)

    parsed = parse_document(str(path))
    assert parsed.title == "测试电子书"
    assert any(element.text == "第一章" and element.type == "title" for element in parsed.elements)
    assert any("EPUB 正文" in element.text for element in parsed.elements)
