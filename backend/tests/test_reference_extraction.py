import json
import zipfile
from pathlib import Path

import pytest
from docx import Document as DocxDocument

from app.chunking.modes import split_document
from app.parsing.base import ParsedDocument, ParsedElement, ParsedReference
from app.parsing.docx_parser import DocxParser
from app.parsing.markdown_parser import MarkdownParser
from app.services.reference_extraction_service import ReferenceExtractionService


class FakeGateway:
    def __init__(self, response: str | Exception):
        self.response = response
        self.calls = 0

    async def complete(self, profile, messages, **kwargs):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class Profile:
    id = "profile-1"
    model_name = "fake-model"


def parsed_document(elements: list[ParsedElement], references=None) -> ParsedDocument:
    return ParsedDocument(
        source="demo.md",
        title="测试",
        elements=elements,
        full_text="\n".join(element.text for element in elements),
        references=references or [],
    )


@pytest.mark.asyncio
async def test_markdown_reference_section_and_inline_markers_are_filtered(tmp_path: Path):
    path = tmp_path / "demo.md"
    path.write_text(
        "# 标题\n\n"
        "正文结论[1]，并有作者观点（Smith, 2020）。\n\n"
        "## 参考文献\n\n"
        "[1] 张三. RAG 系统研究[J]. 软件学报, 2024.\n"
        "[2] Smith J. Retrieval systems. 2020. DOI: 10.1000/demo\n",
        encoding="utf-8",
    )
    parsed = MarkdownParser().parse(str(path))
    result = await ReferenceExtractionService().extract(parsed)

    assert "参考文献" not in [element.text for element in result.parsed.elements]
    assert "[1]" not in result.parsed.full_text
    assert "Smith, 2020" not in result.parsed.full_text
    assert "正文结论" in result.parsed.full_text
    assert len(result.references) == 2
    assert result.references[0].label == "[1]"
    assert result.references[0].source_type == "bibliography"
    assert result.references[0].section_path.endswith("参考文献")
    assert result.meta["inline_markers_removed"] == 2
    assert result.meta["status"] == "ready"
    chunks = await split_document(result.parsed, mode="structure", chunk_size=128, chunk_overlap=16)
    assert chunks
    assert all("参考条目" not in chunk.content for chunk in chunks)


@pytest.mark.asyncio
async def test_inline_cleaning_preserves_code_links_and_parameter_access():
    document = parsed_document(
        [
            ParsedElement(type="title", text="标题", level=1),
            ParsedElement(type="paragraph", text="说明[1]。访问 https://example.com/a[1] 获取详情。"),
            ParsedElement(type="paragraph", text="参数 items[1] 保持不变，参见[文档](https://example.com/docs)。"),
            ParsedElement(type="paragraph", text="PDF 上标正文2。", metadata={"is_superscript": True}),
            ParsedElement(type="code", text="value = items[1]  # [1]"),
        ]
    )
    result = await ReferenceExtractionService().extract(document)

    assert "说明。" in result.parsed.full_text
    assert "https://example.com/a[1]" in result.parsed.full_text
    assert "items[1]" in result.parsed.full_text
    assert "PDF 上标正文。" in result.parsed.full_text
    assert "[文档](https://example.com/docs)" in result.parsed.full_text
    assert "value = items[1]  # [1]" in result.parsed.full_text


@pytest.mark.asyncio
async def test_clear_reference_heading_does_not_call_model():
    gateway = FakeGateway("should not be called")
    document = parsed_document(
        [
            ParsedElement(type="title", text="正文", level=1),
            ParsedElement(type="paragraph", text="这是正文。"),
            ParsedElement(type="title", text="References", level=1),
            ParsedElement(type="paragraph", text="[1] Alpha. 2024."),
        ]
    )
    result = await ReferenceExtractionService().extract(
        document,
        profile=Profile(),
        gateway=gateway,
    )

    assert gateway.calls == 0
    assert "# 正文" in result.parsed.full_text
    assert "这是正文。" in result.parsed.full_text
    assert "References" not in result.parsed.full_text
    assert len(result.references) == 1


@pytest.mark.asyncio
async def test_ambiguous_tail_uses_model_suffix_classification():
    gateway = FakeGateway(
        json.dumps(
            [{"id": "e3", "role": "reference"}, {"id": "e4", "role": "reference"}],
            ensure_ascii=False,
        )
    )
    document = parsed_document(
        [
            ParsedElement(type="title", text="报告", level=1),
            ParsedElement(type="paragraph", text="正文说明。"),
            ParsedElement(type="paragraph", text="继续说明。"),
            ParsedElement(type="paragraph", text="[1] Alpha. 2024."),
            ParsedElement(type="paragraph", text="[2] Beta. 2023."),
        ]
    )
    result = await ReferenceExtractionService().extract(
        document,
        profile=Profile(),
        gateway=gateway,
    )

    assert gateway.calls == 1
    assert result.meta["generated_by"] == "model"
    assert "Alpha" not in result.parsed.full_text
    assert "Beta" not in result.parsed.full_text
    assert len(result.references) == 2


@pytest.mark.asyncio
async def test_model_failure_keeps_ambiguous_tail_as_body():
    gateway = FakeGateway(RuntimeError("model unavailable"))
    document = parsed_document(
        [
            ParsedElement(type="title", text="报告", level=1),
            ParsedElement(type="paragraph", text="正文说明。"),
            ParsedElement(type="paragraph", text="继续说明。"),
            ParsedElement(type="paragraph", text="[1] Alpha. 2024."),
            ParsedElement(type="paragraph", text="[2] Beta. 2023."),
        ]
    )
    result = await ReferenceExtractionService().extract(
        document,
        profile=Profile(),
        gateway=gateway,
    )

    assert result.meta["status"] == "degraded"
    assert "reference_model_unavailable" in result.meta["warnings"]
    assert "Alpha" in result.parsed.full_text
    assert "Beta" in result.parsed.full_text


@pytest.mark.asyncio
async def test_pdf_footnote_block_is_removed_and_stored_as_reference():
    document = ParsedDocument(
        source="demo.pdf",
        title="PDF",
        elements=[
            ParsedElement(type="title", text="第一章", level=1),
            ParsedElement(type="paragraph", text="正文内容。"),
            ParsedElement(
                type="footnote",
                text="[1] 脚注内容。",
                metadata={"page": 2},
                bbox=[0.1, 0.9, 0.9, 0.95],
            ),
        ],
        full_text="第一章\n正文内容。\n[1] 脚注内容。",
        references=[],
    )
    result = await ReferenceExtractionService().extract(document)

    assert result.parsed.full_text == "第一章\n正文内容。"
    assert len(result.references) == 1
    assert result.references[0].label == "[1]"
    assert result.references[0].text == "脚注内容。"
    assert result.references[0].page_start == 2


@pytest.mark.asyncio
async def test_markdown_footnote_definition_is_extracted(tmp_path: Path):
    path = tmp_path / "footnote.md"
    path.write_text("# 标题\n\n正文[^a]。\n\n[^a]: 脚注定义内容。\n", encoding="utf-8")
    parsed = MarkdownParser().parse(str(path))
    result = await ReferenceExtractionService().extract(parsed)

    assert "脚注定义内容" not in result.parsed.full_text
    assert "[^a]" not in result.parsed.full_text
    assert parsed.references[0].label == "a"
    assert result.references[0].source_type == "footnote"


def test_docx_footnotes_xml_is_extracted(tmp_path: Path):
    path = tmp_path / "footnote.docx"
    document = DocxDocument()
    document.add_paragraph("正文")
    document.save(path)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(
            "word/footnotes.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:footnote w:id="0"><w:p><w:r><w:t>separator</w:t></w:r></w:p></w:footnote>
  <w:footnote w:id="1"><w:p><w:r><w:t>测试脚注</w:t></w:r></w:p></w:footnote>
</w:footnotes>""",
        )
    parsed = DocxParser().parse(str(path))

    assert len(parsed.references) == 1
    assert parsed.references[0].source_type == "footnote"
    assert parsed.references[0].text == "测试脚注"


@pytest.mark.asyncio
async def test_parsed_document_native_references_survive_extraction():
    document = parsed_document(
        [
            ParsedElement(type="title", text="标题", level=1),
            ParsedElement(type="paragraph", text="正文。"),
        ],
        references=[ParsedReference(label="1", text="参考条目", source_type="endnote")],
    )
    result = await ReferenceExtractionService().extract(document)

    assert len(result.references) == 1
    assert result.references[0].text == "参考条目"
    assert result.meta["endnote_count"] == 1

@pytest.mark.asyncio
async def test_reference_only_document_has_no_usable_body():
    document = parsed_document(
        [
            ParsedElement(type="title", text="参考文档", level=1),
            ParsedElement(type="title", text="参考文献", level=1),
            ParsedElement(type="paragraph", text="[1] Alpha. 2024."),
        ]
    )
    service = ReferenceExtractionService()
    result = await service.extract(document)

    assert result.references
    assert service.has_usable_body(result.parsed) is False



@pytest.mark.asyncio
async def test_invalid_model_json_keeps_ambiguous_tail_as_body():
    gateway = FakeGateway("not valid json")
    document = parsed_document(
        [
            ParsedElement(type="title", text="报告", level=1),
            ParsedElement(type="paragraph", text="正文说明。"),
            ParsedElement(type="paragraph", text="继续说明。"),
            ParsedElement(type="paragraph", text="[1] Alpha. 2024."),
            ParsedElement(type="paragraph", text="[2] Beta. 2023."),
        ]
    )
    result = await ReferenceExtractionService().extract(
        document,
        profile=Profile(),
        gateway=gateway,
    )

    assert result.meta["status"] == "degraded"
    assert "reference_model_unavailable" in result.meta["warnings"]
    assert "Alpha" in result.parsed.full_text


