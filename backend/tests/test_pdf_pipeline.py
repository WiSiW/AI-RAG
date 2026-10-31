from pathlib import Path

import pymupdf
import pytest

from app.chunking.layout_chunker import LayoutAwareChunker
from app.core.errors import ValidationError
from app.parsing.base import ParsedDocument, ParsedElement, ParsedTable
from app.parsing.pdf_layout import LayoutAnalyzer
from app.parsing.pdf_ocr import OCREngineSpec
from app.parsing.pdf_processor import PDFProcessingPipeline
from app.parsing.pdf_types import OCRPageResult, OCRTextBlock


class FakeOCREngine:
    def __init__(self, engine_id: str, text: str, confidence: float = 0.9):
        self.spec = OCREngineSpec(
            id=engine_id,
            adapter="fake",
            display_name=engine_id,
            config={},
        )
        self.text = text
        self.confidence = confidence
        self.calls = 0

    def available(self) -> tuple[bool, str]:
        return True, "fake"

    async def test(self):
        return True, "ok", {}

    async def recognize(self, image_bytes, *, page_number, width, height, language, layout_hint):
        self.calls += 1
        return OCRPageResult(
            engine_id=self.spec.id,
            page_number=page_number,
            blocks=[OCRTextBlock(text=self.text, bbox=[0.1, 0.1, 0.9, 0.2], confidence=self.confidence)],
            full_text=self.text,
            mean_confidence=self.confidence,
        )


def _text_pdf(path: Path, text: str, *, blank: bool = False) -> None:
    document = pymupdf.open()
    page = document.new_page()
    if not blank:
        page.insert_textbox((72, 72, 520, 760), text, fontsize=12)
    document.save(path)
    document.close()

@pytest.mark.asyncio
async def test_native_pdf_does_not_call_ocr(tmp_path):
    path = tmp_path / "native.pdf"
    document_text = (
        "This guide explains installation precautions, network configuration, serial number registration, "
        "firmware updates, diagnostic indicators, and recovery steps for industrial controllers."
    )
    _text_pdf(path, document_text)
    ocr = FakeOCREngine("ocr-1", "不应调用")
    result = await PDFProcessingPipeline(
        ocr_engines=[ocr],
        layout_analyzer=LayoutAnalyzer(),
        table_manager=None,
    ).process(str(path))

    assert ocr.calls == 0
    assert result.pipeline_meta["native_pages"] == 1
    assert result.pipeline_meta["ocr_pages"] == 0
    assert "installation precautions" in result.parsed.full_text


@pytest.mark.asyncio
async def test_scanned_pdf_falls_back_to_next_ocr_engine(tmp_path):
    path = tmp_path / "scan.pdf"
    _text_pdf(path, "", blank=True)
    bad = FakeOCREngine("ocr-1", "锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷" * 5, confidence=0.2)
    good = FakeOCREngine(
        "ocr-2",
        "扫描文档识别结果：设备启动后检查电源指示灯，然后确认网络连接状态，并记录当前软件版本与序列号信息。",
        confidence=0.9,
    )

    result = await PDFProcessingPipeline(
        ocr_engines=[bad, good],
        layout_analyzer=LayoutAnalyzer(),
        table_manager=None,
    ).process(str(path))

    assert bad.calls == 1
    assert good.calls == 1
    assert result.pipeline_meta["ocr_pages"] == 1
    assert result.pipeline_meta["ocr_engine_counts"] == {"ocr-2": 1}
    assert result.parsed.page_traces[0]["selected_source"] == "ocr"
    assert len(result.parsed.page_traces[0]["ocr_attempts"]) == 2


@pytest.mark.asyncio
async def test_scanned_pdf_fails_when_all_ocr_engines_fail(tmp_path):
    path = tmp_path / "scan-failed.pdf"
    _text_pdf(path, "", blank=True)
    bad = FakeOCREngine("ocr-1", "�", confidence=0.1)
    with pytest.raises(ValidationError, match="没有可提取文本"):
        await PDFProcessingPipeline(
            ocr_engines=[bad],
            layout_analyzer=LayoutAnalyzer(),
            table_manager=None,
        ).process(str(path))


def test_layout_chunker_keeps_large_table_atomic():
    rows = [[f"产品{index}", str(index * 100)] for index in range(20)]
    table = ParsedTable(
        id="table-1",
        page_start=1,
        page_end=1,
        bbox=[0.1, 0.2, 0.9, 0.8],
        caption="季度收入表",
        headers=["产品", "收入"],
        rows=rows,
        markdown="\n".join(["| 产品 | 收入 |", "| --- | --- |", *[f"| {a} | {b} |" for a, b in rows]]),
        description="表格列出各产品收入。",
    )
    parsed = ParsedDocument(
        source="table.pdf",
        title="收入报告",
        elements=[
            ParsedElement(type="title", text="收入报告", level=1),
            ParsedElement(
                type="table",
                text=f"季度收入表\n{table.markdown}",
                bbox=table.bbox,
                metadata={"table_id": table.id, "page": 1},
            ),
        ],
        full_text=table.markdown,
        metadata={"format": "pdf"},
        tables=[table],
    )

    chunks = LayoutAwareChunker(chunk_size=64, chunk_overlap=8).split(parsed)
    table_chunks = [chunk for chunk in chunks if chunk.metadata.get("table_ids")]

    assert len(table_chunks) == 1
    assert table_chunks[0].metadata["table_ids"] == ["table-1"]
    assert "数据规模：20 行" in table_chunks[0].content
    assert "| 产品19 | 1900 |" not in table_chunks[0].content
    assert len(table.rows) == 20