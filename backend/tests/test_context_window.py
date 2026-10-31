from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Document, DocumentChunk, DocumentStatus, DocumentTable
from app.db.session import Base
from app.generation.prompts import build_context_prompt
from app.indexing.types import RetrievedChunk
from app.retrieval.context_window import (
    build_context_prefix,
    clip_around_anchor,
    with_context_prefix,
)
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService


def test_context_prefix_and_anchor_window():
    prefix = build_context_prefix(
        "手册V3",
        "安装 > 连接电源",
        summary="本章介绍安装流程和注意事项。",
    )
    assert prefix.startswith("[文档: 手册V3 / 章节: 安装 > 连接电源")

    parent = "前文。" * 60 + "目标片段内容。" + "后文。" * 60
    window = clip_around_anchor(
        parent,
        "目标片段内容。",
        before_tokens=12,
        after_tokens=12,
    )
    assert "目标片段内容。" in window
    assert window != parent
    assert with_context_prefix(prefix, "原始小块").startswith(prefix)


def test_retrieval_keeps_small_content_and_builds_parent_and_neighbor_context():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        document = Document(
            id="doc-1",
            kb_id="kb-1",
            filename="manual.md",
            object_key="objects/manual.md",
            status=DocumentStatus.READY,
            metadata_json={"title": "手册V3"},
        )
        rows = [
            DocumentChunk(
                id="l2-prev",
                document_id=document.id,
                kb_id="kb-1",
                chunk_index=0,
                content="前一段完整内容。",
                section_path="安装",
                granularity_level="L2",
                context_prefix="[文档: 手册V3 / 章节: 安装]",
            ),
            DocumentChunk(
                id="l2-hit",
                document_id=document.id,
                kb_id="kb-1",
                chunk_index=1,
                content="连接电源前必须关闭设备。",
                section_path="安装",
                metadata_json={"parent_content": "父文档前文。连接电源前必须关闭设备。父文档后文。"},
                granularity_level="L2",
                context_prefix="[文档: 手册V3 / 章节: 安装 / 上级摘要: 安装步骤]",
            ),
            DocumentChunk(
                id="l2-next",
                document_id=document.id,
                kb_id="kb-1",
                chunk_index=2,
                content="接通电源后检查指示灯。",
                section_path="安装",
                granularity_level="L2",
                context_prefix="[文档: 手册V3 / 章节: 安装]",
            ),
        ]
        db.add_all([document, *rows])
        db.commit()

        hit = RetrievedChunk(
            chunk_id="l2-hit",
            document_id="doc-1",
            kb_id="kb-1",
            content="indexed text",
            section="安装",
            source="manual.md",
        )
        hydrated = ChatService._hydrate_chunks(db, [hit])
        expanded = ChatService._expand_context_windows(db, hydrated)

        assert expanded[0].content == "连接电源前必须关闭设备。"
        assert "【" not in expanded[0].context_text
        assert "[命中片段]" in expanded[0].context_text
        assert "[父文档局部上下文]" in expanded[0].context_text
        assert "[前一段上下文]" in expanded[0].context_text
        assert "[后一段上下文]" in expanded[0].context_text
        assert "前一段完整内容" in expanded[0].context_text
        assert "接通电源后检查指示灯" in expanded[0].context_text
    finally:
        db.close()
        engine.dispose()


def test_context_prefix_is_applied_to_l2_before_indexing():
    service = IngestionService()
    document = SimpleDocument()
    l1 = SimpleChunk(
        id="l1",
        content="安装\n安装步骤摘要",
        section_path="安装",
        granularity_level="L1",
        summary="安装步骤摘要",
    )
    l2 = SimpleChunk(
        id="l2",
        content="连接电源前必须关闭设备。",
        section_path="安装",
        granularity_level="L2",
    )

    service._apply_context_prefixes(document, [l1, l2])

    assert l2.context_prefix.startswith("[文档: 手册V3 / 章节: 安装")
    assert "上级摘要: 安装步骤摘要" in l2.context_prefix
    assert service._contextual_text(l2).startswith(l2.context_prefix)


def test_l3_uses_parent_document_window_without_replacing_atom_content():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        document = Document(
            id="doc-l3",
            kb_id="kb-1",
            filename="errors.md",
            object_key="objects/errors.md",
            status=DocumentStatus.READY,
        )
        parent = DocumentChunk(
            id="l2-parent",
            document_id=document.id,
            kb_id="kb-1",
            chunk_index=0,
            content="父段前文。ERROR 500 表示服务异常。父段后文。",
            section_path="故障排查",
            granularity_level="L2",
        )
        child = DocumentChunk(
            id="l3-child",
            document_id=document.id,
            kb_id="kb-1",
            chunk_index=1,
            content="ERROR 500 表示服务异常。",
            section_path="故障排查",
            granularity_level="L3",
            granule_type="error_line",
            parent_chunk_id=parent.id,
            context_prefix="[文档: 错误手册 / 章节: 故障排查 / 父段摘要: 异常处理]",
        )
        db.add_all([document, parent, child])
        db.commit()

        hit = RetrievedChunk(
            chunk_id=child.id,
            document_id=document.id,
            kb_id="kb-1",
            content="indexed",
            section=child.section_path,
            source=document.filename,
        )
        expanded = ChatService._expand_context_windows(
            db, ChatService._hydrate_chunks(db, [hit])
        )[0]

        assert expanded.content == child.content
        assert "父段前文" in expanded.context_text
        assert "父段后文" in expanded.context_text
        assert expanded.metadata["context_parent_included"] is True
    finally:
        db.close()
        engine.dispose()


def test_prompt_uses_expanded_context():
    chunk = RetrievedChunk(
        chunk_id="c1",
        document_id="d1",
        kb_id="kb1",
        content="小块正文",
        section="第一章",
        source="demo.md",
        metadata={"granularity_level": "L2", "context_prefix": "[文档: demo]"},
        context_text="[文档: demo]\n[命中片段]\n小块正文\n[前一段上下文]\n前文",
    )
    prompt = build_context_prompt([chunk])
    assert "[前一段上下文]" in prompt
    assert "粒度：L2" in prompt


class SimpleDocument:
    filename = "manual.md"
    metadata_json = {"title": "手册V3"}


class SimpleChunk:
    def __init__(
        self,
        *,
        id: str,
        content: str,
        section_path: str,
        granularity_level: str,
        summary: str = "",
    ):
        self.id = id
        self.content = content
        self.section_path = section_path
        self.granularity_level = granularity_level
        self.summary = summary
        self.context_prefix = ""
def test_hit_chunk_injects_linked_table_context():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        document = Document(
            id="doc-table",
            kb_id="kb-table",
            filename="report.pdf",
            object_key="objects/report.pdf",
            status=DocumentStatus.READY,
            metadata_json={"title": "季度报告"},
        )
        chunk = DocumentChunk(
            id="chunk-table",
            document_id=document.id,
            kb_id="kb-table",
            chunk_index=0,
            content="销售收入见下表。",
            section_path="经营情况",
            granularity_level="L2",
            metadata_json={"table_ids": ["table-1"]},
        )
        table = DocumentTable(
            id="table-1",
            document_id=document.id,
            kb_id="kb-table",
            page_start=2,
            page_end=2,
            caption="季度收入",
            headers=["产品", "收入"],
            rows=[["产品A", "500"]],
            markdown="| 产品 | 收入 |\n| --- | --- |\n| 产品A | 500 |",
            description="产品A收入为500万美元。",
            extraction_engine="pdfplumber",
            quality_score=0.9,
        )
        db.add_all([document, chunk, table])
        db.commit()

        hit = RetrievedChunk(
            chunk_id=chunk.id,
            document_id=document.id,
            kb_id=document.kb_id,
            content="indexed",
            section=chunk.section_path,
            source=document.filename,
        )
        expanded = ChatService._expand_context_windows(db, ChatService._hydrate_chunks(db, [hit]))

        assert "[关联表格]" in expanded[0].context_text
        assert "| 产品A | 500 |" in expanded[0].context_text
        assert expanded[0].metadata["context_tables"][0]["injected_mode"] == "full"
        assert expanded[0].metadata["context_tables"][0]["id"] == "table-1"
    finally:
        db.close()
        engine.dispose()