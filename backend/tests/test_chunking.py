from app.chunking.semantic_chunker import SemanticChunker, count_tokens
from app.parsing.base import ParsedDocument, ParsedElement


def test_count_tokens_handles_chinese_and_words():
    assert count_tokens("\u4f60\u597d RAG 2026") == 4


def test_chunker_tracks_section_path_and_respects_size():
    paragraph = "这是一个用于测试分块的段落。" * 30
    document = ParsedDocument(
        source="demo.md",
        title="测试",
        elements=[
            ParsedElement(type="title", text="第一章", level=1),
            ParsedElement(type="paragraph", text=paragraph),
            ParsedElement(type="title", text="1.1 小节", level=2),
            ParsedElement(type="paragraph", text="小节内容"),
        ],
        full_text=paragraph,
    )
    chunks = SemanticChunker(chunk_size=80, chunk_overlap=16).split(document)
    assert len(chunks) >= 3
    assert chunks[-1].section_path == "第一章 > 1.1 小节"
    assert all(chunk.token_count <= 80 for chunk in chunks)
    assert chunks[0].section_path == "第一章"


def test_long_text_has_overlap():
    text = "".join(f"词{i}" for i in range(300))
    document = ParsedDocument(
        source="long.md",
        title="长文",
        elements=[ParsedElement(type="paragraph", text=text)],
        full_text=text,
    )
    chunks = SemanticChunker(chunk_size=60, chunk_overlap=12).split(document)
    assert len(chunks) > 2
    assert all(chunk.token_count <= 60 for chunk in chunks)
    assert chunks[0].content[-8:] in chunks[1].content
