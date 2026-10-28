from app.generation.citation_parser import parse_citations
from app.indexing.types import RetrievedChunk


def make_chunk(chunk_id: str, index: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{index}",
        kb_id="kb",
        content=f"这是第 {index} 段参考资料的内容。",
        section=f"第 {index} 节",
        source=f"source-{index}.md",
        rrf_score=0.1,
    )


def test_parse_single_and_compound_citations():
    chunks = [make_chunk("c1", 1), make_chunk("c2", 2), make_chunk("c3", 3)]
    citations = parse_citations("结论甲 [1]，结论乙 [2,3]。", chunks)
    assert [citation.index for citation in citations] == [1, 2, 3]


def test_parse_citation_range_and_filter_invalid():
    chunks = [make_chunk("c1", 1), make_chunk("c2", 2)]
    citations = parse_citations("范围 [1-5]，不存在的编号 [9]。", chunks)
    assert [citation.index for citation in citations] == [1, 2]
