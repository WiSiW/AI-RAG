from pathlib import Path
from tempfile import TemporaryDirectory

from whoosh import index
from whoosh.fields import ID, STORED, TEXT, Schema

from app.indexing.bm25_store import BM25Store, ChineseTokenizer
from app.indexing.types import IndexedChunk, RetrievedChunk
from app.retrieval.hybrid_retriever import rrf_fuse
from app.services.chat_service import resolve_search_relevance


def chunk(chunk_id: str, score: float = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc",
        kb_id="kb",
        content=f"content-{chunk_id}",
        section="正文",
        source="demo.md",
        metadata={"score": score},
    )


def test_rrf_merges_duplicate_chunks():
    bm25 = [chunk("a"), chunk("b")]
    vector = [chunk("b"), chunk("c")]
    result = rrf_fuse(bm25, vector, rrf_k=60, top_k=3)
    assert [item.chunk_id for item in result] == ["b", "a", "c"]
    assert result[0].rrf_score > result[1].rrf_score


def test_rrf_respects_top_k():
    result = rrf_fuse([chunk("a"), chunk("b"), chunk("c")], [], top_k=2)
    assert len(result) == 2


def test_search_relevance_prefers_rerank_score():
    assert resolve_search_relevance(0.72, 0.031) == (0.72, "rerank")


def test_search_relevance_falls_back_to_rrf_score():
    assert resolve_search_relevance(None, 0.031) == (0.031, "rrf")


def test_bm25_multiword_chinese_query_returns_hits():
    with TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        store = BM25Store(temp_dir)
        store.add_chunks(
        "kb-test",
        [
            IndexedChunk(
                chunk_id="c1",
                document_id="d1",
                kb_id="kb-test",
                content="糖尿病患者应该如何控制血糖？",
                section="治疗",
                metadata={"source": "demo.md"},
            ),
            IndexedChunk(
                chunk_id="c2",
                document_id="d1",
                kb_id="kb-test",
                content="高血压患者应减少盐分摄入。",
                section="饮食",
                metadata={"source": "demo.md"},
            ),
            ],
        )

        results = store.search("kb-test", "糖尿病 血糖", 10)

        assert [item.chunk_id for item in results] == ["c1"]
        assert results[0].metadata["bm25_score"] > 0


def test_bm25_search_supports_legacy_index_with_stored_section():
    with TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        index_dir = Path(temp_dir) / "kb"
        index_dir.mkdir()
        legacy_index = index.create_in(
            str(index_dir),
            Schema(
                chunk_id=ID(stored=True, unique=True),
                doc_id=ID(stored=True),
                source=STORED(),
                section=STORED(),
                content=TEXT(analyzer=ChineseTokenizer(), stored=True),
            ),
        )
        writer = legacy_index.writer()
        writer.add_document(
            chunk_id="c1",
            doc_id="d1",
            source="demo.md",
            section="治疗",
            content="糖尿病患者应该如何控制血糖？",
        )
        writer.commit()

        results = BM25Store(temp_dir).search("kb", "糖尿病 血糖", 10)

        assert [item.chunk_id for item in results] == ["c1"]
