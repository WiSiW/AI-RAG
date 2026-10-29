import logging
import threading
from pathlib import Path

import jieba
from whoosh import index
from whoosh.analysis import Token, Tokenizer
from whoosh.fields import ID, STORED, TEXT, Schema
from whoosh.qparser import MultifieldParser
from whoosh.query import Or, Term

from app.indexing.types import IndexedChunk, RetrievedChunk

logger = logging.getLogger(__name__)
_write_lock = threading.Lock()


class ChineseTokenizer(Tokenizer):
    def __call__(
        self,
        value,
        positions=False,
        chars=False,
        keeporiginal=False,
        removestops=True,
        start_pos=0,
        start_char=0,
        mode="",
        **kwargs,
    ):
        token = Token(positions, chars, removestops=removestops, mode=mode, **kwargs)
        for word in jieba.cut(value):
            word = word.strip()
            if not word:
                continue
            token.text = word.casefold()
            token.boost = 1.0
            token.pos = start_pos
            token.startchar = start_char
            token.endchar = start_char + len(word)
            yield token
            start_pos += 1
            start_char += len(word)


class BM25Store:
    def __init__(self, root_dir: str = "data/whoosh"):
        self.root_dir = Path(root_dir)

    def _index_dir(self, kb_id: str, index_version: str = "v1") -> Path:
        base = self.root_dir / kb_id.replace("-", "")
        return base / "v2" if index_version == "v2" else base

    def _schema(self, index_version: str = "v1") -> Schema:
        fields = {
            "chunk_id": ID(stored=True, unique=True),
            "doc_id": ID(stored=True),
            "source": STORED(),
            "section": TEXT(analyzer=ChineseTokenizer(), stored=True),
            "content": TEXT(analyzer=ChineseTokenizer(), stored=True),
        }
        if index_version == "v2":
            fields["keywords"] = TEXT(analyzer=ChineseTokenizer(), stored=True)
        return Schema(**fields)

    @staticmethod
    def _document(chunk: IndexedChunk, include_keywords: bool) -> dict:
        document = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.document_id,
            "source": chunk.metadata.get("source", ""),
            "section": chunk.section,
            "content": chunk.content,
        }
        if include_keywords:
            document["keywords"] = " ".join(chunk.keywords)
        return document

    def _open(self, kb_id: str, index_version: str = "v1"):
        index_dir = self._index_dir(kb_id, index_version)
        index_dir.mkdir(parents=True, exist_ok=True)
        if index.exists_in(str(index_dir)):
            return index.open_dir(str(index_dir))
        return index.create_in(str(index_dir), self._schema(index_version))

    def add_chunks(
        self, kb_id: str, chunks: list[IndexedChunk], *, index_version: str = "v1"
    ) -> None:
        if not chunks:
            return
        with _write_lock:
            ix = self._open(kb_id, index_version)
            include_keywords = "keywords" in ix.schema
            writer = ix.writer()
            for chunk in chunks:
                writer.add_document(**self._document(chunk, include_keywords))
            writer.commit()

    def replace_document(
        self,
        kb_id: str,
        document_id: str,
        chunks: list[IndexedChunk],
        *,
        index_version: str = "v1",
    ) -> None:
        with _write_lock:
            ix = self._open(kb_id, index_version)
            include_keywords = "keywords" in ix.schema
            writer = ix.writer()
            writer.delete_by_term("doc_id", document_id)
            for chunk in chunks:
                writer.add_document(**self._document(chunk, include_keywords))
            writer.commit()

    def search(
        self,
        kb_id: str,
        query: str,
        top_k: int,
        document_id: str | None = None,
        *,
        index_version: str = "v1",
    ) -> list[RetrievedChunk]:
        ix = self._open(kb_id, index_version)
        results: list[RetrievedChunk] = []
        search_fields = ["content"]
        field_boosts = {"content": 1.0}
        if "section" in ix.schema:
            search_fields.append("section")
            field_boosts["section"] = 0.8
        if "keywords" in ix.schema:
            search_fields.append("keywords")
            field_boosts["keywords"] = 1.5
        with ix.searcher() as searcher:
            parser = MultifieldParser(
                search_fields,
                schema=ix.schema,
                fieldboosts=field_boosts,
            )
            try:
                parsed_query = parser.parse(query)
            except Exception:
                parsed_query = self._token_query(query, search_fields)
            result_filter = Term("doc_id", document_id) if document_id else None
            hits = searcher.search(parsed_query, limit=top_k, filter=result_filter)
            for hit in hits:
                results.append(
                    RetrievedChunk(
                        chunk_id=hit["chunk_id"],
                        document_id=hit["doc_id"],
                        kb_id=kb_id,
                        content=hit["content"],
                        section=hit["section"],
                        source=hit["source"],
                        metadata={
                            "bm25_score": float(hit.score),
                            "matched_representations": ["bm25"],
                            "representation_scores": {"bm25": float(hit.score)},
                        },
                    )
                )
        return results

    @staticmethod
    def _token_query(query: str, fields: list[str]) -> Or:
        clauses = [
            Term(field, token.text)
            for token in ChineseTokenizer()(query)
            if token.text.strip()
            for field in fields
        ]
        return Or(clauses)

    def delete_document(
        self, kb_id: str, document_id: str, *, index_version: str = "v1"
    ) -> None:
        with _write_lock:
            ix = self._open(kb_id, index_version)
            writer = ix.writer()
            writer.delete_by_term("doc_id", document_id)
            writer.commit()

    def delete_knowledge_base(self, kb_id: str, *, index_version: str = "v1") -> None:
        index_dir = self._index_dir(kb_id, index_version)
        if not index_dir.exists():
            return
        for child in index_dir.iterdir():
            if child.is_file():
                child.unlink()

    def delete_all_versions(self, kb_id: str) -> None:
        self.delete_knowledge_base(kb_id, index_version="v1")
        self.delete_knowledge_base(kb_id, index_version="v2")

    def health(self) -> tuple[bool, str]:
        try:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            return True, str(self.root_dir.resolve())
        except Exception as exc:
            return False, str(exc)

