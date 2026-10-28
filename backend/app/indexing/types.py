from dataclasses import dataclass, field


@dataclass
class IndexedChunk:
    chunk_id: str
    document_id: str
    kb_id: str
    content: str
    section: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    kb_id: str
    content: str
    section: str
    source: str
    metadata: dict = field(default_factory=dict)
    rrf_score: float = 0
    rerank_score: float | None = None
