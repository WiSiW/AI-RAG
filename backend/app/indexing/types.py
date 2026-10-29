from dataclasses import dataclass, field


@dataclass
class IndexedChunk:
    chunk_id: str
    document_id: str
    kb_id: str
    content: str
    section: str
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    hypothetical_questions: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class IndexedRepresentation:
    chunk_id: str
    document_id: str
    kb_id: str
    section: str
    representation_type: str
    indexed_text: str
    question_index: int = 0
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
    context_text: str = ""
