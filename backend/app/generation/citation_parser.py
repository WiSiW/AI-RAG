import re

from app.indexing.types import RetrievedChunk
from app.schemas import Citation

CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*[,，\-]\s*\d+)*)\]")


def parse_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    indices: set[int] = set()
    for match in CITATION_PATTERN.finditer(answer):
        raw = match.group(1).replace("，", ",")
        for part in raw.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    start, end = [int(value.strip()) for value in part.split("-", 1)]
                except ValueError:
                    continue
                if start <= end and end - start <= 50:
                    indices.update(range(start, end + 1))
            else:
                try:
                    indices.add(int(part))
                except ValueError:
                    continue

    citations: list[Citation] = []
    for index in sorted(indices):
        if not 1 <= index <= len(chunks):
            continue
        chunk = chunks[index - 1]
        snippet = chunk.content[:240] + ("..." if len(chunk.content) > 240 else "")
        citations.append(
            Citation(
                index=index,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                section=chunk.section,
                score=chunk.rerank_score or chunk.rrf_score,
                snippet=snippet,
            )
        )
    return citations
