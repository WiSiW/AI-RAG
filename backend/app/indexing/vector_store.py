import logging

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app.core.config import settings
from app.indexing.types import IndexedChunk, RetrievedChunk

logger = logging.getLogger(__name__)


def collection_name(kb_id: str) -> str:
    return f"kb_{kb_id.replace('-', '')}"


def _ensure_connection() -> None:
    if not connections.has_connection("default"):
        connections.connect(alias="default", uri=settings.milvus_uri)


class MilvusVectorStore:
    def ensure_collection(self, kb_id: str, dimensions: int) -> Collection:
        _ensure_connection()
        name = collection_name(kb_id)
        if utility.has_collection(name):
            collection = Collection(name)
            vector_field = next(field for field in collection.schema.fields if field.name == "embedding")
            params = getattr(vector_field, "params", {}) or {}
            current_dim = int(params.get("dim", 0))
            if current_dim and current_dim != dimensions:
                raise ValueError(
                    f"知识库向量集合维度为 {current_dim}，当前模型维度为 {dimensions}"
                )
            collection.load()
            return collection

        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimensions),
        ]
        schema = CollectionSchema(fields, description=f"Knowledge base {kb_id}")
        collection = Collection(name, schema)
        collection.create_index(
            "embedding",
            {
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 256},
            },
        )
        collection.load()
        return collection

    def replace_document(
        self,
        kb_id: str,
        document_id: str,
        source: str,
        chunks: list[IndexedChunk],
        vectors: list[list[float]],
        dimensions: int,
    ) -> None:
        collection = self.ensure_collection(kb_id, dimensions)
        escaped_doc_id = document_id.replace('"', '\\"')
        collection.delete(expr=f'doc_id == "{escaped_doc_id}"')
        if not chunks:
            collection.flush()
            return
        entities = [
            [chunk.chunk_id for chunk in chunks],
            [document_id for _ in chunks],
            [source for _ in chunks],
            [chunk.section[:2048] for chunk in chunks],
            [chunk.content[:32768] for chunk in chunks],
            vectors,
        ]
        collection.insert(entities)
        collection.flush()

    def search(
        self,
        kb_id: str,
        query_vector: list[float],
        top_k: int,
        dimensions: int,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        collection = self.ensure_collection(kb_id, dimensions)
        expr = None
        if document_id:
            escaped_doc_id = document_id.replace('"', '\\"')
            expr = f'doc_id == "{escaped_doc_id}"'
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": 128}},
            limit=top_k,
            output_fields=["chunk_id", "doc_id", "source", "section", "text"],
            expr=expr,
        )
        retrieved: list[RetrievedChunk] = []
        for hit in results[0]:
            entity = hit.entity
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(entity.get("chunk_id")),
                    document_id=str(entity.get("doc_id")),
                    kb_id=kb_id,
                    content=str(entity.get("text")),
                    section=str(entity.get("section")),
                    source=str(entity.get("source")),
                    metadata={"vector_score": float(hit.score), "vector_rank": len(retrieved) + 1},
                )
            )
        return retrieved

    def delete_document(self, kb_id: str, document_id: str) -> None:
        _ensure_connection()
        name = collection_name(kb_id)
        if not utility.has_collection(name):
            return
        collection = Collection(name)
        escaped_doc_id = document_id.replace('"', '\\"')
        collection.delete(expr=f'doc_id == "{escaped_doc_id}"')
        collection.flush()

    def drop_knowledge_base(self, kb_id: str) -> None:
        _ensure_connection()
        name = collection_name(kb_id)
        if utility.has_collection(name):
            utility.drop_collection(name)

    def health(self) -> tuple[bool, str]:
        try:
            _ensure_connection()
            return True, "connected"
        except Exception as exc:
            return False, str(exc)

