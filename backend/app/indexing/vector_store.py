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
from app.indexing.types import IndexedChunk, IndexedRepresentation, RetrievedChunk

logger = logging.getLogger(__name__)


def collection_name(kb_id: str, index_version: str = "v2") -> str:
    base = f"kb_{kb_id.replace('-', '')}"
    return f"{base}_v2" if index_version == "v2" else base


def _ensure_connection() -> None:
    if not connections.has_connection("default"):
        connections.connect(alias="default", uri=settings.milvus_uri)


def _vector_dimension(collection: Collection) -> int:
    vector_field = next(field for field in collection.schema.fields if field.name == "embedding")
    params = getattr(vector_field, "params", {}) or {}
    return int(params.get("dim", 0))


class MilvusVectorStore:
    def ensure_collection(
        self, kb_id: str, dimensions: int, index_version: str = "v2"
    ) -> Collection:
        _ensure_connection()
        name = collection_name(kb_id, index_version)
        if utility.has_collection(name):
            collection = Collection(name)
            current_dim = _vector_dimension(collection)
            if current_dim and current_dim != dimensions:
                raise ValueError(
                    f"知识库向量集合维度为 {current_dim}，当前模型维度为 {dimensions}"
                )
            collection.load()
            return collection

        common_fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=2048),
        ]
        if index_version == "v2":
            fields = common_fields + [
                FieldSchema(name="representation_type", dtype=DataType.VARCHAR, max_length=16),
                FieldSchema(name="question_index", dtype=DataType.INT8),
                FieldSchema(name="indexed_text", dtype=DataType.VARCHAR, max_length=32768),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimensions),
            ]
        else:
            fields = common_fields + [
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimensions),
            ]
        schema = CollectionSchema(fields, description=f"Knowledge base {kb_id} {index_version}")
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
        representations: list[IndexedRepresentation] | None,
        vectors: list[list[float]],
        dimensions: int,
        *,
        index_version: str = "v2",
        chunks: list[IndexedChunk] | None = None,
    ) -> None:
        collection = self.ensure_collection(kb_id, dimensions, index_version)
        escaped_doc_id = document_id.replace('"', '\\"')
        collection.delete(expr=f'doc_id == "{escaped_doc_id}"')
        if index_version == "v2":
            representations = representations or []
            if not representations:
                collection.flush()
                return
            if len(representations) != len(vectors):
                raise ValueError("表示数量与向量数量不一致")
            entities = [
                [item.chunk_id for item in representations],
                [document_id for _ in representations],
                [source for _ in representations],
                [item.section[:2048] for item in representations],
                [item.representation_type for item in representations],
                [item.question_index for item in representations],
                [item.indexed_text[:32768] for item in representations],
                vectors,
            ]
        else:
            chunks = chunks or []
            if not chunks:
                collection.flush()
                return
            if len(chunks) != len(vectors):
                raise ValueError("chunk 数量与向量数量不一致")
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
        *,
        representation_type: str | None = None,
        index_version: str = "v2",
    ) -> list[RetrievedChunk]:
        collection = self.ensure_collection(kb_id, dimensions, index_version)
        conditions: list[str] = []
        if document_id:
            escaped_doc_id = document_id.replace('"', '\\"')
            conditions.append(f'doc_id == "{escaped_doc_id}"')
        if representation_type and index_version == "v2":
            escaped_type = representation_type.replace('"', '\\"')
            conditions.append(f'representation_type == "{escaped_type}"')
        expr = " and ".join(conditions) or None
        output_fields = ["chunk_id", "doc_id", "source", "section"]
        output_fields.append("indexed_text" if index_version == "v2" else "text")
        if index_version == "v2":
            output_fields.extend(["representation_type", "question_index"])
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": 128}},
            limit=top_k,
            output_fields=output_fields,
            expr=expr,
        )
        retrieved: list[RetrievedChunk] = []
        for hit in results[0]:
            entity = hit.entity
            hit_type = str(entity.get("representation_type", "original"))
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(entity.get("chunk_id")),
                    document_id=str(entity.get("doc_id")),
                    kb_id=kb_id,
                    content=str(entity.get("indexed_text" if index_version == "v2" else "text")),
                    section=str(entity.get("section")),
                    source=str(entity.get("source")),
                    metadata={
                        "vector_score": float(hit.score),
                        "vector_rank": len(retrieved) + 1,
                        "representation_type": hit_type,
                        "question_index": int(entity.get("question_index", 0)),
                        **(
                            {
                                "matched_question": str(
                                    entity.get(
                                        "indexed_text" if index_version == "v2" else "text"
                                    )
                                )
                            }
                            if hit_type == "question"
                            else {}
                        ),
                    },
                )
            )
        return retrieved

    def export_representations(
        self, kb_id: str, dimensions: int, *, index_version: str = "v2"
    ) -> list[dict]:
        _ensure_connection()
        name = collection_name(kb_id, index_version)
        if not utility.has_collection(name):
            raise ValueError("知识库向量集合不存在")
        collection = Collection(name)
        current_dim = _vector_dimension(collection)
        if current_dim and current_dim != dimensions:
            raise ValueError("知识库向量维度与导出配置不一致")
        collection.load()
        collection.flush()
        output_fields = [
            "chunk_id",
            "doc_id",
            "source",
            "section",
            "representation_type",
            "question_index",
            "indexed_text",
            "embedding",
        ]
        rows: list[dict] = []
        iterator = collection.query_iterator(
            batch_size=1000,
            expr="pk >= 0",
            output_fields=output_fields,
        )
        try:
            while page := iterator.next():
                for row in page:
                    rows.append(
                        {
                            "chunk_id": str(row.get("chunk_id", "")),
                            "doc_id": str(row.get("doc_id", "")),
                            "source": str(row.get("source", "")),
                            "section": str(row.get("section", "")),
                            "representation_type": str(
                                row.get("representation_type", "original")
                            ),
                            "question_index": int(row.get("question_index", 0) or 0),
                            "indexed_text": str(row.get("indexed_text", "")),
                            "embedding": [
                                float(value) for value in row.get("embedding", [])
                            ],
                        }
                    )
        finally:
            iterator.close()
        return rows

    def delete_document(
        self, kb_id: str, document_id: str, *, index_version: str = "v2"
    ) -> None:
        _ensure_connection()
        name = collection_name(kb_id, index_version)
        if not utility.has_collection(name):
            return
        collection = Collection(name)
        escaped_doc_id = document_id.replace('"', '\\"')
        collection.delete(expr=f'doc_id == "{escaped_doc_id}"')
        collection.flush()

    def drop_knowledge_base(self, kb_id: str, *, index_version: str | None = None) -> None:
        _ensure_connection()
        versions = [index_version] if index_version else ["v1", "v2"]
        for version in versions:
            name = collection_name(kb_id, version)
            if utility.has_collection(name):
                utility.drop_collection(name)

    def health(self) -> tuple[bool, str]:
        try:
            _ensure_connection()
            return True, "connected"
        except Exception as exc:
            return False, str(exc)
