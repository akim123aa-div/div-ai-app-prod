"""The vector index, and the only module that knows Qdrant exists.

Everything above this file asks for "the chunks nearest this vector" and gets
them. Swapping Qdrant for pgvector, Weaviate or a numpy array is a rewrite of
this module and of nothing else. That is the reason it is a module.

In Lesson 1 the payload carried the whole chunk, which made Qdrant both the
index and the store. From Lesson 2 it carries two fields: the chunk ID, which is
the key back to the `chunks` table in Postgres, and the document slug, which the
search filter needs. The text lives in one place. Qdrant holds only what can be
recomputed from it, so dropping the collection loses nothing that
`python -m app.reindex` cannot rebuild.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, FilterSelector,
                                  MatchValue, PointStruct, VectorParams)

from app.config import settings
from app.logs import get_logger

log = get_logger(__name__)

_client: QdrantClient | None = None


def get_index() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url, timeout=60)
    return _client


def point_id(chunk_id: str) -> str:
    """Qdrant wants a UUID or an integer, and our IDs are strings like
    `albany#p7#3`. A deterministic UUID keeps re-ingestion idempotent: the same
    chunk overwrites itself instead of arriving twice."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _doc_filter(doc: str) -> Filter:
    return Filter(must=[FieldCondition(key="doc", match=MatchValue(value=doc))])


def exists() -> bool:
    return get_index().collection_exists(settings.qdrant_collection)


def drop() -> None:
    get_index().delete_collection(settings.qdrant_collection)
    log.info("collection %r dropped", settings.qdrant_collection)


def create(dim: int) -> None:
    """An empty collection, if there is none."""
    if not exists():
        get_index().create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        log.info("collection %r created at %d dimensions", settings.qdrant_collection, dim)


def delete_document(doc: str) -> None:
    """Remove one document's points, so re-ingesting it cannot leave strays."""
    if exists():
        get_index().delete(collection_name=settings.qdrant_collection,
                           points_selector=FilterSelector(filter=_doc_filter(doc)))


def upsert(chunks: list[dict], vectors, batch: int = 256) -> int:
    """Index chunk rows. The payload is the link back to Postgres, not a copy."""
    client = get_index()
    for i in range(0, len(chunks), batch):
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=[PointStruct(id=point_id(c["id"]), vector=v.tolist(),
                                payload={"chunk_id": c["id"], "doc": c["doc"]})
                    for c, v in zip(chunks[i:i + batch], vectors[i:i + batch])],
        )
    return len(chunks)


def search(vector, limit: int, doc: str | None = None) -> list[tuple[str, float]]:
    """Chunk IDs nearest the vector, optionally restricted to one document.

    IDs and scores only. The caller looks the text up in the rows it loaded from
    Postgres. The filter goes into the query rather than being applied to the
    results, which is the Lesson 16 point: filter before you score, not after.
    """
    hits = get_index().query_points(
        collection_name=settings.qdrant_collection, query=vector.tolist(),
        limit=limit, query_filter=_doc_filter(doc) if doc else None,
        with_payload=True).points
    return [(h.payload["chunk_id"], float(h.score)) for h in hits]


def count() -> int:
    """Points in the collection, and zero rather than an error if there is none."""
    if not exists():
        return 0
    return int(get_index().count(settings.qdrant_collection).count)
