"""The vector index, and the only module that knows Qdrant exists.

Everything above this file asks for "the chunks nearest this vector" and gets
them. Swapping Qdrant for pgvector, Weaviate or a numpy array is a rewrite of
this module and of nothing else. That is the reason it is a module.

One thing here is a deliberate placeholder. In Lesson 1 the payload carries the
chunk text, which makes Qdrant both the index and the store. It works, and it is
wrong: an index is derived data, and derived data should be rebuildable from
something else. Lesson 2 puts the text in Postgres and leaves the vectors here.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.chunking import Chunk
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


def recreate(dim: int) -> None:
    """Drop the collection and make an empty one. Ingestion is not incremental
    in Lesson 1; Lesson 3 makes it per-document."""
    client = get_index()
    client.delete_collection(settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    log.info("collection %r recreated at %d dimensions", settings.qdrant_collection, dim)


def upsert(chunks: list[Chunk], vectors, batch: int = 256) -> int:
    client = get_index()
    for i in range(0, len(chunks), batch):
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=[PointStruct(id=point_id(c.id), vector=v.tolist(), payload=c.to_dict())
                    for c, v in zip(chunks[i:i + batch], vectors[i:i + batch])],
        )
    return len(chunks)


def search(vector, limit: int, doc: str | None = None) -> list[tuple[dict, float]]:
    """Nearest chunks, optionally restricted to one document.

    The filter goes into the query rather than being applied to the results,
    which is the Lesson 16 point: filter before you score, not after.
    """
    flt = None
    if doc:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        flt = Filter(must=[FieldCondition(key="doc", match=MatchValue(value=doc))])
    hits = get_index().query_points(
        collection_name=settings.qdrant_collection, query=vector.tolist(),
        limit=limit, query_filter=flt, with_payload=True).points
    return [(h.payload, float(h.score)) for h in hits]


def all_chunks() -> list[dict]:
    """Every chunk payload, for the BM25 side of the hybrid retriever.

    BM25 needs the whole corpus in memory, so something has to hold it. In
    Lesson 1 that something is Qdrant, scrolled at startup. The production
    answer is a search engine that owns its own index; the Lesson 2 answer is
    Postgres. Either way this function is the seam.
    """
    out, offset = [], None
    while True:
        points, offset = get_index().scroll(
            collection_name=settings.qdrant_collection, limit=1024,
            offset=offset, with_payload=True, with_vectors=False)
        out.extend(p.payload for p in points)
        if offset is None:
            return out


def count() -> int:
    return int(get_index().count(settings.qdrant_collection).count)
