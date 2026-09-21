"""The ingestion pipeline: PDFs in, a searchable index out.

One function does the whole thing, and it is deliberately dull. The interesting
decisions were made in `chunking.py`; this module only sequences them and talks
to the store.

In Lesson 2 the same function writes documents and chunks to Postgres first and
indexes Qdrant from those rows. In Lesson 3 it stops being a batch job over a
directory and becomes something an upload triggers. The shape survives both.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.chunking import Chunk, chunk_document
from app.config import ROOT, settings
from app.index import recreate, upsert
from app.logs import get_logger
from app.models import embed_documents, embedding_dim

log = get_logger(__name__)


def load_manifest() -> dict[str, str]:
    """Which PDFs make up the corpus, and what each one is called."""
    return json.loads((ROOT / "data" / "corpus.json").read_text())["documents"]


def chunk_corpus(docs: dict[str, str] | None = None) -> list[Chunk]:
    docs = docs or load_manifest()
    out: list[Chunk] = []
    for doc, title in docs.items():
        path = settings.corpus_path / f"{doc}.pdf"
        if not path.exists():
            log.warning("missing %s, skipping", path)
            continue
        out.extend(chunk_document(path, doc=doc, title=title))
    return out


def ingest(docs: dict[str, str] | None = None) -> dict:
    """Parse, chunk, embed, index. Returns what happened, for the caller to print."""
    t0 = time.perf_counter()
    chunks = chunk_corpus(docs)
    if not chunks:
        raise RuntimeError(f"no PDFs found in {settings.corpus_path}")

    t_parse = time.perf_counter() - t0
    vectors = embed_documents([c.embed_text for c in chunks], progress=True)
    t_embed = time.perf_counter() - t0 - t_parse

    recreate(embedding_dim())
    upsert(chunks, vectors)

    stats = {
        "documents": len({c.doc for c in chunks}),
        "chunks": len(chunks),
        "dim": embedding_dim(),
        "parse_seconds": round(t_parse, 1),
        "embed_seconds": round(t_embed, 1),
        "total_seconds": round(time.perf_counter() - t0, 1),
        "usd": 0.0,        # the embedder runs here. This zero is the point of `models.py`.
    }
    log.info("ingested %(chunks)d chunks from %(documents)d documents in "
             "%(total_seconds).1fs", stats)
    return stats
