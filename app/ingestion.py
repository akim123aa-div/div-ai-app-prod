"""The ingestion pipeline: PDFs in, rows in Postgres, an index derived from them.

Two stages, in this order:

1. parse and chunk each PDF, and write the document and its chunks to Postgres
2. read those rows back and embed them into Qdrant

Stage 2 reads from the table, not from the chunks stage 1 still has in memory.
That is deliberate. It makes `index_documents` the same function whether it runs
straight after a parse or a month later against a dropped collection, and it
means the index can only contain what the database says exists.

`rebuild_index` is stage 2 alone, over every row. It is what
`python -m app.reindex` runs, and it never opens a PDF.

From Lesson 3 there are two ways in. `ingest` is the batch job over the corpus
directory, run from a terminal. `ingest_upload` is one document, run by the API
in the background after an upload, and it is the job pattern: the row starts
`pending`, the two stages run, the API's retriever is rebuilt, and only then
does the row say `ready`. Otherwise it says `failed`, with a reason. The two stages are the same functions either way.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from app import db, index
from app.chunking import Chunk, chunk_document, page_count
from app.config import ROOT, settings
from app.logs import get_logger
from app.models import embed_documents, embedding_dim

log = get_logger(__name__)


def load_manifest() -> dict[str, str]:
    """Which PDFs make up the corpus, and what each one is called."""
    return json.loads((ROOT / "data" / "corpus.json").read_text())["documents"]


def slugify(name: str) -> str:
    """A document ID from a filename: `Annual Report 2023.pdf` -> `annual-report-2023`.
    The same file uploaded twice gets the same ID, and so replaces itself."""
    return re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-")[:60] or "document"


def store_document(doc: str, title: str, path: Path | None = None,
                   status: str = "ready") -> int:
    """Stage 1 for one PDF: parse, chunk, write to Postgres. Returns the chunk count."""
    path = path or settings.corpus_path / f"{doc}.pdf"
    chunks: list[Chunk] = chunk_document(path, doc=doc, title=title)
    if not chunks:
        raise ValueError("no text could be extracted; is it a scan?")
    return db.replace_document(doc, title=title, filename=path.name,
                               pages=page_count(path), chunks=chunks, status=status)


def index_documents(docs: list[str] | None = None, progress: bool = False) -> int:
    """Stage 2: embed chunk rows from Postgres into Qdrant. All of them if `docs`
    is None, and then the collection is dropped first so nothing stale survives."""
    rows = db.load_chunks(docs, ready_only=docs is None)
    if docs is None:
        index.drop()
    else:
        for d in docs:
            index.delete_document(d)
    index.create(embedding_dim())
    vectors = embed_documents([Chunk(**r).embed_text for r in rows], progress=progress)
    return index.upsert(rows, vectors)


def ingest(docs: dict[str, str] | None = None) -> dict:
    """Parse, store, embed, index. Returns what happened, for the caller to print.

    With no `docs`, the whole corpus, and the collection is rebuilt from scratch.
    With some, only those documents are replaced, in both stores.
    """
    t0 = time.perf_counter()
    full = not docs
    docs = docs or load_manifest()
    db.init_db()

    stored = {}
    for doc, title in docs.items():
        if not (settings.corpus_path / f"{doc}.pdf").exists():
            log.warning("missing %s.pdf in %s, skipping", doc, settings.corpus_path)
            continue
        stored[doc] = store_document(doc, title)
    if not stored:
        raise RuntimeError(f"no PDFs found in {settings.corpus_path}")
    t_parse = time.perf_counter() - t0

    n = index_documents(None if full else list(stored), progress=True)
    stats = {
        "documents": len(stored),
        "chunks": sum(stored.values()),
        "indexed": n,
        "dim": embedding_dim(),
        "parse_seconds": round(t_parse, 1),
        "embed_seconds": round(time.perf_counter() - t0 - t_parse, 1),
        "total_seconds": round(time.perf_counter() - t0, 1),
        "usd": 0.0,        # the embedder runs here. This zero is the point of `models.py`.
    }
    log.info("ingested %(chunks)d chunks from %(documents)d documents in "
             "%(total_seconds).1fs", stats)
    return stats


def ingest_upload(doc: str, title: str, path: Path) -> None:
    """One uploaded PDF, run by the API as a background task after the response.

    Nobody is waiting on this function's return value, so it has none, and it
    must not raise: an exception here goes to a log that the client never sees.
    Everything the client needs to know ends up in the document's row, which is
    what it is polling.
    """
    from app.retrieval import refresh_retriever

    t0 = time.perf_counter()
    log.info("upload %s: ingesting in the background", doc)
    try:
        n = store_document(doc, title, path=path, status="pending")
        log.info("upload %s: %d chunks in Postgres after %.1fs; embedding them",
                 doc, n, time.perf_counter() - t0)
        index_documents([doc])
        log.info("upload %s: %d points in Qdrant after %.1fs; rebuilding BM25",
                 doc, n, time.perf_counter() - t0)
        # The retriever holds every chunk in memory for BM25, built at startup. A
        # new document is invisible to it until it is rebuilt from the table, so
        # the rebuild comes before `ready`: ready means searchable.
        refresh_retriever(include=[doc])
        db.set_status(doc, "ready")
    except Exception as e:                    # any failure becomes a status, not a crash
        log.exception("ingesting %s failed", doc)
        index.delete_document(doc)
        db.set_status(doc, "failed", error=f"{type(e).__name__}: {e}"[:500])
        return
    log.info("upload %s: ready, %d chunks in %.1fs", doc, n, time.perf_counter() - t0)


def rebuild_index() -> dict:
    """Throw the vector index away and derive it again from Postgres."""
    t0 = time.perf_counter()
    n = index_documents(None, progress=True)
    log.info("rebuilt index: %d chunks in %.1fs", n, time.perf_counter() - t0)
    return {"chunks_in_postgres": db.count_chunks(), "points_in_qdrant": index.count(),
            "seconds": round(time.perf_counter() - t0, 1), "pdfs_opened": 0}
