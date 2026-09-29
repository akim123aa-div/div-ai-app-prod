"""The GenAI Lesson 15 retriever: dense plus BM25, fused, then reranked.

Nothing here is new. What is new is that it is a class with a lifetime. In a
notebook the BM25 index was a global built by a cell; here it is state that has
to be built once and reused, because rebuilding it per question would put a
corpus scan in the latency of every request.

From Lesson 2 the chunks come from Postgres, not from Qdrant. BM25 is rebuilt
from the `chunks` table at startup, and the dense side gets chunk IDs back from
the vector index and looks them up here. The text lives in one store.

`get_retriever()` is the seam the API needs: it builds one at startup and every
request handler uses it. `refresh_retriever()` builds a new one from the table
and swaps it in, which the API does after an upload finishes. A request already
running keeps the retriever it started with; the next one gets the new one.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.db import load_chunks
from app.index import count as index_count, search
from app.logs import get_logger
from app.models import embed_query, rerank_scores

log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


@dataclass
class Hit:
    """One retrieved chunk and the scores that got it here."""

    chunk: dict
    score: float           # reranker relevance, 0 to 1
    fused: float           # reciprocal rank fusion score, before reranking

    @property
    def source(self) -> str:
        return f"{self.chunk['title']}, page {self.chunk['page']}"


class Retriever:
    """Built once, queried many times.

    `include` adds documents that are not `ready` yet. The upload job uses it to
    build a retriever that can already see the new document, and only then marks
    the document ready, so that a client who sees `ready` can search it at once.
    """

    def __init__(self, include: list[str] | None = None) -> None:
        t0 = time.perf_counter()
        self.chunks = load_chunks()
        if include:
            self.chunks += load_chunks(include, ready_only=False)
        if not self.chunks:
            # An empty corpus is a real state now: the API starts before the first
            # upload. Every question is refused by the gate until one arrives.
            log.warning("no chunks in Postgres; ingest or upload a document")
        elif (n := index_count()) != len(self.chunks):
            log.warning("Qdrant has %d points and Postgres %d chunks; "
                        "run `python -m app.reindex`", n, len(self.chunks))
        self.texts = [f"{c['title']}, page {c['page']}\n{c['text']}" for c in self.chunks]
        self.by_id = {c["id"]: i for i, c in enumerate(self.chunks)}
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts]) if self.texts else None
        log.info("retriever ready: %d chunks, BM25 built in %.1fs",
                 len(self.chunks), time.perf_counter() - t0)

    # ---- the three stages, each one a method you can call on its own -----
    def dense(self, query: str, k: int) -> dict[int, int]:
        """Chunk position -> rank, from the vector index. Qdrant returns IDs; the
        text they point at is the row loaded from Postgres."""
        hits = search(embed_query(query), limit=k)
        return {self.by_id[cid]: r for r, (cid, _) in enumerate(hits, 1)
                if cid in self.by_id}

    def lexical(self, query: str, k: int) -> dict[int, int]:
        """Chunk position -> rank, from BM25 over the same text."""
        scores = self.bm25.get_scores(tokenize(query))
        return {int(i): r for r, i in enumerate(np.argsort(-scores)[:k], 1)}

    def fuse(self, *rankings: dict[int, int], k: int = 60) -> dict[int, float]:
        """Reciprocal rank fusion. Ranks combine; raw scores from two different
        scales do not, which is the whole reason RRF exists."""
        fused: dict[int, float] = {}
        for ranking in rankings:
            for i, rank in ranking.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank)
        return fused

    def retrieve(self, query: str, top_k: int | None = None,
                 shortlist: int | None = None) -> list[Hit]:
        """The whole pipeline: fuse two rankings, rerank the shortlist, cut to k."""
        top_k = top_k or settings.top_k
        shortlist = shortlist or settings.shortlist
        if not self.chunks:
            return []

        fused = self.fuse(self.dense(query, shortlist), self.lexical(query, shortlist))
        candidates = sorted(fused, key=lambda i: -fused[i])[:shortlist]
        scores = rerank_scores(query, [self.texts[i][:2400] for i in candidates])
        order = np.argsort(-scores)

        return [Hit(chunk=self.chunks[candidates[j]], score=float(scores[j]),
                    fused=float(fused[candidates[j]])) for j in order[:top_k]]


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def refresh_retriever(include: list[str] | None = None) -> Retriever:
    """Rebuild from Postgres and swap. The swap is one assignment, so no request
    ever sees a half-built retriever."""
    global _retriever
    _retriever = Retriever(include)
    return _retriever
