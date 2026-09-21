"""The GenAI Lesson 15 retriever: dense plus BM25, fused, then reranked.

Nothing here is new. What is new is that it is a class with a lifetime. In a
notebook the BM25 index was a global built by a cell; here it is state that has
to be built once and reused, because rebuilding it per question would put a
corpus scan in the latency of every request.

`get_retriever()` is the seam Lesson 3 needs: the API builds one at startup and
hands it to every request handler.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.index import all_chunks, search
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
    """Built once, queried many times."""

    def __init__(self) -> None:
        t0 = time.perf_counter()
        self.chunks = all_chunks()
        if not self.chunks:
            raise RuntimeError("the index is empty; run `python -m app.ingest` first")
        self.texts = [f"{c['title']}, page {c['page']}\n{c['text']}" for c in self.chunks]
        self.by_id = {c["id"]: i for i, c in enumerate(self.chunks)}
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        log.info("retriever ready: %d chunks, BM25 built in %.1fs",
                 len(self.chunks), time.perf_counter() - t0)

    # ---- the three stages, each one a method you can call on its own -----
    def dense(self, query: str, k: int) -> dict[int, int]:
        """Chunk position -> rank, from the vector index."""
        hits = search(embed_query(query), limit=k)
        return {self.by_id[p["id"]]: r for r, (p, _) in enumerate(hits, 1)
                if p["id"] in self.by_id}

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
