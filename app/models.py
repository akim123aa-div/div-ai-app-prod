"""The two models that run inside this process, on the CPU.

GenAI Lessons 14 to 16 called an embedding API and used a chat model as the
reranker. Both were the right choice for a notebook, where the cost is a few
cents and the alternative is a download. In a service they are the wrong choice
twice over: every ingested page is a billed API call, and the reranker sits in
the latency of every question.

So both move in-process. A small bi-encoder embeds, a small cross-encoder
reranks, and neither needs a GPU or a network. The production version of this
line is two model servers with their own hardware, which is the row in the
module's simplification table.

Loading is lazy and cached, because the weights take a few seconds to arrive and
nothing should pay that twice. Lesson 3 warms both at startup so that no user's
first question does.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import settings
from app.logs import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    log.info("loading embedder %s", settings.embed_model)
    return SentenceTransformer(settings.embed_model, device="cpu")


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    log.info("loading reranker %s", settings.rerank_model)
    return CrossEncoder(settings.rerank_model, device="cpu", max_length=512)


def embed_documents(texts: list[str], batch_size: int = 64,
                    progress: bool = False) -> np.ndarray:
    """Vectors for corpus text. Normalized, so a dot product is a cosine.

    This is the slow step of ingestion and it is slow in an honest way: a few
    thousand passages through a small transformer on a CPU takes minutes. A GPU
    or a hosted embedding API makes it seconds, and both are the trade named in
    the module's simplification table.
    """
    return get_embedder().encode(
        list(texts), batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=progress, convert_to_numpy=True).astype("float32")


def embed_query(text: str) -> np.ndarray:
    """A query, embedded with the instruction prefix bge models are trained with.

    The prefix is a property of this model family, not a general technique. Using
    the wrong one costs a few points of recall silently, which is why the corpus
    side and the query side are two functions rather than one with a flag.
    """
    prefix = "Represent this sentence for searching relevant passages: "
    return get_embedder().encode(
        [prefix + text], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")[0]


def rerank_scores(query: str, passages: list[str]) -> np.ndarray:
    """Relevance in 0..1 for each passage, from a cross-encoder.

    The cross-encoder reads the query and the passage together, which is what a
    bi-encoder cannot do and why it ranks better. It is also why it cannot be
    precomputed: there is no passage vector to store.

    The raw output is a logit; a sigmoid turns it into something a threshold can
    be set on. That threshold is the refusal gate from GenAI Lesson 16, and its
    value changed when the reranker did. The mechanism survived the swap, the
    number did not.
    """
    if not passages:
        return np.zeros(0, dtype="float32")
    logits = np.asarray(get_reranker().predict(
        [(query, p) for p in passages], batch_size=32, show_progress_bar=False),
        dtype="float32")
    return 1.0 / (1.0 + np.exp(-logits))


def embedding_dim() -> int:
    return int(get_embedder().get_sentence_embedding_dimension())
