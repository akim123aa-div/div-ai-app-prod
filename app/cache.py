"""The response cache: an answer paid for once and served again. Lesson 4.

A cache is only as correct as its key. Serve a stored answer when anything that
shaped it has changed and the cache returns a wrong answer, quickly and with
confidence. So the key holds every input to the answer, and each part is in it
for a reason that fits in a sentence:

    query           the standalone question, normalised. It is the condensed one,
                    not the words typed, because "and the year before?" means a
                    different thing in every conversation.
    corpus          the retriever's fingerprint. An upload or a delete changes what
                    can be retrieved, so an answer from before it may cite a chunk
                    that is gone or miss one that has arrived.
    prompt_version  a hash of the prompt files. A new prompt is a different
                    instruction, and the old answers were written under the old one.
    model           a different model writes a different answer, and you changed
                    model because you wanted that difference.
    top_k           more or fewer context blocks is a different context.

What is not in the key is a decision too. The conversation history is left out:
the standalone query already carries what the history contributes to retrieval,
and keying on it would make every follow-up a miss. The cost is that a cached
answer is worded without that conversation's history. User ID is left out, so
one user's answer serves another. That is right for a shared corpus and wrong
the day documents become private to a user.

Two kinds of answer are never stored: one written by the fallback provider,
which is the degraded mode and should not outlive the outage, and a refusal from
the retrieval gate, which cost nothing to produce.

The cache is a Postgres table here. Production puts it in Redis, with an expiry,
because it is read on every request and losing it costs only money.
"""

from __future__ import annotations

import hashlib
import json
import re

from app import db
from app.config import settings
from app.generation import Answer, prompt_version
from app.llm import get_client
from app.retrieval import get_retriever


def normalise(query: str) -> str:
    """Case, spacing and a trailing question mark change no answer."""
    return re.sub(r"\s+", " ", query.lower()).strip().rstrip("?.! ")


def key_parts(query: str, top_k: int | None = None) -> dict:
    return {"query": normalise(query), "corpus": get_retriever().fingerprint,
            "prompt_version": prompt_version(), "model": get_client().model,
            "top_k": top_k or settings.top_k}


def make_key(parts: dict) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:24]


def lookup(parts: dict) -> dict | None:
    if not settings.cache_answers:
        return None
    return db.cache_get(make_key(parts))


def store(parts: dict, a: Answer) -> bool:
    """Keep an answer the primary model was paid to write. Returns whether it did."""
    if not settings.cache_answers or a.fallback or a.n_in == 0:
        return False
    db.cache_put(make_key(parts), query=parts["query"], corpus=parts["corpus"],
                 prompt_version=parts["prompt_version"], model=parts["model"],
                 answer={"text": a.text, "citations": a.citations, "sources": a.sources,
                         "refused": a.refused, "reason": a.reason},
                 n_tokens=a.n_in + a.n_out, usd=a.usd)
    return True
