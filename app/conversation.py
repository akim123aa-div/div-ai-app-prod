"""One turn of a conversation, end to end. Lesson 4.

Up to Lesson 3 the chat handler called `answer_question` and saved what came
back. The pipeline could answer a question, but nothing in it answered a
conversation. This module sits between the two, and every chat request goes
through it:

    1. window    the past turns that go back to the model      memory.py
    2. condense  the new question as a standalone query         generation.condense
    3. cache     this query, these documents, already answered? cache.py
    4. answer    retrieve, gate, generate, with the window      generation.py
    5. record    both messages, and what the whole turn cost    db.record_turn

The budget is checked before step 1, by the API. A refused request should cost
nothing and get a real status code, and Lesson 3 settled that anything checkable
is checked before the first byte. It is not checked here, so the command line
and the golden set are not budgeted.

The cost recorded for a turn is all of it: the condensing call, a summarising
call if this turn made one, and the answer. That total is what the budget sums.

`answer_turn` and `stream_turn` share every step except generation, the same
split that `generation.py` makes.

Each step logs one line at INFO, so a terminal running `python -m app.serve`
reads as the story of a turn: who asked, what the window held, what the
question became, whether the cache answered, and what the turn cost. Lesson 5
watches it live; Lesson 8 keeps it as a trace you can search.

From Lesson 8 a turn is the root of a trace (`tracing.py`), named after its user
and its conversation, and its trace ID goes back to the client with the answer.
The log lines stay. They are for whoever is watching; the trace is for whoever
comes looking afterwards.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from app import cache, db, memory, tracing
from app.generation import Answer, answer_question, condense, stream_answer
from app.llm import Completion
from app.logs import get_logger
from app.tracing import observe

log = get_logger(__name__)


@dataclass
class Turn:
    """One question in one conversation, and everything it cost."""

    question: str                       # as the user typed it
    query: str                          # what the retriever saw
    answer: Answer
    window: memory.Window
    cached: bool = False
    calls: list[Completion] = field(default_factory=list)   # condense, summarise
    conversation_id: int | None = None
    trace_id: str | None = None                             # where to read it (Lesson 8)

    @property
    def n_in(self) -> int:
        return self.answer.n_in + sum(c.n_in for c in self.calls)

    @property
    def n_out(self) -> int:
        return self.answer.n_out + sum(c.n_out for c in self.calls)

    @property
    def usd(self) -> float:
        return self.answer.usd + sum(c.usd for c in self.calls)


def prepare(question: str, conversation_id: int | None, user_id: str | None,
            top_k: int | None):
    """Steps 1 to 3's inputs: the window, the standalone query, the cache key."""
    log.info("turn: %s asks %r in conversation %s", user_id, question[:80],
             conversation_id or "(new)")
    w = memory.window(conversation_id)
    log.info("window: %d past messages + %d-token summary = %d tokens (all of it: %d)",
             w.recent, w.summary_tokens, w.tokens, w.full_tokens)
    query, c = condense(w.messages, question)
    if query != question:
        log.info("condensed to %r", query[:100])
    calls = [x for x in (w.call, c) if x is not None]
    return w, query, calls, cache.key_parts(query, top_k)


@observe(name="cache", capture_output=False)
def lookup(parts: dict) -> dict | None:
    hit = cache.lookup(parts)
    log.info("cache %s", "hit: no retrieval, no model call" if hit else "miss")
    tracing.output("hit" if hit else "miss", key=cache.make_key(parts))
    return hit


def from_cache(query: str, hit: dict) -> Answer:
    """A stored answer, as if it had just been written, at no cost."""
    return Answer(question=query, text=hit["text"], citations=hit["citations"],
                  sources=hit.get("sources", []), refused=hit["refused"],
                  reason=hit["reason"], model=hit["model"])


def record(t: Turn, user_id: str | None) -> Turn:
    a = t.answer
    t.conversation_id = db.record_turn(
        t.question, a.text, citations=a.citations, refused=a.refused, model=a.model,
        n_in=t.n_in, n_out=t.n_out, usd=t.usd,
        conversation_id=t.conversation_id, user_id=user_id)
    t.trace_id = tracing.trace_id()
    tracing.label_turn(user_id, t.conversation_id, {
        "answer": a.text, "query": t.query, "cached": t.cached, "refused": a.refused,
        "citations": [c["source"] for c in a.citations], "model": a.model,
        "usd": round(t.usd, 6)})
    outcome = (f"refused ({a.reason})" if a.refused
               else f"answered, {len(a.citations)} citation(s)")
    log.info("saved to conversation %d: %s, %d in / %d out, $%.5f, %.1fs%s",
             t.conversation_id, outcome, t.n_in, t.n_out, t.usd, a.seconds,
             ", from the cache" if t.cached else "")
    return t


@observe(name="turn", capture_output=False)
def answer_turn(question: str, conversation_id: int | None = None,
                user_id: str | None = None, top_k: int | None = None) -> Turn:
    """One turn, answered whole."""
    t0 = time.perf_counter()
    w, query, calls, parts = prepare(question, conversation_id, user_id, top_k)
    if hit := lookup(parts):
        a, cached = from_cache(query, hit), True
    else:
        a, cached = answer_question(query, top_k=top_k, history=w.messages), False
        cache.store(parts, a)
    a.seconds = time.perf_counter() - t0
    return record(Turn(question, query, a, w, cached, calls, conversation_id), user_id)


@observe(name="turn", capture_output=False)
def stream_turn(question: str, conversation_id: int | None = None,
                user_id: str | None = None,
                top_k: int | None = None) -> Iterator[str | Turn]:
    """The same turn, as text fragments while it is written, then the `Turn`.

    A cache hit has no fragments to wait for, so its whole text goes out as one.
    """
    t0 = time.perf_counter()
    w, query, calls, parts = prepare(question, conversation_id, user_id, top_k)
    if hit := lookup(parts):
        a, cached = from_cache(query, hit), True
        if not a.refused:
            yield a.text
    else:
        for x in stream_answer(query, top_k=top_k, history=w.messages):
            if isinstance(x, Answer):
                a = x
            else:
                yield x
        cached = False
        cache.store(parts, a)
    a.seconds = time.perf_counter() - t0
    yield record(Turn(question, query, a, w, cached, calls, conversation_id), user_id)
