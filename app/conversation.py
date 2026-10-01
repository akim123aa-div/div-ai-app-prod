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
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from app import cache, db, memory
from app.generation import Answer, answer_question, condense, stream_answer
from app.llm import Completion


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

    @property
    def n_in(self) -> int:
        return self.answer.n_in + sum(c.n_in for c in self.calls)

    @property
    def n_out(self) -> int:
        return self.answer.n_out + sum(c.n_out for c in self.calls)

    @property
    def usd(self) -> float:
        return self.answer.usd + sum(c.usd for c in self.calls)


def prepare(question: str, conversation_id: int | None, top_k: int | None):
    """Steps 1 to 3's inputs: the window, the standalone query, the cache key."""
    w = memory.window(conversation_id)
    query, c = condense(w.messages, question)
    calls = [x for x in (w.call, c) if x is not None]
    return w, query, calls, cache.key_parts(query, top_k)


def from_cache(query: str, hit: dict) -> Answer:
    """A stored answer, as if it had just been written, at no cost."""
    return Answer(question=query, text=hit["text"], citations=hit["citations"],
                  refused=hit["refused"], reason=hit["reason"], model=hit["model"])


def record(t: Turn, user_id: str | None) -> Turn:
    a = t.answer
    t.conversation_id = db.record_turn(
        t.question, a.text, citations=a.citations, refused=a.refused, model=a.model,
        n_in=t.n_in, n_out=t.n_out, usd=t.usd,
        conversation_id=t.conversation_id, user_id=user_id)
    return t


def answer_turn(question: str, conversation_id: int | None = None,
                user_id: str | None = None, top_k: int | None = None) -> Turn:
    """One turn, answered whole."""
    t0 = time.perf_counter()
    w, query, calls, parts = prepare(question, conversation_id, top_k)
    if hit := cache.lookup(parts):
        a, cached = from_cache(query, hit), True
    else:
        a, cached = answer_question(query, top_k=top_k, history=w.messages), False
        cache.store(parts, a)
    a.seconds = time.perf_counter() - t0
    return record(Turn(question, query, a, w, cached, calls, conversation_id), user_id)


def stream_turn(question: str, conversation_id: int | None = None,
                user_id: str | None = None,
                top_k: int | None = None) -> Iterator[str | Turn]:
    """The same turn, as text fragments while it is written, then the `Turn`.

    A cache hit has no fragments to wait for, so its whole text goes out as one.
    """
    t0 = time.perf_counter()
    w, query, calls, parts = prepare(question, conversation_id, top_k)
    if hit := cache.lookup(parts):
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
