"""Question in, grounded answer with citations out. GenAI Lesson 16, as a module.

Three things from that lesson are load-bearing and all three are here:

* numbered context blocks, so there is something for a citation to point at
* a citation rule, which is worth more as a constraint on the model than as a
  feature for the reader
* two refusal gates: the reranker's score, checked before the generator runs,
  and a prompt clause that makes NOT_IN_CONTEXT an allowed reply

The prompts live in `app/prompts/*.md` rather than in this file. A prompt is
edited far more often than the code around it, it is read by people who do not
write Python, and it wants a diff of its own.

Lesson 4 adds three things, none of which changes a single-question answer's logic:

* `history`, the conversation window from `memory.py`, placed between the system
  prompt and the question, so a follow-up is answered knowing what came before
* the injection guard: retrieved text wrapped in <document> tags, and a prompt
  clause that says nothing inside them is an instruction (`GUARD_CONTEXT`)
* `prompt_version()`, a hash of everything that shapes the answer apart from the
  question and the context. The response cache keys on it, so editing a prompt
  file retires every answer the old prompt wrote, without anyone remembering to.

There are two ways to get an answer, and they share every step but one.
`answer_question` waits for the whole reply; `stream_answer` yields it in
fragments as the model writes it. Retrieval, the gate, the prompt and the
citation parsing are the same functions in both, so the API's two chat
endpoints cannot drift apart.

Lesson 8 traces the three steps here, `answer`, `retrieve` and `condense`, and
gives every `Answer` its `sources`: the passages it was written from, with their
scores. The API returns them, so the golden set can score retrieval over HTTP.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache

from app import tracing
from app.config import ROOT, settings
from app.llm import Completion, get_client
from app.logs import get_logger
from app.retrieval import Hit, get_retriever
from app.tracing import observe

log = get_logger(__name__)

REFUSAL = "NOT_IN_CONTEXT"
CITE = re.compile(r"\[(\d+)\]")


@lru_cache(maxsize=8)
def prompt(name: str) -> str:
    """Prompts are files. Cached, because a request should not hit the disk."""
    return (ROOT / "app" / "prompts" / f"{name}.md").read_text().strip()


@dataclass
class Answer:
    """What one question produced, and enough of the trail to check it."""

    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)   # the hits, as plain data (Lesson 8)
    citations: list[dict] = field(default_factory=list)
    refused: bool = False
    reason: str = ""
    model: str | None = None
    n_in: int = 0
    n_out: int = 0
    usd: float = 0.0
    seconds: float = 0.0
    fallback: bool = False

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.text,
                "citations": self.citations, "refused": self.refused,
                "reason": self.reason, "model": self.model,
                "tokens": {"in": self.n_in, "out": self.n_out},
                "usd": round(self.usd, 6),
                "seconds": round(self.seconds, 2),
                "sources": self.sources}


def sources(hits: list[Hit]) -> list[dict]:
    """The retrieved passages as plain data: what a response, a cache row or a trace holds."""
    return [{"chunk_id": h.chunk["id"], "doc": h.chunk["doc"], "page": h.chunk["page"],
             "source": h.source, "score": round(h.score, 3)} for h in hits]


def system_prompt() -> str:
    """The answer prompt, plus the injection clause when the guard is on."""
    p = prompt("answer")
    return f"{p}\n\n{prompt('guard')}" if settings.guard_context else p


def prompt_version() -> str:
    """A content hash, not a number someone bumps. If the words change, it changes."""
    shape = system_prompt() + ("|guarded" if settings.guard_context else "|plain")
    return hashlib.sha256(shape.encode()).hexdigest()[:10]


def strip_markers(text: str) -> str:
    """Remove [n] markers. In a past turn they point at blocks no longer in the window."""
    return re.sub(r"\s*\[\d+\]", "", text)


def build_context(hits: list[Hit]) -> str:
    """Numbered blocks, each with the source line a citation resolves back to.

    The block is the whole chunk. There is no second length limit here, and that
    is deliberate: `chunking.py` already bounds a chunk at CHUNK_TOKENS, so a
    limit in this module can only ever disagree with that one.

    It disagreed twice. The original 1100-character cut trimmed the tail off the
    median chunk and cost two of the golden set's answerable questions. Raising
    it to 2400 still cut 19 chunks, because the two limits are in different units
    and do not convert: on this corpus a 400-token chunk of table fragments runs
    to 3,752 characters, eleven characters per token against a typical three.

    The lesson generalises past this bug. A budget measured in characters, over
    data bounded in tokens, fails exactly on the pages where the characters are
    unusual, which on a corpus of annual reports is the pages holding the
    numbers. If you want to cap what reaches the model, cap it in tokens, and
    Lesson 4 is where that budget gets built.

    With the guard on (Lesson 4) each block is wrapped in <document> tags, so the
    prompt can say where data starts and stops. A tag the data can forge is not
    a boundary, so a closing tag inside a chunk is broken before it is sent.
    """
    if not settings.guard_context:
        return "\n\n".join(
            f"[{j}] {h.source}\n{h.chunk['text'].strip()}"
            for j, h in enumerate(hits, 1))
    return "\n\n".join(
        f'<document index="{j}" source="{h.source}">\n'
        f"{h.chunk['text'].strip().replace('</document', '</ document')}\n</document>"
        for j, h in enumerate(hits, 1))


def resolve_citations(text: str, hits: list[Hit]) -> list[dict]:
    """Turn the markers the model wrote into sources it could not have invented.

    The model emits an index. The mapping from index to document lives here, in
    code, where there is nothing to hallucinate.
    """
    out, seen = [], set()
    for m in CITE.findall(text):
        j = int(m)
        if 1 <= j <= len(hits) and j not in seen:
            seen.add(j)
            h = hits[j - 1]
            out.append({"marker": j, "chunk_id": h.chunk["id"], "doc": h.chunk["doc"],
                        "page": h.chunk["page"], "source": h.source,
                        "score": round(h.score, 3)})
    return out


@observe(name="retrieve", as_type="retriever", capture_output=False)
def retrieve(question: str, top_k: int | None = None,
             gate: float | None = None) -> tuple[list[Hit], str]:
    """The hits, and a reason to refuse before generating ('' if there is none).

    The gate is the reranker's: with RERANK off there is no score it can judge, so
    nothing is refused here and every question reaches the model.
    """
    gate = settings.gate if gate is None else gate
    hits = get_retriever().retrieve(question, top_k=top_k)
    best = max((h.score for h in hits), default=0.0)
    log.info("retrieved %d: %s", len(hits),
             ", ".join(f"{h.chunk['doc']} p{h.chunk['page']} {h.score:.2f}" for h in hits))
    why = ""
    if settings.rerank and best < gate:
        log.info("refused before generating: best reranker score %.3f < gate %.2f",
                 best, gate)
        why = f"retrieval gate: best score {best:.3f} below {gate:.2f}"
    tracing.output({"hits": sources(hits), "refused": why or None},
                   rerank=settings.rerank, gate=gate if settings.rerank else None)
    return hits, why


def answer_messages(question: str, hits: list[Hit],
                    history: list[dict] | None = None) -> list[dict]:
    """System prompt, then the conversation so far, then the context and question.

    The context goes with the newest question and nowhere else. Past turns are
    sent as text only: their context blocks are not resent, which is most of
    why a conversation's window grows by hundreds of tokens a turn, not thousands.
    """
    return [{"role": "system", "content": system_prompt()},
            *(history or []),
            {"role": "user",
             "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}"}]


def finish(question: str, hits: list[Hit], c: Completion, t0: float) -> Answer:
    """The model's reply, checked for a refusal and its citations resolved."""
    text = c.text.strip()
    refused = text.startswith(REFUSAL)
    tracing.output({"text": text, "refused": refused, "citations": CITE.findall(text)})
    return Answer(question=question, text=text, hits=hits, sources=sources(hits),
                  citations=[] if refused else resolve_citations(text, hits),
                  refused=refused,
                  reason="the model found no support in the context" if refused else "",
                  model=c.model, n_in=c.n_in, n_out=c.n_out, usd=c.usd,
                  seconds=time.perf_counter() - t0, fallback=c.fallback)


@observe(name="answer", capture_input=False, capture_output=False)
def answer_question(question: str, top_k: int | None = None, gate: float | None = None,
                    history: list[dict] | None = None) -> Answer:
    """Retrieve, decide whether to answer at all, then answer."""
    t0 = time.perf_counter()
    hits, why = retrieve(question, top_k, gate)
    if why:
        return Answer(question=question, text=REFUSAL, hits=hits, sources=sources(hits),
                      refused=True, reason=why, seconds=time.perf_counter() - t0)
    c = get_client().complete(answer_messages(question, hits, history), max_tokens=220)
    return finish(question, hits, c, t0)


@observe(name="answer", capture_input=False, capture_output=False)
def stream_answer(question: str, top_k: int | None = None, gate: float | None = None,
                  history: list[dict] | None = None) -> Iterator[str | Answer]:
    """The same answer, as text fragments while the model writes it, and then the
    finished `Answer`, with citations and cost, as the last item.

    Citations come last because they cannot come earlier: a marker is only known
    once the model has written it, and the list is only complete at the end.

    One thing is held back. A refusal is the literal NOT_IN_CONTEXT, and it
    arrives in fragments like "NOT", "_IN", "_CONTEXT". Streaming those would put
    the sentinel on a user's screen before anyone knew it was one. So nothing is
    sent while the text so far could still be the start of a refusal; the first
    fragment that rules it out releases everything held.
    """
    t0 = time.perf_counter()
    hits, why = retrieve(question, top_k, gate)
    if why:
        yield Answer(question=question, text=REFUSAL, hits=hits, sources=sources(hits),
                     refused=True, reason=why, seconds=time.perf_counter() - t0)
        return

    held, released, done = "", False, None
    for x in get_client().stream(answer_messages(question, hits, history), max_tokens=220):
        if isinstance(x, Completion):
            done = x
        elif released:
            yield x
        else:
            held += x
            head = held.lstrip()
            if not (REFUSAL.startswith(head) or head.startswith(REFUSAL)):
                released = True
                yield held
    a = finish(question, hits, done, t0)
    if held and not released and not a.refused:    # a reply shorter than the sentinel
        yield held
    yield a


@observe(name="condense", capture_output=False)
def condense(turns: list[dict], question: str) -> tuple[str, Completion | None]:
    """Rewrite a follow-up as a standalone query, the GenAI Lesson 16 step.

    The retriever sees one string and no conversation, so "and the year before?"
    has to become "Aurora Innovation patents at year end 2021" before it gets
    there. `turns` is the window from `memory.py`: its summary, then the recent
    turns. Returns the query and the call that made it, for the accounting; no
    history means no call, and the question goes through as it is.
    """
    if not turns:
        tracing.output(question, call="none: no history")
        return question, None
    history = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    c = get_client().complete(
        [{"role": "system", "content": prompt("condense")},
         {"role": "user", "content": f"{history}\nuser: {question}\n\nStandalone query:"}],
        max_tokens=60)
    query = c.text.strip().strip('"') or question
    tracing.output(query)
    return query, c
