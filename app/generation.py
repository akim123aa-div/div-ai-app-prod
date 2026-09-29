"""Question in, grounded answer with citations out. GenAI Lesson 16, as a module.

Three things from that lesson are load-bearing and all three are here:

* numbered context blocks, so there is something for a citation to point at
* a citation rule, which is worth more as a constraint on the model than as a
  feature for the reader
* two refusal gates: the reranker's score, checked before the generator runs,
  and a prompt clause that makes NOT_IN_CONTEXT an allowed reply

The prompts live in `app/prompts/*.md` rather than in this file. A prompt is
edited far more often than the code around it, it is read by people who do not
write Python, and it wants a diff of its own. Lesson 4 adds a version to it.

There are two ways to get an answer, and they share every step but one.
`answer_question` waits for the whole reply; `stream_answer` yields it in
fragments as the model writes it. Retrieval, the gate, the prompt and the
citation parsing are the same functions in both, so the API's two chat
endpoints cannot drift apart.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache

from app.config import ROOT, settings
from app.llm import Completion, get_client
from app.logs import get_logger
from app.retrieval import Hit, get_retriever

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
    citations: list[dict] = field(default_factory=list)
    refused: bool = False
    reason: str = ""
    model: str | None = None
    n_in: int = 0
    n_out: int = 0
    usd: float = 0.0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.text,
                "citations": self.citations, "refused": self.refused,
                "reason": self.reason, "model": self.model,
                "tokens": {"in": self.n_in, "out": self.n_out},
                "usd": round(self.usd, 6),
                "seconds": round(self.seconds, 2),
                "sources": [h.source for h in self.hits]}


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
    """
    return "\n\n".join(
        f"[{j}] {h.source}\n{h.chunk['text'].strip()}"
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


def retrieve(question: str, top_k: int | None = None,
             gate: float | None = None) -> tuple[list[Hit], str]:
    """The hits, and a reason to refuse before generating ('' if there is none)."""
    gate = settings.gate if gate is None else gate
    hits = get_retriever().retrieve(question, top_k=top_k)
    best = max((h.score for h in hits), default=0.0)
    if best < gate:
        log.info("refused before generating: best reranker score %.3f < gate %.2f",
                 best, gate)
        return hits, f"retrieval gate: best score {best:.3f} below {gate:.2f}"
    return hits, ""


def answer_messages(question: str, hits: list[Hit]) -> list[dict]:
    return [{"role": "system", "content": prompt("answer")},
            {"role": "user",
             "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}"}]


def finish(question: str, hits: list[Hit], c: Completion, t0: float) -> Answer:
    """The model's reply, checked for a refusal and its citations resolved."""
    text = c.text.strip()
    refused = text.startswith(REFUSAL)
    return Answer(question=question, text=text, hits=hits,
                  citations=[] if refused else resolve_citations(text, hits),
                  refused=refused,
                  reason="the model found no support in the context" if refused else "",
                  model=c.model, n_in=c.n_in, n_out=c.n_out, usd=c.usd,
                  seconds=time.perf_counter() - t0)


def answer_question(question: str, top_k: int | None = None,
                    gate: float | None = None) -> Answer:
    """Retrieve, decide whether to answer at all, then answer."""
    t0 = time.perf_counter()
    hits, why = retrieve(question, top_k, gate)
    if why:
        return Answer(question=question, text=REFUSAL, hits=hits, refused=True,
                      reason=why, seconds=time.perf_counter() - t0)
    c = get_client().complete(answer_messages(question, hits), max_tokens=220)
    return finish(question, hits, c, t0)


def stream_answer(question: str, top_k: int | None = None,
                  gate: float | None = None) -> Iterator[str | Answer]:
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
        yield Answer(question=question, text=REFUSAL, hits=hits, refused=True,
                     reason=why, seconds=time.perf_counter() - t0)
        return

    held, released, done = "", False, None
    for x in get_client().stream(answer_messages(question, hits), max_tokens=220):
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


def condense(turns: list[dict], question: str) -> str:
    """Rewrite a follow-up as a standalone query. Unused until Lesson 4, which is
    when the application starts owning a conversation. Here so that the pipeline
    that arrives in Lesson 3 is already the whole pipeline."""
    if not turns:
        return question
    history = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    return get_client().complete(
        [{"role": "system", "content": prompt("condense")},
         {"role": "user", "content": f"{history}\nuser: {question}\n\nStandalone query:"}],
        max_tokens=60).text.strip().strip('"')
