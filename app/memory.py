"""The conversation window: which past turns go back to the model. Lesson 4.

The model remembers nothing between calls (GenAI Lesson 7). Every turn it sees
is a turn this application chose to send again, and paid for again. So the
application owns the memory, and this module is the rule for what it sends:

    history  =  a summary of the older turns  +  the recent turns in full
                and the two together stay under HISTORY_TOKENS

Sending everything works for a while. A turn here is about 150 tokens of
question and answer, so twenty turns is 3,000 tokens resent with every new
question, and the bill for a conversation grows with the square of its length.

When the turns not yet summarised would push the history over the budget, the
oldest of them are folded into the summary, by one model call, until the recent
part is at most half the budget. The summary itself is held to a quarter of the
budget (and to SUMMARY_TOKENS). So right after a fold the history is at most
three quarters full, and the next few turns fit before the next fold. Folding
down to half rather than to just under the line is what keeps the summarising
call to once every few turns instead of once a turn. The summary is stored in
`summaries`, so the next request starts from it rather than summarising the
whole conversation again.

The quarter was learned the hard way. With the summary allowed half the budget,
it grew to fill its half, the recent turns filled theirs, and every turn after
the second fold folded again.

Two details in how a past turn is written back:

* citation markers are removed. "[2]" pointed at a context block of that turn,
  and that block is not in this window. Left in, it points at the wrong block.
* a refusal is replaced by a sentence saying so. NOT_IN_CONTEXT is a sentinel
  for the code, and in a history it reads like an instruction to repeat it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app import db
from app.chunking import ENC
from app.config import settings
from app.generation import REFUSAL, prompt, strip_markers
from app import tracing
from app.llm import Completion, get_client
from app.logs import get_logger
from app.tracing import observe

log = get_logger(__name__)

NO_ANSWER = "(No answer: the documents did not cover this.)"


def count(text: str) -> int:
    return len(ENC.encode(text))


@dataclass
class Window:
    """What one turn sends back to the model, and what it left out."""

    messages: list[dict] = field(default_factory=list)  # summary, then turns: role, content
    summary: str = ""
    summary_tokens: int = 0
    recent: int = 0               # past messages sent in full
    folded: int = 0               # past messages present only through the summary
    tokens: int = 0               # summary + recent: the number HISTORY_TOKENS limits
    full_tokens: int = 0          # what sending every past message in full would cost
    call: Completion | None = None    # the summarising call, if this turn made one

    def info(self) -> dict:
        return {"summary_tokens": self.summary_tokens, "recent_messages": self.recent,
                "folded_messages": self.folded, "tokens": self.tokens,
                "full_tokens": self.full_tokens}


def as_history(m: dict) -> str:
    """A stored message, as it is written back into a later window."""
    if m["role"] == "assistant" and (m["refused"] or m["content"].startswith(REFUSAL)):
        return NO_ANSWER
    return strip_markers(m["content"]).strip()


def newest_turns(msgs: list[dict], limit: int) -> list[dict]:
    """The longest run of whole turns, newest first, that fits in `limit` tokens.

    A turn is a question and its answer, and it is kept or folded as a pair: an
    answer without its question is not much use to anyone. The newest turn is
    always kept, even if it alone is over the limit.
    """
    turns: list[list[dict]] = []
    for m in msgs:
        if m["role"] == "user" or not turns:
            turns.append([m])
        else:
            turns[-1].append(m)
    kept, used = [], 0
    for t in reversed(turns):
        n = sum(m["n"] for m in t)
        if kept and used + n > limit:
            break
        kept[:0], used = t, used + n
    return kept


def summary_limit() -> int:
    """A quarter of the history budget, and never more than SUMMARY_TOKENS."""
    return min(settings.summary_tokens, settings.history_tokens // 4)


def summarise(summary: str, leaving: list[dict]) -> tuple[str, Completion]:
    """Fold the turns leaving the window into the summary. One model call.

    The limit is said in words as well as enforced in tokens: `max_tokens` alone
    cuts a summary off mid-sentence, and the model can only aim at a length it
    has been told. It overshoots anyway, often enough that a summary cut off by
    the limit is trimmed back to its last whole sentence rather than stored as
    "Brave B". A token is about three quarters of a word here.
    """
    turns = "\n".join(f"{m['role']}: {m['text']}" for m in leaving)
    c = get_client().complete(
        [{"role": "system", "content": prompt("summarise")},
         {"role": "user", "content": f"Summary so far:\n{summary or '(empty)'}\n\n"
                                     f"Turns leaving the window:\n{turns}\n\n"
                                     f"Keep the new summary under {summary_limit() * 2 // 3} words.\n\n"
                                     f"New summary:"}],
        max_tokens=summary_limit())
    text = c.text.strip()
    if c.finish == "length":                # cut off anyway: keep the last whole sentence
        text = text[: text.rfind(". ") + 1] or text
    return text, c


@observe(name="window", capture_output=False)
def window(conversation_id: int | None) -> Window:
    """Build this turn's window from Postgres, summarising first if it has to."""
    if conversation_id is None:
        return Window()
    msgs = db.load_messages(conversation_id)
    for m in msgs:
        m["text"] = as_history(m)
        m["n"] = count(m["text"])

    stored = db.get_summary(conversation_id)
    summary, s_tok, upto = ((stored["text"], stored["n_tokens"], stored["upto_message_id"])
                            if stored else ("", 0, 0))
    live = [m for m in msgs if m["id"] > upto]
    call = None

    if s_tok + sum(m["n"] for m in live) > settings.history_tokens:
        keep = newest_turns(live, settings.history_tokens // 2)
        leaving = live[: len(live) - len(keep)]
        if leaving:
            summary, call = summarise(summary, leaving)
            s_tok = count(summary)
            db.save_summary(conversation_id, leaving[-1]["id"], summary, s_tok)
            log.info("conversation %d: folded %d messages into a %d-token summary",
                     conversation_id, len(leaving), s_tok)
            live = keep

    messages = ([{"role": "system",
                  "content": f"Summary of the earlier conversation:\n{summary}"}]
                if summary else [])
    messages += [{"role": m["role"], "content": m["text"]} for m in live]
    tracing.output(messages, folded=len(msgs) - len(live), summarised=call is not None)
    return Window(messages=messages, summary=summary, summary_tokens=s_tok,
                  recent=len(live), folded=len(msgs) - len(live),
                  tokens=s_tok + sum(m["n"] for m in live),
                  full_tokens=sum(m["n"] for m in msgs), call=call)
