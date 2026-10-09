"""Traces: what happened inside one request, written down while it happens. Lesson 8.

GenAI Lesson 17 put Langfuse's `@observe` on a notebook's functions. Here the same
decorator goes on the request path, so that every turn the API answers leaves a
trace, and a bad answer can be found later and read step by step.

A trace is a tree. Its root is one turn, and each child is one step the turn took,
with what went in, what came out, and how long it took:

    turn                 conversation.py   who asked, in which conversation, what came back
      window             memory.py         the past turns sent back to the model
        llm              llm.py            the summarising call, when the window folds
      condense           generation.py     the follow-up rewritten as a standalone query
        llm
      cache              conversation.py   hit or miss, and every part of the key
      answer             generation.py     retrieve, gate, generate
        retrieve         generation.py     the hits with their scores, and the gate's verdict
          rerank         retrieval.py      the cross-encoder over the shortlist
        llm              llm.py            the whole prompt, the model, tokens and cost

The log lines from Lesson 5 tell the same story, once, to whoever is watching a
terminal. A trace keeps it, with the prompt and the passages, and finds it again by
conversation, by user or by trace ID.

Three rules, each one a line or two below:

* Tracing is optional. With no LANGFUSE_PUBLIC_KEY every decorator does nothing,
  so the tests, the clean-clone test and a laptop without Langfuse run unchanged.
* A tracer that is down costs the traces, never the answer. Spans are sent in
  batches from a background thread; if Langfuse does not answer they are dropped
  and a line is logged. That is why the API does not wait for Langfuse in compose.
* A trace holds what the prompt held: pages of the documents. Lesson 7's argument
  for a model on your own hardware applies to traces too, so Langfuse runs in
  compose, and nothing in a trace leaves the machine.

This module is the only one that imports `langfuse`, as `llm.py` is the only one
that knows a model provider's URL. The rest of the package says `@observe` and
calls the three helpers at the bottom.
"""

from __future__ import annotations

from typing import Any

from langfuse import Langfuse, observe, propagate_attributes

from app.config import settings
from app.logs import get_logger

log = get_logger(__name__)

__all__ = ["observe", "client", "enabled", "output", "generation", "label_turn", "trace_id",
           "url", "flush"]

enabled = bool(settings.langfuse_public_key and settings.langfuse_secret_key)

# One client per process. Built here, from settings, before anything is decorated:
# `@observe` finds it as the process's Langfuse client. Disabled, it is a no-op.
client = Langfuse(public_key=settings.langfuse_public_key or None,
                   secret_key=settings.langfuse_secret_key or None,
                   base_url=settings.langfuse_base_url,
                   tracing_enabled=enabled)
if enabled:
    log.info("tracing to Langfuse at %s", settings.langfuse_base_url)


def output(value: Any = None, *, level: str | None = None,
           status: str | None = None, **metadata: Any) -> None:
    """Set what the current step produced, when its return value is not the useful part.

    A `level` of WARNING or ERROR makes the step stand out in the trace.
    """
    if enabled:
        client.update_current_span(output=value, metadata=metadata or None,
                                    level=level, status_message=status)


def generation(*, messages: list[dict], text: str, model: str, n_in: int, n_out: int,
               usd: float, max_tokens: int, temperature: float) -> None:
    """Fill in the current model call: the whole prompt, the reply, tokens and cost."""
    if enabled:
        client.update_current_generation(
            input=messages, output=text, model=model,
            model_parameters={"max_tokens": max_tokens, "temperature": temperature},
            usage_details={"input": n_in, "output": n_out},
            cost_details={"total": usd})


def label_turn(user_id: str | None, conversation_id: int | None, result: dict) -> None:
    """Name the turn's trace after its user and conversation, and set what it returned.

    Langfuse calls a conversation a session. The conversation ID of a new conversation
    is only known once the turn is saved, which is why this runs last, not first.
    """
    if not enabled:
        return
    with propagate_attributes(user_id=user_id or "anonymous",
                              session_id=str(conversation_id) if conversation_id else None,
                              metadata={"rerank": str(settings.rerank).lower()}):
        client.update_current_span(output=result)


def trace_id() -> str | None:
    """The ID of the trace this code is running inside, to hand back to the client."""
    return client.get_current_trace_id() if enabled else None


def url(trace_id: str) -> str:
    """Where a person reads that trace: Langfuse's page for it."""
    return client.get_trace_url(trace_id=trace_id)


def flush() -> None:
    """Send what is still buffered. A short-lived process calls this before it exits."""
    if enabled:
        client.flush()
