"""/chat: one turn of a conversation, answered whole or as a stream.

`POST /chat` waits for the complete answer and returns it as JSON. It is the
simple client's endpoint: a script, a test, the golden set run over HTTP.

`POST /chat/stream` returns server-sent events: the GenAI Lesson 7 streaming,
seen from the server's side. The model's fragments are forwarded as they arrive,
so a user reads the first words while the rest are being written. Five event
names make up the whole vocabulary, each with a schema in `schemas.py`:

    delta       {text}                         zero or more, in order
    citations   {citations, refused, reason}   once, after the last delta
    usage       {model, n_in, n_out, usd}      once
    done        {conversation_id, seconds,     last; the turn is saved
                 query, cached, window}
    error       {message}                      instead of the rest, if anything fails

`error` exists because of when a stream fails. The status line, 200, is sent
with the first byte. A failure after that cannot become a 500; the only way to
tell the client is inside the stream. So everything that can be checked before
the first byte is checked in a dependency, which runs before the response
starts and can still answer with a status code.

Lesson 4 adds two of those checks, and moves the work into `conversation.py`:

    who      the X-User-Id header names the user. Missing means "anonymous".
    owner    a conversation started by one user cannot be continued by another: 404,
             the same answer as for one that does not exist, so IDs cannot be probed
    budget   a user who has spent their tokens gets 429 and a Retry-After header

The handler is still four lines. Memory, condensation, the cache and the
fallback are all behind `answer_turn`, where the command line can reach them too.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app import budget, db
from app.api.schemas import (ChatRequest, ChatResponse, CitationsEvent, DeltaEvent,
                             DoneEvent, ErrorEvent, Usage)
from app.conversation import Turn, answer_turn, stream_turn
from app.logs import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def user(x_user_id: Annotated[str | None, Header(max_length=64)] = None) -> str:
    """Who is asking, as far as a header can say. Not authentication; see budget.py."""
    return x_user_id or budget.ANONYMOUS


def checked(req: ChatRequest, user_id: Annotated[str, Depends(user)]) -> ChatRequest:
    """Everything that can fail with a status code, checked before the first byte."""
    if req.conversation_id is not None:
        exists, owner = db.conversation_owner(req.conversation_id)
        if not exists or (owner is not None and owner != user_id):
            raise HTTPException(404, f"no conversation {req.conversation_id}")
    b = budget.check(user_id)
    if b.exhausted:
        raise HTTPException(
            429, f"{user_id!r} has used {b.used:,} of {b.limit:,} tokens in the last "
                 f"{budget.WINDOW_HOURS} hours",
            headers={"Retry-After": str(int(b.reset_seconds) or 1)})
    return req


def usage(t: Turn) -> Usage:
    return Usage(model=t.answer.model, n_in=t.n_in, n_out=t.n_out, usd=round(t.usd, 6))


@router.post("", response_model=ChatResponse)
def chat(req: Annotated[ChatRequest, Depends(checked)],
         user_id: Annotated[str, Depends(user)]) -> ChatResponse:
    """Answer one turn and return it whole."""
    t = answer_turn(req.question, req.conversation_id, user_id, top_k=req.top_k)
    a = t.answer
    return ChatResponse(conversation_id=t.conversation_id, answer=a.text, refused=a.refused,
                        reason=a.reason, citations=a.citations, usage=usage(t),
                        seconds=round(a.seconds, 2), query=t.query, cached=t.cached,
                        window=t.window.info())


@router.post("/stream", response_class=EventSourceResponse)
def chat_stream(req: Annotated[ChatRequest, Depends(checked)],
                user_id: Annotated[str, Depends(user)]) -> Iterable[ServerSentEvent]:
    """Answer one turn as server-sent events: delta*, citations, usage, done."""
    try:
        for x in stream_turn(req.question, req.conversation_id, user_id, top_k=req.top_k):
            if isinstance(x, str):
                yield ServerSentEvent(event="delta", data=DeltaEvent(text=x))
            else:
                t = x
        a = t.answer
        yield ServerSentEvent(event="citations", data=CitationsEvent(
            citations=a.citations, refused=a.refused, reason=a.reason))
        yield ServerSentEvent(event="usage", data=usage(t))
    except Exception as e:                  # too late for a status code; say it in-band
        log.exception("chat stream failed")
        yield ServerSentEvent(event="error", data=ErrorEvent(message=f"{type(e).__name__}: {e}"))
        return
    yield ServerSentEvent(event="done", data=DoneEvent(
        conversation_id=t.conversation_id, seconds=round(a.seconds, 2), query=t.query,
        cached=t.cached, window=t.window.info()))
