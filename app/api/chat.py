"""/chat: one question, answered whole or as a stream.

`POST /chat` waits for the complete answer and returns it as JSON. It is the
simple client's endpoint: a script, a test, the golden set run over HTTP.

`POST /chat/stream` returns server-sent events: the GenAI Lesson 7 streaming,
seen from the server's side. The model's fragments are forwarded as they arrive,
so a user reads the first words while the rest are being written. Five event
names make up the whole vocabulary, each with a schema in `schemas.py`:

    delta       {text}                         zero or more, in order
    citations   {citations, refused, reason}   once, after the last delta
    usage       {model, n_in, n_out, usd}      once
    done        {conversation_id, seconds}     last; the turn is saved
    error       {message}                      instead of the rest, if anything fails

`error` exists because of when a stream fails. The status line, 200, is sent
with the first byte. A failure after that cannot become a 500; the only way to
tell the client is inside the stream. So everything that can be checked before
the first byte is checked in a dependency, which runs before the response
starts and can still answer 404 or 422.

Both endpoints save the question and the answer with `db.record_turn`, so a
conversation exists outside the process that served it. Lesson 4 is where the
stored history starts being sent back to the model; here it is only kept.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app import db
from app.api.schemas import (ChatRequest, ChatResponse, CitationsEvent, DeltaEvent,
                             DoneEvent, ErrorEvent, Usage)
from app.generation import Answer, answer_question, stream_answer
from app.logs import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def checked(req: ChatRequest) -> ChatRequest:
    """Everything that can fail with a status code, checked before the first byte."""
    if req.conversation_id is not None and not db.conversation_exists(req.conversation_id):
        raise HTTPException(404, f"no conversation {req.conversation_id}")
    return req


def record(req: ChatRequest, a: Answer) -> int:
    return db.record_turn(req.question, a.text, citations=a.citations,
                          refused=a.refused, model=a.model, n_in=a.n_in,
                          n_out=a.n_out, usd=a.usd, conversation_id=req.conversation_id)


def usage(a: Answer) -> Usage:
    return Usage(model=a.model, n_in=a.n_in, n_out=a.n_out, usd=round(a.usd, 6))


@router.post("", response_model=ChatResponse)
def chat(req: Annotated[ChatRequest, Depends(checked)]) -> ChatResponse:
    """Answer one question and return it whole."""
    a = answer_question(req.question, top_k=req.top_k)
    return ChatResponse(conversation_id=record(req, a), answer=a.text, refused=a.refused,
                        reason=a.reason, citations=a.citations, usage=usage(a),
                        seconds=round(a.seconds, 2))


@router.post("/stream", response_class=EventSourceResponse)
def chat_stream(req: Annotated[ChatRequest, Depends(checked)]) -> Iterable[ServerSentEvent]:
    """Answer one question as server-sent events: delta*, citations, usage, done."""
    try:
        for x in stream_answer(req.question, top_k=req.top_k):
            if isinstance(x, str):
                yield ServerSentEvent(event="delta", data=DeltaEvent(text=x))
            else:
                a = x
        yield ServerSentEvent(event="citations", data=CitationsEvent(
            citations=a.citations, refused=a.refused, reason=a.reason))
        yield ServerSentEvent(event="usage", data=usage(a))
        conv = record(req, a)
    except Exception as e:                  # too late for a status code; say it in-band
        log.exception("chat stream failed")
        yield ServerSentEvent(event="error", data=ErrorEvent(message=f"{type(e).__name__}: {e}"))
        return
    yield ServerSentEvent(event="done", data=DoneEvent(conversation_id=conv,
                                                        seconds=round(a.seconds, 2)))
