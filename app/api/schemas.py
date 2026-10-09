"""Every body the API accepts or returns, declared once.

A schema does three jobs at the same time. It validates what comes in, so a
handler never sees an empty question or a `top_k` of minus one: FastAPI answers
422 before the handler runs. It shapes what goes out, so a column added to a
table does not leak into a response by accident. And it is the documentation:
the `/docs` page is generated from these classes, and so is any client someone
generates from `/openapi.json`.

The event models at the bottom are the streaming chat vocabulary. Lesson 5's UI
is written against them and against nothing else: `ui/client.py` reads these
field names, and no Python from this package.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---- documents ---------------------------------------------------------------
class DocumentOut(BaseModel):
    id: str
    title: str
    filename: str
    pages: int
    status: Literal["pending", "ready", "failed"]
    error: str | None = None
    chunks: int
    created_at: datetime


class ChunkOut(BaseModel):
    """The passage a citation points at (Lesson 5's sources panel)."""
    id: str
    doc: str
    title: str
    page: int
    section: str
    text: str


# ---- chat --------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = Field(
        default=None, description="add this turn to an existing conversation")
    top_k: int | None = Field(default=None, ge=1, le=20,
                              description="context blocks; the server default if unset")

    model_config = {"json_schema_extra": {"examples": [
        {"question": "How many patents did Aurora Innovation hold at year end?"}]}}


class Citation(BaseModel):
    marker: int
    chunk_id: str
    doc: str
    page: int
    source: str
    score: float


class Source(BaseModel):
    """One passage the answer was written from, cited or not (Lesson 8)."""
    chunk_id: str
    doc: str
    page: int
    source: str
    score: float


class Usage(BaseModel):
    model: str | None
    n_in: int
    n_out: int
    usd: float


class WindowInfo(BaseModel):
    """What this turn sent back of the conversation so far (Lesson 4)."""
    summary_tokens: int = Field(description="the summary standing in for older turns")
    recent_messages: int = Field(description="past messages sent in full")
    folded_messages: int = Field(description="past messages present only in the summary")
    tokens: int = Field(description="summary + recent: what HISTORY_TOKENS limits")
    full_tokens: int = Field(description="what sending the whole history would have cost")


class ChatResponse(BaseModel):
    conversation_id: int
    answer: str
    refused: bool
    reason: str
    citations: list[Citation]
    usage: Usage
    seconds: float
    query: str = Field(description="the standalone query the retriever saw")
    cached: bool = Field(description="served from the response cache, at no cost")
    window: WindowInfo
    sources: list[Source] = Field(
        default=[], description="every passage retrieved for the answer (Lesson 8)")
    trace_id: str | None = Field(
        default=None, description="this turn's trace in Langfuse, if tracing is on (Lesson 8)")


# ---- the streaming vocabulary: one model per SSE event name ------------------
class DeltaEvent(BaseModel):
    """event: delta. A fragment of the answer, to append to what is on screen."""
    text: str


class CitationsEvent(BaseModel):
    """event: citations. Sent once, after the last delta."""
    citations: list[Citation]
    refused: bool
    reason: str


class DoneEvent(BaseModel):
    """event: done. The turn is saved; the stream ends after this.

    Lesson 4 added `query`, `cached` and `window`, and Lesson 8 `trace_id`. A client
    written against Lesson 3 ignores them, which is how a vocabulary grows without
    breaking anyone."""
    conversation_id: int
    seconds: float
    query: str = ""
    cached: bool = False
    window: WindowInfo | None = None
    trace_id: str | None = None


class ErrorEvent(BaseModel):
    """event: error. Something failed after the 200 was sent. The stream ends."""
    message: str


# ---- conversations -----------------------------------------------------------
class ConversationSummary(BaseModel):
    id: int
    user_id: str | None
    title: str
    created_at: datetime
    messages: int
    usd: float


class MessageOut(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation]
    refused: bool
    model: str | None
    n_in: int
    n_out: int
    usd: float
    created_at: datetime


class ConversationOut(BaseModel):
    id: int
    user_id: str | None
    title: str
    created_at: datetime
    messages: list[MessageOut]


# ---- budgets (Lesson 4) -------------------------------------------------------
class BudgetOut(BaseModel):
    user_id: str
    limit: int = Field(description="tokens in + out allowed per window")
    used: int
    remaining: int
    window_hours: int
    reset_seconds: float = Field(description="until the oldest counted turn leaves the window")


# ---- the service -------------------------------------------------------------
class Health(BaseModel):
    status: Literal["ok"]
    pid: int = Field(description="the server's process ID, so a script can stop it")
    model: str
    fallback_model: str | None = Field(description="tried when the primary fails")
    prompt_version: str
    corpus: str = Field(description="fingerprint of what the retriever can find")
    guard_context: bool
    rerank: bool = Field(description="the cross-encoder and its refusal gate are on (Lesson 8)")
    tracing: str | None = Field(description="where traces go, or nothing (Lesson 8)")
    documents: dict[str, int] = Field(description="document count by status")
    chunks: int = Field(description="chunks the retriever is searching right now")
    points: int = Field(description="vectors in Qdrant")
    startup_seconds: dict[str, float] = Field(
        description="what the server paid once, at startup, so no request pays it")
