"""/conversations: what was said, read back from Postgres.

Read-only. Conversations are written by the chat endpoints, one turn per
request, and this router only lists and fetches them. Lesson 5's sidebar is a
call to the first endpoint, and opening a conversation is a call to the second.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from app import db
from app.api.schemas import ConversationOut, ConversationSummary

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationSummary])
def list_conversations(limit: Annotated[int, Query(ge=1, le=100)] = 20) -> list[dict]:
    """Newest first."""
    return db.list_conversations(limit)


@router.get("/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: int) -> dict:
    """Every turn, with the citations and cost stored at the time."""
    conv = db.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(404, f"no conversation {conversation_id}")
    return conv
