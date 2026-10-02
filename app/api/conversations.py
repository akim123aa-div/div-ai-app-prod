"""/conversations: what was said, read back from Postgres.

Read-only. Conversations are written by the chat endpoints, one turn per
request, and this router only lists and fetches them. Lesson 5's sidebar is a
call to the first endpoint, and opening a conversation is a call to the second.

Both answer for the user named in `X-User-Id`, with the rule `/chat` already
applies: someone else's conversation is a 404, the same as one that does not
exist. Conversations from before Lesson 4 have no owner, so they can still be
opened by ID and are in nobody's list. This was Lesson 4's first exercise.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app import db
from app.api.chat import user
from app.api.schemas import ConversationOut, ConversationSummary

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationSummary])
def list_conversations(user_id: Annotated[str, Depends(user)],
                       limit: Annotated[int, Query(ge=1, le=100)] = 20) -> list[dict]:
    """The caller's conversations, newest first."""
    return db.list_conversations(limit, user_id=user_id)


@router.get("/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: int,
                     user_id: Annotated[str, Depends(user)]) -> dict:
    """Every turn, with the citations and cost stored at the time."""
    conv = db.get_conversation(conversation_id)
    if conv is None or conv["user_id"] not in (None, user_id):
        raise HTTPException(404, f"no conversation {conversation_id}")
    return conv
