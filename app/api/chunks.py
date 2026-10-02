"""/chunks: the passage behind a citation. Lesson 5.

A citation names a document, a page and a chunk ID. That is enough for code, and
not enough for a reader, who wants to see the words the answer leaned on. The
sources panel fetches them here, one chunk per citation, when it draws them.

Why a route and not a `text` field on every citation: a message stores its
citations as a copy (see `db.Message`), and a copy of five 400-token passages on
every answer would make the messages table mostly duplicated chunks. The cost of
the route is that the passage is read now, not when the answer was written. If
the document has been deleted since, the citation still says where the answer
came from, and this answers 404. If it has been uploaded again, the same ID can
name different text, and the client cannot tell. Storing the passage is the fix
for that, and the price is the duplication.

A chunk ID has a `#` in it, which in a URL starts the fragment, so a client
must percent-encode it: `/chunks/albany%23p7%233`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app import db
from app.api.schemas import ChunkOut

router = APIRouter(prefix="/chunks", tags=["chunks"])


@router.get("/{chunk_id}", response_model=ChunkOut)
def get_chunk(chunk_id: str) -> dict:
    """One chunk's text, page and document. What a citation's `chunk_id` points at."""
    c = db.get_chunk(chunk_id)
    if c is None:
        raise HTTPException(404, f"no chunk {chunk_id!r}")
    return c
