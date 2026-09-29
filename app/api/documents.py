"""/documents: the job pattern.

Parsing and embedding a PDF takes tens of seconds on a CPU, longer than a client,
a proxy or a user will hold a request open. So an upload is not processed in the
request. The handler does only the cheap, certain things: check the file looks
like a PDF, save it, write a `pending` row. It answers 202 Accepted with the
document ID, and the work runs after the response has gone, as a background task.
The client polls `GET /documents/{id}` until the status is `ready` or `failed`.

Two kinds of failure, and they arrive differently. A file that is plainly wrong,
not a PDF or too big, is refused in the request with a 4xx, because the handler
can tell at once. A file that looks right and breaks the parser can only fail in
the job, so it becomes `status: failed` with the reason in `error`.

Background tasks run inside the API process. A restart kills a job in flight,
which is why startup marks leftover `pending` rows failed. A task queue with its
own workers is the production version, and the same pattern.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

from fastapi import (APIRouter, BackgroundTasks, File, Form, HTTPException, Response,
                     UploadFile)

from app import db, index
from app.api.schemas import DocumentOut
from app.config import settings
from app.ingestion import ingest_upload, slugify
from app.retrieval import refresh_retriever

router = APIRouter(prefix="/documents", tags=["documents"])


def _found(doc: str) -> dict:
    d = db.get_document(doc)
    if d is None:
        raise HTTPException(404, f"no document {doc!r}")
    return d


@router.post("", status_code=202, response_model=DocumentOut)
def upload(file: Annotated[UploadFile, File(description="a PDF")],
           background: BackgroundTasks, response: Response,
           title: Annotated[str | None, Form()] = None) -> dict:
    """Accept a PDF and ingest it in the background. Poll the returned document."""
    if file.file.read(5) != b"%PDF-":
        raise HTTPException(415, "not a PDF: the file does not start with %PDF-")
    file.file.seek(0)

    doc = slugify(file.filename or "document")
    if (d := db.get_document(doc)) and d["status"] == "pending":
        raise HTTPException(409, f"{doc!r} is already being ingested")

    settings.upload_path.mkdir(parents=True, exist_ok=True)
    dest = settings.upload_path / f"{doc}.pdf"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    if dest.stat().st_size > settings.max_upload_mb * 2**20:
        dest.unlink()
        raise HTTPException(413, f"larger than {settings.max_upload_mb} MB")

    title = title or Path(file.filename or doc).stem
    db.start_document(doc, title=title, filename=file.filename or dest.name)
    background.add_task(ingest_upload, doc, title, dest)    # runs after the response
    response.headers["Location"] = f"/documents/{doc}"
    return _found(doc)


@router.get("", response_model=list[DocumentOut])
def list_documents() -> list[dict]:
    return db.list_documents()


@router.get("/{doc}", response_model=DocumentOut)
def get_document(doc: str) -> dict:
    """The thing a client polls after an upload."""
    return _found(doc)


@router.delete("/{doc}", status_code=204)
def delete_document(doc: str) -> None:
    """Remove a document from both stores and from the running retriever."""
    if _found(doc)["status"] == "pending":
        raise HTTPException(409, f"{doc!r} is being ingested; wait for it to finish")
    db.delete_document(doc)
    index.delete_document(doc)
    (settings.upload_path / f"{doc}.pdf").unlink(missing_ok=True)
    refresh_retriever()
