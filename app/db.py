"""The relational store, and the only module that writes SQL.

Postgres is the source of truth. Every document and every chunk has a row here,
and the vector index in `index.py` is derived from those rows: drop the Qdrant
collection and `python -m app.reindex` rebuilds it from this database, with no
PDF parsed. The reverse is not true, which is what "source of truth" means.

Four tables, and each one is here because something must survive a restart:

    documents      what was ingested, and whether it worked      pending -> ready | failed
    chunks         the text the retriever searches and cites
    conversations  a thread of turns, owned by a user             (Lesson 4 keys budgets on user_id)
    messages       each turn, with its citations and what it cost

Everything else the application holds is derived and rebuildable: vectors live in
Qdrant, BM25 is rebuilt from `chunks` at startup, models are reloaded from disk.

The API in `app/api/` calls the functions at the bottom of this file and never
writes SQL itself. Swap Postgres for another database and this module
changes; nothing above it does.

`create_all` builds the tables. It creates tables that are missing and never
alters one that exists, so adding a column to a model here does nothing to a
database that already has the table. A real schema change needs a migration tool,
and for SQLAlchemy that is Alembic. This module stops at naming it.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from sqlalchemy import (JSON, Engine, ForeignKey, Text, create_engine, delete, func,
                        select)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from app.config import settings
from app.logs import get_logger

log = get_logger(__name__)


class Base(DeclarativeBase):
    pass


# ---- the schema: one class per table -----------------------------------------
class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(primary_key=True)          # the slug, e.g. "albany"
    title: Mapped[str]
    filename: Mapped[str]
    pages: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(default="ready")        # pending | ready | failed
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True)


class Chunk(Base):
    __tablename__ = "chunks"

    # The same string the chunker made, e.g. "albany#p7#3". It is also what the
    # Qdrant point ID is derived from, so it is the key that joins the two stores.
    id: Mapped[str] = mapped_column(primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page: Mapped[int]
    section: Mapped[str] = mapped_column(default="")
    text: Mapped[str] = mapped_column(Text)
    n_tokens: Mapped[int]

    document: Mapped[Document] = relationship(back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str | None] = mapped_column(index=True)    # a header in Lesson 4, not auth
    title: Mapped[str] = mapped_column(default="")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan",
        passive_deletes=True, order_by="Message.id")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str]                                           # user | assistant
    content: Mapped[str] = mapped_column(Text)
    # Citations are a copy, not a foreign key to `chunks`. A message is a record
    # of what was said; it must survive the document it cites being re-chunked.
    citations: Mapped[list] = mapped_column(JSON, default=list)
    refused: Mapped[bool] = mapped_column(default=False)
    model: Mapped[str | None]
    n_in: Mapped[int] = mapped_column(default=0)
    n_out: Mapped[int] = mapped_column(default=0)
    usd: Mapped[float] = mapped_column(default=0.0)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


# ---- connection --------------------------------------------------------------
@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """One engine per process. It owns a connection pool, so a new one per call
    would open a new connection per call."""
    return create_engine(settings.database_url, pool_pre_ping=True)


def session() -> Session:
    """`with session() as s, s.begin():` commits on success, rolls back on error."""
    return Session(get_engine(), expire_on_commit=False)


def init_db() -> None:
    """Create any missing table. Safe to call on every start; see the module note
    on what it will not do."""
    Base.metadata.create_all(get_engine())


# ---- documents and chunks ----------------------------------------------------
def replace_document(doc: str, title: str, filename: str, pages: int,
                     chunks: list, status: str = "ready") -> int:
    """Write one document and its chunks, replacing any earlier version of it.

    One transaction: a reader sees the old chunks or the new ones, never half.
    `chunks` are `app.chunking.Chunk` objects. An upload writes them as `pending`
    and flips the status once the index has them too; see `ingestion.py`.
    """
    with session() as s, s.begin():
        s.execute(delete(Document).where(Document.id == doc))   # cascades to chunks
        s.add(Document(id=doc, title=title, filename=filename, pages=pages,
                       status=status, chunks=[
                           Chunk(id=c.id, page=c.page, section=c.section,
                                 text=c.text, n_tokens=c.n_tokens) for c in chunks]))
    return len(chunks)


def load_chunks(docs: list[str] | None = None, ready_only: bool = True) -> list[dict]:
    """Chunks with their document's title, in a stable order, as plain dicts.

    Dicts rather than ORM objects, because the retriever holds these for the life
    of the process and should not hold a database session with them.

    Only `ready` documents by default, because that is what the retriever may
    search. The indexer passes `ready_only=False`: it is the step that makes a
    pending document ready, so it has to be able to see one.
    """
    q = (select(Chunk.id, Chunk.document_id, Document.title, Chunk.page,
                Chunk.section, Chunk.text)
         .join(Document).order_by(Chunk.id))
    if ready_only:
        q = q.where(Document.status == "ready")
    if docs:
        q = q.where(Chunk.document_id.in_(docs))
    with session() as s:
        return [{"id": r.id, "doc": r.document_id, "title": r.title, "page": r.page,
                 "section": r.section, "text": r.text} for r in s.execute(q)]


def count_chunks() -> int:
    with session() as s:
        return s.scalar(select(func.count()).select_from(Chunk)) or 0


def _documents_query():
    return (select(Document.id, Document.title, Document.filename, Document.pages,
                   Document.status, Document.error, Document.created_at,
                   func.count(Chunk.id).label("chunks"))
            .outerjoin(Chunk).group_by(Document.id).order_by(Document.id))


def list_documents() -> list[dict]:
    with session() as s:
        return [dict(r._mapping) for r in s.execute(_documents_query())]


def document_counts() -> dict[str, int]:
    with session() as s:
        return dict(s.execute(select(Document.status, func.count())
                              .group_by(Document.status)).all())


def get_document(doc: str) -> dict | None:
    with session() as s:
        r = s.execute(_documents_query().where(Document.id == doc)).first()
        return dict(r._mapping) if r else None


def start_document(doc: str, title: str, filename: str) -> None:
    """The row an upload writes before any work is done: status `pending`.

    This is the first half of the job pattern. The row exists before the work
    does, so a client polling for it finds something at once, and a crash leaves
    evidence rather than nothing. An existing document of the same ID stays
    searchable under its old chunks until the new ones replace it.
    """
    with session() as s, s.begin():
        d = s.get(Document, doc)
        if d is None:
            s.add(Document(id=doc, title=title, filename=filename, status="pending"))
        else:
            d.title, d.filename, d.status, d.error = title, filename, "pending", None


def set_status(doc: str, status: str, error: str | None = None) -> None:
    """Move a document along pending -> ready | failed. A failed document keeps
    its row, with the reason, and loses its chunks: nothing half-built is left
    for the retriever to find."""
    with session() as s, s.begin():
        d = s.get(Document, doc)
        if d is None:
            return
        d.status, d.error = status, error
        if status == "failed":
            s.execute(delete(Chunk).where(Chunk.document_id == doc))


def fail_interrupted() -> int:
    """Mark every `pending` document failed. Called once when the API starts.

    A background task lives inside the API process and dies with it. Any row
    still pending at startup belongs to a job that no process is running, and it
    would say `pending` forever. A task queue keeps the job outside the process,
    so a restart does not lose it; that is the production version of this line.
    """
    with session() as s, s.begin():
        stuck = s.scalars(select(Document).where(Document.status == "pending")).all()
        for d in stuck:
            d.status, d.error = "failed", "interrupted: the server restarted mid-job"
            s.execute(delete(Chunk).where(Chunk.document_id == d.id))
        return len(stuck)


def delete_document(doc: str) -> bool:
    """Remove a document and, through the cascade, its chunks."""
    with session() as s, s.begin():
        return s.execute(delete(Document).where(Document.id == doc)).rowcount > 0


# ---- conversations and messages ----------------------------------------------
def record_turn(question: str, answer: str, *, citations: list[dict],
                refused: bool, model: str | None, n_in: int, n_out: int, usd: float,
                conversation_id: int | None = None, user_id: str | None = None) -> int:
    """Save one question and its answer. Starts a conversation if none is given.

    Returns the conversation ID. The API calls this at the end of every chat
    request, which is what lets a conversation outlive the process that served it.
    """
    with session() as s, s.begin():
        conv = s.get(Conversation, conversation_id) if conversation_id else None
        if conv is None:
            conv = Conversation(user_id=user_id, title=question[:80])
            s.add(conv)
        conv.messages.append(Message(role="user", content=question))
        conv.messages.append(Message(role="assistant", content=answer,
                                     citations=citations, refused=refused, model=model,
                                     n_in=n_in, n_out=n_out, usd=usd))
        s.flush()
        return conv.id


def conversation_exists(conversation_id: int) -> bool:
    with session() as s:
        return s.get(Conversation, conversation_id) is not None


def list_conversations(limit: int = 20) -> list[dict]:
    """Newest first, with a message count and what the conversation has cost."""
    q = (select(Conversation.id, Conversation.user_id, Conversation.title,
                Conversation.created_at, func.count(Message.id).label("messages"),
                func.coalesce(func.sum(Message.usd), 0.0).label("usd"))
         .outerjoin(Message).group_by(Conversation.id)
         .order_by(Conversation.id.desc()).limit(limit))
    with session() as s:
        return [dict(r._mapping) for r in s.execute(q)]


def get_conversation(conversation_id: int) -> dict | None:
    with session() as s:
        conv = s.get(Conversation, conversation_id)
        if conv is None:
            return None
        return {"id": conv.id, "user_id": conv.user_id, "title": conv.title,
                "created_at": conv.created_at,
                "messages": [{"id": m.id, "role": m.role, "content": m.content,
                              "citations": m.citations, "refused": m.refused,
                              "model": m.model, "n_in": m.n_in, "n_out": m.n_out,
                              "usd": m.usd, "created_at": m.created_at}
                             for m in conv.messages]}
