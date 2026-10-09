"""The FastAPI app: startup, the routers, a health check, and error mapping.

    python -m app.serve                      # or: uvicorn app.api.main:app
    open http://localhost:8000/docs

The Deep Learning Lesson 10 server was one file with a model loaded at import
and a single route. This is the same thing organised: one router per resource,
a schema for every body, and the expensive objects built once in `lifespan`,
before the server accepts its first request.

What `lifespan` warms, and why each one is not left to the first request:

    database     create missing tables; fail jobs a restart interrupted
    embedder     ~130 MB of weights from disk
    reranker     ~90 MB of weights from disk
    retriever    every chunk from Postgres, and BM25 built over them
    first call   torch's own one-off setup, paid on a throwaway input

The seconds each one took are returned by `GET /health`, so the notebook can
show what a per-request load would have cost. On the way out, `lifespan` sends
the traces still buffered (Lesson 8), so the last requests before a restart are
not the ones that go missing.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import db, index, tracing
from app.api import chat, chunks, conversations, documents, usage
from app.api.schemas import Health
from app.config import settings
from app.generation import prompt_version
from app.llm import LLMError, LLMFatal
from app.logs import get_logger, setup_logging
from app.models import embed_query, get_embedder, get_reranker, rerank_scores
from app.retrieval import get_retriever

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    steps = {
        "database": lambda: (db.init_db(), db.fail_interrupted()),
        "embedder": get_embedder,
        "reranker": get_reranker,
        "retriever": get_retriever,
        "first call": lambda: rerank_scores("warm", ["up"]) + embed_query("warm up")[0],
    }
    app.state.startup = {}
    for name, step in steps.items():
        t0 = time.perf_counter()
        step()
        app.state.startup[name] = round(time.perf_counter() - t0, 2)
    log.info("ready in %.1fs: %s", sum(app.state.startup.values()), app.state.startup)
    yield                                    # the server runs here
    log.info("shutting down")
    tracing.flush()


app = FastAPI(
    title="docchat",
    version="m9-l8",
    summary="Grounded answers over uploaded PDFs, with citations. Module 9 reference app.",
    lifespan=lifespan,
)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(usage.router)
app.include_router(chunks.router)


@app.get("/health", response_model=Health, tags=["service"])
def health(request: Request) -> Health:
    """Up, and searching how much. A client's first call, and a container's healthcheck in Lesson 6."""
    return Health(status="ok", pid=os.getpid(), model=settings.gen_model,
                  fallback_model=(settings.fallback_model
                                  if settings.fallback_api_key else None),
                  prompt_version=prompt_version(), corpus=get_retriever().fingerprint,
                  guard_context=settings.guard_context, rerank=settings.rerank,
                  tracing=settings.langfuse_base_url if tracing.enabled else None,
                  documents=db.document_counts(),
                  chunks=len(get_retriever().chunks), points=index.count(),
                  startup_seconds=request.app.state.startup)


@app.exception_handler(LLMError)
def llm_failed(request: Request, e: LLMError) -> JSONResponse:
    """The provider failed after the client's retries. That is not the caller's
    fault, so not a 4xx, and not a bug here, so not a bare 500: a 502 says an
    upstream service failed, and 503 says try again later."""
    status = 502 if isinstance(e, LLMFatal) else 503
    return JSONResponse({"detail": f"model provider: {e}"}, status_code=status)
