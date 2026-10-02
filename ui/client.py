"""Everything the UI knows about the API, and nothing about the pipeline.

The UI is a client like any other: a notebook, a script, a colleague's React
app. It reaches the service through the routes `/docs` lists and the bodies
`app/api/schemas.py` declares, and through nothing else. That is checkable:
no file under `ui/` imports the `app` package; read the imports at the top of
each. The UI can be restarted, rewritten or deployed on another machine, and the
API does not notice. Lesson 6 gives it a container of its own, with no torch in it.

This file has no Streamlit in it either. Every URL, header and status code the
UI depends on is here, so a script or a terminal chat can call the same functions
the page calls, and a different frontend could start from this file.

Two things a browser's `EventSource` would do for you are done by hand:

    events()     read server-sent events off the wire: `event:` and `data:` lines,
                 a blank line ends one, and a line starting `:` is a keep-alive
    APIError     every way a call can fail, as one exception with a status code,
                 the server's `detail`, and `Retry-After` when there is one

Every call is logged to the terminal running `streamlit run`: method, path,
status and milliseconds. Put that terminal beside the API's and each click on
the page shows up twice, once as a request leaving and once as it arrives.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

# The one setting the UI needs. From the host it is http://localhost:8000; inside
# compose, Lesson 6 sets it to http://api:8000. A value already in the
# environment wins over .env, which is how compose will pass it.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
API_URL = os.environ.get("API_URL", "http://localhost:8000")

log = logging.getLogger("ui")


def setup_logging() -> None:
    """The UI's log, to stderr in the format the API uses. Streamlit configures
    only its own loggers, so this one needs a handler. Safe to call on every rerun."""
    if log.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-16s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    log.propagate = False


class APIError(Exception):
    """A call that did not succeed. `status` is None when the API could not be reached."""

    def __init__(self, status: int | None, detail: str, retry_after: float | None = None):
        super().__init__(detail)
        self.status, self.detail, self.retry_after = status, detail, retry_after


@lru_cache(maxsize=1)
def connection() -> httpx.Client:
    """One connection pool for the whole UI process, shared by every browser tab.

    The read timeout is long because a streamed answer may wait for the primary
    provider to time out and the fallback to start (Lesson 4, Section 5)."""
    return httpx.Client(base_url=API_URL, timeout=httpx.Timeout(10.0, read=120.0))


def failed(r: httpx.Response) -> APIError:
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        detail = r.text
    if not isinstance(detail, str):          # a 422 lists what failed validation
        detail = "; ".join(e.get("msg", str(e)) for e in detail)
    retry = r.headers.get("retry-after")
    return APIError(r.status_code, detail, float(retry) if retry else None)


def events(lines: Iterable[str]) -> Iterator[tuple[str, dict]]:
    """(event name, JSON data) for each server-sent event in a stream of lines."""
    name, data = "message", []
    for line in lines:
        if not line:                         # a blank line: the event is complete
            if data:
                yield name, json.loads("\n".join(data))
            name, data = "message", []
        elif line.startswith(":"):           # a comment: FastAPI's keep-alive ping
            continue
        else:
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                name = value
            elif field == "data":
                data.append(value)


class Client:
    """The API, as one user sees it. `user` goes out as X-User-Id on every call."""

    def __init__(self, user: str):
        self.user = user
        self.http = connection()

    def _call(self, method: str, path: str, **kw) -> httpx.Response:
        t0 = time.perf_counter()
        try:
            r = self.http.request(method, path, headers={"X-User-Id": self.user}, **kw)
        except httpx.TransportError as e:
            log.warning("%s %s -> no answer (%s)", method, path, type(e).__name__)
            raise APIError(None, f"the API at {API_URL} is not answering "
                                 f"({type(e).__name__})") from e
        log.info("%s %s -> %d  %.0f ms", method, path, r.status_code,
                 1000 * (time.perf_counter() - t0))
        if r.is_error:
            raise failed(r)
        return r

    def _get(self, path: str, **kw):
        return self._call("GET", path, **kw).json()

    # ---- the service and the user ------------------------------------------
    def health(self) -> dict:
        return self._get("/health", timeout=3.0)

    def usage(self) -> dict:
        return self._get("/usage")

    # ---- documents: the job pattern from the client's side -------------------
    def documents(self) -> list[dict]:
        return self._get("/documents")

    def document(self, doc: str) -> dict:
        return self._get(f"/documents/{quote(doc, safe='')}")

    def upload(self, filename: str, data: bytes) -> dict:
        """Starts ingestion and returns the `pending` document. Poll `document()`."""
        files = {"file": (filename, data, "application/pdf")}
        return self._call("POST", "/documents", files=files).json()

    # ---- conversations and the passages citations point at -----------------
    def conversations(self, limit: int = 20) -> list[dict]:
        return self._get("/conversations", params={"limit": limit})

    def conversation(self, conversation_id: int) -> dict:
        return self._get(f"/conversations/{conversation_id}")

    def chunk(self, chunk_id: str) -> dict | None:
        """The passage behind a citation, or None if it has left the corpus.
        The `#` in a chunk ID would start a URL fragment, so it is encoded."""
        try:
            return self._get(f"/chunks/{quote(chunk_id, safe='')}")
        except APIError as e:
            if e.status == 404:
                return None
            raise

    # ---- chat ----------------------------------------------------------------
    def ask(self, question: str, conversation_id: int | None = None) -> dict:
        """One turn, answered whole: POST /chat."""
        body = {"question": question, "conversation_id": conversation_id}
        return self._call("POST", "/chat", json=body).json()

    def stream(self, question: str,
               conversation_id: int | None = None) -> Iterator[tuple[str, dict]]:
        """One turn as it is written: POST /chat/stream, yielded as (event, data).

        A refusal by status code (404, 422, 429) arrives before the stream starts
        and is raised as APIError. A failure after that arrives as an `error`
        event, because the 200 has already been sent (Lesson 3)."""
        body = {"question": question, "conversation_id": conversation_id}
        t0, deltas = time.perf_counter(), 0
        try:
            with self.http.stream("POST", "/chat/stream", json=body,
                                  headers={"X-User-Id": self.user}) as r:
                log.info("POST /chat/stream -> %d  headers after %.0f ms", r.status_code,
                         1000 * (time.perf_counter() - t0))
                if r.is_error:
                    r.read()
                    raise failed(r)
                for name, data in events(r.iter_lines()):
                    if name == "delta" and not deltas:
                        log.info("  first words after %.2fs", time.perf_counter() - t0)
                    deltas += name == "delta"
                    if name != "delta":
                        log.info("  event %-9s after %.2fs%s", name, time.perf_counter() - t0,
                                 f" ({deltas} deltas before it)" if name == "citations" else "")
                    yield name, data
        except httpx.TransportError as e:
            raise APIError(None, f"the API at {API_URL} stopped answering "
                                 f"({type(e).__name__})") from e
