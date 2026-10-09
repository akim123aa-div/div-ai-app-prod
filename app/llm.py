"""The GenAI Lesson 7 client, moved out of `course_utils.py` and into a module.

Same three ideas it always had: one interface over several providers, retries
that know which failures are worth retrying, and per-call accounting. What is
new is only where it lives. `import course_utils` worked because the file sat
next to the notebook; `from app.llm import chat` works from anywhere, including
from a container, a test, and the API's request handlers.

Lesson 4 adds a fallback provider here: `FallbackClient` holds two clients and
offers the same two methods, so nothing that calls `get_client()` changed. Lesson 7
points the fallback's `base_url` at Ollama. Neither is a rewrite, which is the
reason the transport is one module.

Lesson 8 traces every call: each one is a `generation` in the request's trace,
holding the whole prompt, the reply, the model, the tokens and the cost. A call
that fails is marked as an error, so a fallback shows up as a red step followed
by the one that answered.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field

import requests

from app import tracing
from app.config import settings
from app.logs import get_logger
from app.tracing import observe

log = get_logger(__name__)

# Dollars per million tokens. Wrong prices are worse than no prices, so an
# unknown model costs zero and the caller sees a zero rather than a fiction.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
NEVER_RETRY_CODES = {
    "insufficient_quota", "billing_hard_limit_reached",
    "context_length_exceeded", "invalid_api_key", "model_not_found",
}


class LLMError(Exception):
    """Base class, so a caller can catch one thing."""


class LLMRetryable(LLMError):
    """429, 5xx, timeouts. Trying again may work."""


class LLMFatal(LLMError):
    """400, 401, 403, 404, quota. Trying again will not."""


def cost_of(model: str, n_in: int, n_out: int) -> float:
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    return n_in / 1e6 * p_in + n_out / 1e6 * p_out


def should_retry(status: int, code: str | None = None) -> bool:
    """The whole retry decision in one function you can unit-test. Lesson 8 does.

    Two 429s can want opposite answers and the difference is one string: a rate
    limit clears, an exhausted quota does not.
    """
    if code in NEVER_RETRY_CODES:
        return False
    return status in RETRYABLE_STATUS


@dataclass
class Completion:
    """One normalized result, whichever provider produced it."""

    text: str
    finish: str
    model: str
    n_in: int
    n_out: int
    seconds: float
    usd: float
    fallback: bool = False          # written by the fallback provider (Lesson 4)
    raw: dict = field(repr=False, default_factory=dict)


class LLMClient:
    """Chat and streaming chat over any OpenAI-shaped endpoint."""

    def __init__(self, base_url=None, api_key=None, model=None,
                 timeout=(5.0, 90.0), max_retries=4):
        self.base_url = (base_url or settings.openai_base).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.model = model or settings.gen_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.mount("https://", requests.adapters.HTTPAdapter(
            pool_connections=16, pool_maxsize=16))
        self.usd = 0.0          # what this client has spent since it was built

    def __repr__(self):
        return f"<LLMClient {self.model} @ {self.base_url}>"

    # ---- transport ------------------------------------------------------
    def _post(self, body: dict, stream: bool = False):
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for i in range(self.max_retries + 1):
            try:
                r = self.session.post(url, headers=headers, json=body,
                                      timeout=self.timeout, stream=stream)
                if r.status_code == 200:
                    return r
                try:
                    code = (r.json().get("error") or {}).get("code")
                except ValueError:
                    code = None
                msg = f"HTTP {r.status_code} {code or ''}: {r.text[:200]}"
                if not should_retry(r.status_code, code):
                    raise LLMFatal(msg)
                if i == self.max_retries:
                    raise LLMRetryable(msg)
                delay = float(r.headers.get("Retry-After")
                              or random.uniform(0, min(30.0, 2.0 ** i)))
            except (requests.ConnectionError, requests.Timeout) as e:
                if i == self.max_retries:
                    raise LLMRetryable(f"{type(e).__name__}: {e}") from e
                delay = random.uniform(0, min(30.0, 2.0 ** i))
            log.warning("retrying in %.1fs (attempt %d)", delay, i + 1)
            time.sleep(delay)
        raise LLMRetryable("retries exhausted")     # unreachable, kept honest

    # ---- the two public methods -----------------------------------------
    @observe(name="llm", as_type="generation", capture_input=False, capture_output=False)
    def complete(self, messages, max_tokens=300, temperature=0.0, **extra) -> Completion:
        if not self.api_key:
            raise LLMFatal("no API key; copy .env.example to .env and fill it in")
        t0 = time.perf_counter()
        d = self._post({"model": self.model, "messages": messages,
                        "max_tokens": max_tokens, "temperature": temperature,
                        **extra}).json()
        ch, u = d["choices"][0], d.get("usage") or {}
        n_in, n_out = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        usd = cost_of(self.model, n_in, n_out)
        self.usd += usd
        c = Completion(text=ch["message"].get("content") or "",
                       finish=ch.get("finish_reason", "other"),
                       model=d.get("model", self.model), n_in=n_in, n_out=n_out,
                       seconds=time.perf_counter() - t0, usd=usd, raw=d)
        log.info("llm %s  %d in / %d out  %.2fs  $%.5f",
                 c.model, c.n_in, c.n_out, c.seconds, c.usd)
        tracing.generation(messages=messages, text=c.text, model=c.model, n_in=n_in,
                           n_out=n_out, usd=usd, max_tokens=max_tokens, temperature=temperature)
        return c

    @observe(name="llm", as_type="generation", capture_input=False, capture_output=False)
    def stream(self, messages, max_tokens=300, temperature=0.0, **extra):
        """Yield text fragments as they arrive, then one `Completion` for the whole.

        The last item is the accounting: tokens and cost only exist once the
        provider sends its final usage event, after the last fragment. Callers
        loop, and treat a `Completion` as the end:

            for x in client.stream(messages):
                if isinstance(x, Completion): done = x
                else: print(x, end="")

        The API's streaming chat endpoint puts this behind HTTP.
        """
        if not self.api_key:
            raise LLMFatal("no API key; copy .env.example to .env and fill it in")
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True}, **extra}
        t0, n_in, n_out, model, finish = time.perf_counter(), 0, 0, self.model, "other"
        parts: list[str] = []
        r = self._post(body, stream=True)
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue                              # comments and keep-alives
            payload = line[6:]
            if payload == "[DONE]":                   # a sentinel, not JSON
                break
            ev = json.loads(payload)
            model = ev.get("model") or model
            if ev.get("usage"):
                n_in = ev["usage"]["prompt_tokens"]
                n_out = ev["usage"]["completion_tokens"]
            for ch in ev.get("choices", []):
                finish = ch.get("finish_reason") or finish
                piece = ch.get("delta", {}).get("content")
                if piece:
                    parts.append(piece)
                    yield piece
        usd = cost_of(self.model, n_in, n_out)
        self.usd += usd
        c = Completion(text="".join(parts), finish=finish, model=model, n_in=n_in,
                       n_out=n_out, seconds=time.perf_counter() - t0, usd=usd)
        log.info("llm %s stream  %d in / %d out  %.2fs  $%.5f",
                 c.model, c.n_in, c.n_out, c.seconds, c.usd)
        tracing.generation(messages=messages, text=c.text, model=c.model, n_in=n_in,
                           n_out=n_out, usd=usd, max_tokens=max_tokens, temperature=temperature)
        yield c


class FallbackClient:
    """Two clients behind the interface of one. The second runs when the first fails.

    Retries and a fallback answer different failures. A retry waits out a blip:
    one 503, one dropped connection. A fallback routes around an outage, where
    waiting longer only makes the user wait longer. So the primary is built with
    one retry and a short timeout (`PRIMARY_RETRIES`, `PRIMARY_TIMEOUT`), and any
    `LLMError` it still raises sends the call to the secondary. That includes the
    fatal ones: a revoked key or an exhausted quota belongs to one provider's
    account, and the other provider has its own.

    A stream can only fall back before its first fragment. After that the user
    has read half an answer from one model, and the other would start again.
    """

    def __init__(self, primary: LLMClient, secondary: LLMClient):
        self.primary, self.secondary = primary, secondary
        self.model = primary.model              # the model the application asks for
        self.fallbacks = 0                      # how often the secondary has answered

    def __repr__(self):
        return f"<FallbackClient {self.primary!r} -> {self.secondary!r}>"

    def _falling_back(self, e: LLMError) -> None:
        self.fallbacks += 1
        log.warning("primary %s failed (%s); falling back to %s",
                    self.primary.model, str(e)[:120], self.secondary.model)
        tracing.output(level="WARNING", status=f"fell back to {self.secondary.model}")

    def complete(self, messages, **kw) -> Completion:
        try:
            return self.primary.complete(messages, **kw)
        except LLMError as e:
            self._falling_back(e)
        c = self.secondary.complete(messages, **kw)
        c.fallback = True
        return c

    def stream(self, messages, **kw):
        started = False
        try:
            for x in self.primary.stream(messages, **kw):
                started = True
                yield x
            return
        except LLMError as e:
            if started:                         # half an answer is out; nothing to do
                raise
            self._falling_back(e)
        for x in self.secondary.stream(messages, **kw):
            if isinstance(x, Completion):
                x.fallback = True
            yield x


_client: LLMClient | FallbackClient | None = None


def build_client() -> LLMClient | FallbackClient:
    """The primary from `.env`, wrapped in a fallback if one is configured."""
    if not (settings.fallback_model and settings.fallback_api_key):
        return LLMClient()
    primary = LLMClient(timeout=(3.0, settings.primary_timeout),
                        max_retries=settings.primary_retries)
    # The secondary is the last resort, so it gets a long read timeout and one retry.
    # Retrying a model that is slow rather than down only queues a second copy of
    # the same prompt behind the first.
    secondary = LLMClient(base_url=settings.fallback_base,
                          api_key=settings.fallback_api_key, model=settings.fallback_model,
                          timeout=(5.0, settings.fallback_timeout), max_retries=1)
    return FallbackClient(primary, secondary)


def get_client() -> LLMClient | FallbackClient:
    """One client per process, because a new one means a new connection pool."""
    global _client
    if _client is None:
        _client = build_client()
    return _client


def chat(messages, **kw) -> str:
    """One call, one string back. The common case, spelled the short way."""
    return get_client().complete(messages, **kw).text
