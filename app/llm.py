"""The GenAI Lesson 7 client, moved out of `course_utils.py` and into a module.

Same three ideas it always had: one interface over several providers, retries
that know which failures are worth retrying, and per-call accounting. What is
new is only where it lives. `import course_utils` worked because the file sat
next to the notebook; `from app.llm import chat` works from anywhere, including
from a container, a test, and the API's request handlers.

Lesson 4 adds a fallback provider here. Lesson 7 points `base_url` at Ollama.
Neither of those is a rewrite, which is the reason the transport is one module.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field

import requests

from app.config import settings
from app.logs import get_logger

log = get_logger(__name__)

# Dollars per million tokens. Wrong prices are worse than no prices, so an
# unknown model costs zero and the caller sees a zero rather than a fiction.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
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
        return c

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
        yield c


_client: LLMClient | None = None


def get_client() -> LLMClient:
    """One client per process, because a new one means a new connection pool."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def chat(messages, **kw) -> str:
    """One call, one string back. The common case, spelled the short way."""
    return get_client().complete(messages, **kw).text
