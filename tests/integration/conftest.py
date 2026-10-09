"""The running stack, for the integration test: the real API, a fake model.

    docker compose up -d postgres qdrant       # what this needs running first
    pytest tests/integration                   # about 30 seconds, most of it the API's startup

Two session fixtures. `fake_llm` starts the fake model in a thread of this process.
`api` starts `python -m app.serve` as a process of its own, on a free port, with
the environment changed in exactly the places Lesson 7 changed it for Ollama:

    OPENAI_BASE      the fake's URL instead of OpenAI's
    OPENAI_API_KEY   any word; the fake reads none
    GEN_MODEL        a name the price table does not know, so every answer costs $0
    FALLBACK_API_KEY empty: no fallback, so a failure here is a failure
    CACHE_ANSWERS    off: every question goes through the whole pipeline

Nothing in `app/` knows it is being tested. The database and the vector index are
the real ones, the same Postgres and Qdrant the app uses. That is the point of an
integration test, and its cost: the test cleans up the document it uploads, and a
production suite would run against databases of its own.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

import httpx
import pytest

from app.config import ROOT, settings
from fake_llm import FakeLLM


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def fake_llm():
    with FakeLLM(free_port()) as fake:
        yield fake


@pytest.fixture(scope="session")
def api(fake_llm, tmp_path_factory):
    for name, url in (("postgres", settings.database_url), ("qdrant", settings.qdrant_url)):
        u = urlsplit(url)
        try:
            socket.create_connection((u.hostname, u.port), timeout=2).close()
        except OSError:
            pytest.fail(f"{name} is not answering on {u.hostname}:{u.port}. "
                        f"Start it: docker compose up -d postgres qdrant")

    port = free_port()
    env = {**os.environ, "OPENAI_BASE": fake_llm.url, "OPENAI_API_KEY": "fake",
           "GEN_MODEL": "fake-llm", "FALLBACK_API_KEY": "", "CACHE_ANSWERS": "false",
           "API_HOST": "127.0.0.1", "API_PORT": str(port)}
    log = tmp_path_factory.mktemp("api") / "api.log"
    with log.open("w") as out:
        proc = subprocess.Popen([sys.executable, "-m", "app.serve"], cwd=ROOT, env=env,
                                stdout=out, stderr=subprocess.STDOUT)
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60)
    deadline = time.monotonic() + 180
    while True:                                 # the models load before /health answers
        try:
            if client.get("/health").status_code == 200:
                break
        except httpx.TransportError:
            pass
        if proc.poll() is not None or time.monotonic() > deadline:
            proc.kill()
            pytest.fail(f"the API did not start:\n{log.read_text()[-3000:]}")
        time.sleep(0.5)
    yield client
    proc.terminate()
    proc.wait(timeout=30)
