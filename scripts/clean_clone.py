"""The clean-clone test: does the committed repository start from nothing, with one command?

    python scripts/clean_clone.py            # clone, start, upload, ask, tear down
    python scripts/clean_clone.py --keep     # leave the stack running afterwards, to look at

"Works on my machine" passes on your machine because of everything that is on it
and not in the repository: a .venv, a warm model cache, a database that already
holds the corpus, a file you forgot to commit. This script runs on your machine
too, so it removes those things by hand:

    1. git clone      only what is committed travels; uncommitted changes are not tested
    2. copy .env      the one file that is deliberately not in git
    3. compose up     a project of its own, `docchat-clean`, so new volumes, a new network,
                      and an empty database. Host ports are free ones, so your running
                      stack is untouched
    4. upload a PDF   and poll until it is ready
    5. ask about it   the answer must name the fact and carry a citation
    6. the UI         answers on its port, and its page renders without an exception.
                      A health endpoint only proves the server started: the page is a
                      script that runs per visitor, and can fail on its first import
    7. compose down   -v: the volumes go too, and so do the images this project built

The images build from the build cache your own `docker compose build` left, so a
clone of the same code builds in seconds. Delete that cache and it takes minutes,
which is the honest first-time number. The models are downloaded into the new
volume every time, about 200 MB, because a new machine has to do that too.

This is the final project's first requirement, so run it on yours.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
PROJECT = "docchat-clean"
PDF = "data/pdfs/wheeler.pdf"
# Runs the page once, headless, inside the ui container, from outside /app so the
# import path is the one `streamlit run` gives it. Prints the exception, or "ok".
RENDER = ("from streamlit.testing.v1 import AppTest; "
          "at = AppTest.from_file('/app/ui/app.py', default_timeout=30).run(); "
          "print(at.exception[0].value if at.exception else 'ok')")
QUESTION = "What is the trading symbol for the Notes issued by Wheeler Real Estate Investment Trust?"
EXPECT = "whlrl"
T0 = time.perf_counter()


def say(step: str, detail: str = "") -> None:
    print(f"{time.perf_counter() - T0:6.1f}s  {step:<10} {detail}", flush=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(cmd: list[str], cwd: Path, env: dict | None = None) -> str:
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"FAIL: {' '.join(cmd)}\n{p.stdout[-2000:]}{p.stderr[-2000:]}")
    return p.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keep", action="store_true", help="leave the stack running")
    args = ap.parse_args()

    if not (ROOT / ".env").exists():
        raise SystemExit("FAIL: no .env here to copy. cp .env.example .env and fill it in.")
    if run(["git", "status", "--porcelain"], ROOT).strip():
        say("warning", "uncommitted changes in this repository are not part of the test")

    work = Path(tempfile.mkdtemp(prefix="docchat-clean-"))
    repo = work / "reference_app"
    run(["git", "clone", "--quiet", str(ROOT), str(repo)], work)
    shutil.copy(ROOT / ".env", repo / ".env")
    head = run(["git", "log", "-1", "--format=%h %s"], repo).strip()
    say("cloned", f"{head[:70]}  into {repo}")

    # Shell variables win over .env when compose fills in ${...}, so these move the
    # host ports without touching the copied file.
    ports = {k: str(free_port()) for k in
             ("POSTGRES_PORT", "QDRANT_PORT", "QDRANT_GRPC_PORT", "API_PORT", "UI_PORT")}
    env = {**os.environ, **ports}
    compose = ["docker", "compose", "-p", PROJECT]
    ok = False
    try:
        say("up", f"project {PROJECT}, API on :{ports['API_PORT']}, UI on :{ports['UI_PORT']}")
        run(compose + ["up", "-d", "--build", "--wait", "--wait-timeout", "600"], repo, env)
        say("healthy", run(compose + ["ps", "--format", "{{.Service}}"], repo, env).split().__repr__())

        api = httpx.Client(base_url=f"http://localhost:{ports['API_PORT']}", timeout=120)
        h = api.get("/health").json()
        say("health", f"{h['documents'] or 'no'} documents, {h['chunks']} chunks: a fresh, empty stack")

        pdf = repo / PDF
        r = api.post("/documents", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
        r.raise_for_status()
        while (d := api.get(r.headers["location"]).json())["status"] == "pending":
            time.sleep(1)
        say("uploaded", f"{d['id']}: {d['status']}, {d.get('pages')} pages, {d.get('chunks')} chunks")
        if d["status"] != "ready":
            raise SystemExit(f"FAIL: ingestion failed: {d.get('error')}")

        a = api.post("/chat", json={"question": QUESTION}, headers={"X-User-Id": "clean-clone"}).json()
        cited, named = len(a.get("citations", [])), EXPECT in a.get("answer", "").lower()
        say("answered", f"{a.get('answer', '')[:90]!r}")
        say("", f"names {EXPECT.upper()}: {named}; citations: {cited}")

        ui = httpx.get(f"http://localhost:{ports['UI_PORT']}/_stcore/health", timeout=10)
        page = run(compose + ["exec", "-T", "-w", "/", "ui", "python", "-c", RENDER], repo, env)
        page = page.strip().splitlines()[-1] if page.strip() else "no output"
        say("ui", f"/_stcore/health -> {ui.status_code} {ui.text}; the page renders: {page}")
        ok = named and cited > 0 and ui.status_code == 200 and page == "ok"
    finally:
        if args.keep:
            say("kept", f"cd {repo} && docker compose -p {PROJECT} down -v --rmi local  # when done")
        else:
            run(compose + ["down", "-v", "--rmi", "local"], repo, env)
            shutil.rmtree(work, ignore_errors=True)
            say("down", "containers, network, volumes and this project's images removed")
    print(f"\n{'PASS' if ok else 'FAIL'}: fresh clone, copy .env, docker compose up, upload a PDF, "
          f"get a cited answer.  {time.perf_counter() - T0:.0f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
