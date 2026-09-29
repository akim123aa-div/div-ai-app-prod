"""Entry point: python -m app.serve

Starts the API with uvicorn on API_HOST:API_PORT from `.env`. The same as

    uvicorn app.api.main:app --host 127.0.0.1 --port 8000

with the host and port read from configuration instead of typed. Lesson 6's
Dockerfile runs this line, with API_HOST set to 0.0.0.0 so that other
containers can reach it.

`--reload` restarts the server when a file under `app/` changes. It is for
development: every restart reloads both models and rebuilds BM25.
"""

import argparse

import uvicorn

from app.config import settings


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the docchat API.")
    ap.add_argument("--reload", action="store_true", help="restart on code changes")
    args = ap.parse_args()
    uvicorn.run("app.api.main:app", host=settings.api_host, port=settings.api_port,
                reload=args.reload, reload_dirs=["app"] if args.reload else None)


if __name__ == "__main__":
    main()
