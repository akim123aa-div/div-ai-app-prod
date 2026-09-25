"""Entry point: python -m app.reindex

Drop the vector index and rebuild it from the `chunks` table. No PDF is opened.

This command is the Lesson 2 claim made executable: Postgres is the source of
truth and Qdrant is derived from it. If this ever needs a PDF to succeed, some
piece of the application's state has been living in the index, where it can be
lost.

Run it after changing EMBED_MODEL, after a Qdrant volume is lost, or after
restoring Postgres from a backup. It re-embeds every chunk, so it takes as long
as the embedding step of `python -m app.ingest`, minus the parsing.
"""

import json

from app.db import count_chunks, init_db
from app.ingestion import rebuild_index
from app.logs import setup_logging


def main() -> None:
    setup_logging()
    init_db()
    if not count_chunks():
        raise SystemExit("no chunks in Postgres; run `python -m app.ingest` first")
    print(json.dumps(rebuild_index(), indent=2))


if __name__ == "__main__":
    main()
