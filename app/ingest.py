"""Entry point: python -m app.ingest

The pipeline runs from a terminal, with no Jupyter anywhere in the picture.
That is most of what Lesson 1 is about: a notebook cannot be started by cron,
by a Dockerfile, or by a colleague who does not have your kernel state.
"""

import argparse
import json

from app.ingestion import ingest, load_manifest
from app.logs import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse the corpus and build the index.")
    ap.add_argument("--only", nargs="*", metavar="DOC",
                    help="document slugs to ingest, default all of them")
    args = ap.parse_args()

    setup_logging()
    docs = load_manifest()
    if args.only:
        docs = {k: v for k, v in docs.items() if k in set(args.only)}
    print(json.dumps(ingest(docs), indent=2))


if __name__ == "__main__":
    main()
