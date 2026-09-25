"""Entry point: python -m app.ask "your question"

    $ python -m app.ask "How many patents did Aurora Innovation hold at year end?"

The same pipeline Lesson 3 puts behind HTTP and Lesson 5 puts behind a chat box.
It is worth running it from a terminal once, because everything after this is a
transport wrapped around exactly these three calls.

From Lesson 2 each question and its answer are saved to Postgres as a one-turn
conversation, with the citations, tokens and cost. The process exits; the row
does not. Pass `--conversation ID` to add a turn to an existing one.
"""

import argparse
import json

from app import db
from app.generation import answer_question
from app.logs import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the corpus one question.")
    ap.add_argument("question", help="the question, in quotes")
    ap.add_argument("--k", type=int, default=None, help="context blocks to use")
    ap.add_argument("--json", action="store_true", help="print the whole result as JSON")
    ap.add_argument("--conversation", type=int, default=None, help="add to this conversation")
    ap.add_argument("--no-save", action="store_true", help="do not write to Postgres")
    args = ap.parse_args()

    setup_logging()
    db.init_db()
    a = answer_question(args.question, top_k=args.k)
    conv = None
    if not args.no_save:
        conv = db.record_turn(args.question, a.text, citations=a.citations,
                              refused=a.refused, model=a.model, n_in=a.n_in,
                              n_out=a.n_out, usd=a.usd,
                              conversation_id=args.conversation)

    if args.json:
        print(json.dumps({**a.to_dict(), "conversation_id": conv}, indent=2))
        return

    print(f"\n{a.text}\n")
    if a.refused:
        print(f"refused: {a.reason}")
    for c in a.citations:
        print(f"  [{c['marker']}] {c['source']}  (relevance {c['score']:.2f})")
    print(f"\n{a.seconds:.1f}s, ${a.usd:.5f}"
          + (f", saved as conversation {conv}" if conv else ""))


if __name__ == "__main__":
    main()
