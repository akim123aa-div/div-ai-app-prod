"""Entry point: python -m app.ask "your question"

    $ python -m app.ask "How many patents did Aurora Innovation hold at year end?"

The same pipeline Lesson 3 puts behind HTTP and Lesson 5 puts behind a chat box.
It is worth running it from a terminal once, because everything after this is a
transport wrapped around exactly these three calls.

From Lesson 2 each question and its answer are saved to Postgres as a one-turn
conversation, with the citations, tokens and cost. The process exits; the row
does not. Pass `--conversation ID` to add a turn to an existing one.

From Lesson 4 that turn goes through `answer_turn`, the same function the API
calls, so a follow-up asked here is condensed against the stored history:

    $ python -m app.ask --conversation 76 "And how many did it hold the year before?"

`--no-save` still answers through the plain pipeline, with no memory and no cache.

From Lesson 8 a saved turn prints the link to its trace, when tracing is on.
"""

import argparse
import json

from app import db, tracing
from app.conversation import answer_turn
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
    conv, query, trace = None, args.question, None
    if args.no_save:
        a = answer_question(args.question, top_k=args.k)
    else:
        t = answer_turn(args.question, args.conversation, top_k=args.k)
        a, conv, query, trace = t.answer, t.conversation_id, t.query, t.trace_id
        a.usd = t.usd                       # the whole turn, condensing included
    tracing.flush()                         # the process ends here; send its trace first

    if args.json:
        print(json.dumps({**a.to_dict(), "conversation_id": conv}, indent=2))
        return

    if query != args.question:
        print(f"\nsearched for: {query}")
    print(f"\n{a.text}\n")
    if a.refused:
        print(f"refused: {a.reason}")
    for c in a.citations:
        print(f"  [{c['marker']}] {c['source']}  (relevance {c['score']:.2f})")
    print(f"\n{a.seconds:.1f}s, ${a.usd:.5f}"
          + (f", saved as conversation {conv}" if conv else ""))
    if trace:
        print(f"trace: {tracing.url(trace)}")


if __name__ == "__main__":
    main()
