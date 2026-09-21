"""Run the golden set against the pipeline and write a scorecard.

    python scripts/eval_golden.py --label m9-l1

This is the GenAI Lesson 17 evaluation, moved out of a notebook and into a file
that anything can run: you, a terminal, Lesson 8's `pytest`, a CI job you write
yourself. That move is the whole point. A number that lives in a cell is a number
nobody checks after the cell scrolls off the screen.

The scorecard it writes is the baseline for the rest of the module. Every later
lesson changes something -- a database, an API, a container, a different model --
and the question every time is whether these four numbers moved.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from app.config import ROOT
from app.generation import answer_question
from app.logs import get_logger, setup_logging
from app.retrieval import get_retriever

log = get_logger("eval")

GOLDEN = ROOT / "data" / "golden" / "golden_set.json"
RUNS = ROOT / "data" / "runs"


def accepted(text: str, accept: list[str]) -> bool:
    """The grading rule the golden set states: a substring match after lowercasing
    and removing commas. Crude, checkable, and it does not cost an LLM call."""
    t = text.lower().replace(",", "")
    return any(a.lower().replace(",", "") in t for a in accept)


def score_one(q: dict) -> dict:
    a = answer_question(q["q"])
    hit = any(h.chunk["doc"] == q["doc"] and h.chunk["page"] in q["pages"] for h in a.hits)
    return {
        "id": q["id"], "question": q["q"], "answerable": q["answerable"],
        "expected": q["answer"], "answer": a.text, "refused": a.refused,
        "hit@5": hit if q["answerable"] else None,
        "correct": accepted(a.text, q["accept"]) if q["answerable"] else None,
        "false_answer": (not a.refused) if not q["answerable"] else None,
        "citations": len(a.citations), "best_score": round(
            max((h.score for h in a.hits), default=0.0), 3),
        "seconds": round(a.seconds, 2), "usd": round(a.usd, 6),
    }


def scorecard(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["answerable"]]
    no = [r for r in rows if not r["answerable"]]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0
    return {
        "hit@5": mean([r["hit@5"] for r in ok]),
        "answer accuracy": mean([r["correct"] for r in ok]),
        "false answers": mean([r["false_answer"] for r in no]),
        "answers carrying a citation": mean([r["citations"] > 0 for r in ok]),
        "refusals on answerable": mean([r["refused"] for r in ok]),
        "median seconds": round(sorted(r["seconds"] for r in rows)[len(rows) // 2], 2),
        "usd": round(sum(r["usd"] for r in rows), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the pipeline on the golden set.")
    ap.add_argument("--label", default="run", help="what to call this run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N questions only")
    args = ap.parse_args()

    setup_logging("WARNING")        # one line per question is enough output
    questions = json.loads(GOLDEN.read_text())["questions"][: args.limit]

    get_retriever()                 # build the index before timing anything
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(score_one, questions))
    card = scorecard(rows)
    card["wall seconds"] = round(time.perf_counter() - t0, 1)

    RUNS.mkdir(parents=True, exist_ok=True)
    out = {"label": args.label, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "questions": len(rows), "scorecard": card, "rows": rows}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    (RUNS / f"{args.label}-{stamp}.json").write_text(json.dumps(out, indent=2))
    (RUNS / "latest.json").write_text(json.dumps(out, indent=2))

    width = max(len(k) for k in card)
    print(f"\n{args.label}: {len(rows)} questions in {card['wall seconds']}s\n")
    for k, v in card.items():
        print(f"  {k:<{width}}  {v}")
    print(f"\nwritten: {RUNS / 'latest.json'}")


if __name__ == "__main__":
    main()
