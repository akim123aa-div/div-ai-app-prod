"""Run the golden set against the pipeline and write a scorecard.

    python scripts/eval_golden.py --label m9-l1                              # in this process
    python scripts/eval_golden.py --label m9-l8 --api http://localhost:8000  # through the API

This is the GenAI Lesson 17 evaluation, moved out of a notebook and into a file
that anything can run: you, a terminal, Lesson 8's `pytest`, a CI job you write
yourself. That move is the whole point. A number that lives in a cell is a number
nobody checks after the cell scrolls off the screen.

The scorecard it writes is the baseline for the rest of the module. Every later
lesson changes something -- a database, an API, a container, a different model --
and the question every time is whether these four numbers moved.

Until Lesson 8 it called `answer_question` in its own process, so it measured the
pipeline and nothing around it. `--api` asks the running service instead, by
`POST /chat`, which is what a user's question goes through: the conversation, the
cache, the fallback, the configuration the server was started with. Each question
is a new conversation from a user of its own, `golden-<label>-<id>`, so no memory
carries from one question to the next and no user's daily budget runs out halfway.
Each row keeps the conversation and the trace it made, which is how a wrong answer
in a scorecard becomes a trace you can read.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import requests

from app.config import ROOT
from app.generation import answer_question
from app.logs import get_logger, setup_logging

log = get_logger("eval")

GOLDEN = ROOT / "data" / "golden" / "golden_set.json"
RUNS = ROOT / "data" / "runs"


def accepted(text: str, accept: list[str]) -> bool:
    """The grading rule the golden set states: a substring match after lowercasing
    and removing commas. Crude, checkable, and it does not cost an LLM call."""
    t = text.lower().replace(",", "")
    return any(a.lower().replace(",", "") in t for a in accept)


def ask_here(q: dict) -> dict:
    """One question through the pipeline, in this process."""
    a = answer_question(q["q"])
    return {"answer": a.text, "refused": a.refused, "sources": a.sources,
            "citations": len(a.citations), "seconds": a.seconds, "usd": a.usd}


def ask_api(q: dict, api: str, label: str) -> dict:
    """One question through the running API, as a new conversation."""
    r = requests.post(f"{api}/chat", json={"question": q["q"]}, timeout=600,
                      headers={"X-User-Id": f"golden-{label}-{q['id']}"})
    r.raise_for_status()
    a = r.json()
    return {"answer": a["answer"], "refused": a["refused"], "sources": a["sources"],
            "citations": len(a["citations"]), "seconds": a["seconds"], "usd": a["usage"]["usd"],
            "cached": a["cached"], "conversation_id": a["conversation_id"],
            "trace_id": a["trace_id"]}


def grade(q: dict, a: dict) -> dict:
    hit = any(s["doc"] == q["doc"] and s["page"] in q["pages"] for s in a["sources"])
    return {
        "id": q["id"], "question": q["q"], "answerable": q["answerable"],
        "expected": q["answer"], "answer": a["answer"], "refused": a["refused"],
        "hit@5": hit if q["answerable"] else None,
        "correct": accepted(a["answer"], q["accept"]) if q["answerable"] else None,
        "false_answer": (not a["refused"]) if not q["answerable"] else None,
        "citations": a["citations"], "best_score": round(
            max((s["score"] for s in a["sources"]), default=0.0), 3),
        "seconds": round(a["seconds"], 2), "usd": round(a["usd"], 6),
        **{k: a[k] for k in ("cached", "conversation_id", "trace_id") if k in a},
    }


def scorecard(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["answerable"]]
    no = [r for r in rows if not r["answerable"]]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0
    card = {
        "hit@5": mean([r["hit@5"] for r in ok]),
        "answer accuracy": mean([r["correct"] for r in ok]),
        "false answers": mean([r["false_answer"] for r in no]),
        "answers carrying a citation": mean([r["citations"] > 0 for r in ok]),
        "refusals on answerable": mean([r["refused"] for r in ok]),
        "median seconds": round(sorted(r["seconds"] for r in rows)[len(rows) // 2], 2),
        "usd": round(sum(r["usd"] for r in rows), 4),
    }
    if "cached" in rows[0]:
        card["served from the cache"] = mean([r["cached"] for r in rows])
    return card


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the pipeline on the golden set.")
    ap.add_argument("--label", default="run", help="what to call this run")
    ap.add_argument("--api", default=None, help="ask this running API instead, e.g. http://localhost:8000")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N questions only")
    args = ap.parse_args()

    setup_logging("WARNING")        # one line per question is enough output
    questions = json.loads(GOLDEN.read_text())["questions"][: args.limit]

    if args.api:
        requests.get(f"{args.api}/health", timeout=10).raise_for_status()
        ask = partial(ask_api, api=args.api.rstrip("/"), label=args.label)
    else:
        from app.retrieval import get_retriever
        get_retriever()             # build the index before timing anything
        ask = ask_here
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda q: grade(q, ask(q)), questions))
    card = scorecard(rows)
    card["wall seconds"] = round(time.perf_counter() - t0, 1)
    if not args.api:
        from app import tracing
        tracing.flush()             # in-process answers are traced too; send them first

    RUNS.mkdir(parents=True, exist_ok=True)
    out = {"label": args.label, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "via": args.api or "in-process", "questions": len(rows), "scorecard": card, "rows": rows}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    (RUNS / f"{args.label}-{stamp}.json").write_text(json.dumps(out, indent=2))
    (RUNS / "latest.json").write_text(json.dumps(out, indent=2))

    width = max(len(k) for k in card)
    print(f"\n{args.label}: {len(rows)} questions in {card['wall seconds']}s, via {out['via']}\n")
    for k, v in card.items():
        print(f"  {k:<{width}}  {v}")
    print(f"\nwritten: {RUNS / 'latest.json'}")


if __name__ == "__main__":
    main()
