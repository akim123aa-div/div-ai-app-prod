"""Print a trace as a tree, from Langfuse's API: one request, step by step.

    python scripts/show_trace.py 4f1c...e2            # one trace, by the ID the API returned
    python scripts/show_trace.py --session 812        # every turn of conversation 812

Langfuse's own page shows the same tree with more on it, at the URL printed at the
bottom. This script exists for the times you want it in a terminal, in a notebook
or in a test, and to show that a trace is data you can query, not only a page you
look at. Lesson 8 reads traces with it.

A trace arrives a few seconds after the request that wrote it: the API sends spans
in batches, and Langfuse's worker writes them to ClickHouse in batches too. So
`load` waits until the tree has a finished root and stops growing.
"""

from __future__ import annotations

import argparse
import json
import time

from app import tracing

FIELDS = "core,basic,usage,io,metadata,model"
# Steps that start in the same millisecond are shown in the order the pipeline runs them.
ORDER = {"window": 0, "condense": 1, "cache": 2, "answer": 3, "retrieve": 4, "rerank": 5}


def _parse(v):
    """Langfuse returns input and output as the JSON strings it stored."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def load(trace_id: str, wait: float = 30.0) -> list[dict]:
    """Every observation in one trace, as dicts. Waits for a late trace to arrive."""
    obs, seen, deadline = [], -1, time.monotonic() + wait
    while time.monotonic() < deadline:
        obs = [o.model_dump() for o in tracing.client.api.observations.get_many(
            trace_id=trace_id, fields=FIELDS, limit=1000).data]
        done = any(o["parent_observation_id"] is None and o["end_time"] for o in obs)
        if done and len(obs) == seen:
            break
        seen = len(obs)
        time.sleep(1.5)
    for o in obs:
        o["input"], o["output"] = _parse(o.get("input")), _parse(o.get("output"))
        o["metadata"] = {k: v for k, v in (o.get("metadata") or {}).items()
                         if not k.startswith(("scope.", "resourceAttributes."))}
    return obs


def session(conversation_id: int | str) -> list[str]:
    """The trace IDs of a conversation's turns, oldest first."""
    roots = tracing.client.api.observations.get_many(
        session_id=str(conversation_id), is_root_observation=True, fields="core", limit=100).data
    return [o.trace_id for o in sorted(roots, key=lambda o: o.start_time)]


def _short(text, n: int = 90) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def describe(o: dict) -> str:
    """One line for one step: what it did, in the terms that step cares about."""
    out, meta, name = o["output"], o["metadata"], o["name"]
    if name == "turn":
        out = out or {}
        off = ", rerank OFF" if str(meta.get("rerank")).lower() == "false" else ""
        return (f"{o.get('user_id') or '?'}, conversation {o.get('session_id') or '?'}{off}"
                f"{', from the cache' if out.get('cached') else ''}: {_short(out.get('answer', ''), 70)!r}")
    if name == "window":
        return f"{len(out or [])} past messages sent back" + (", summarised" if meta.get("summarised") else "")
    if name == "condense":
        return f"{meta['call']}" if meta.get("call") else f"-> {_short(out, 70)!r}"
    if name == "cache":
        return str(out)
    if name == "retrieve":
        hits = ", ".join(f"{h['doc']} p{h['page']} {h['score']:.3f}" for h in out.get("hits", []))
        gate = f"gate {meta['gate']}" if meta.get("gate") is not None else "no gate"
        refused = f"  REFUSED: {out['refused']}" if out.get("refused") else ""
        return f"rerank {'on' if meta.get('rerank') else 'OFF'}, {gate}: {hits}{refused}"
    if name == "answer":
        return _short((out or {}).get("text", ""), 80) if isinstance(out, dict) else ""
    if o["type"] == "GENERATION":
        u, c = o.get("usage_details") or {}, o.get("cost_details") or {}
        return (f"{o.get('model') or '?'}, {u.get('input', 0):,} in / {u.get('output', 0)} out, "
                f"${c.get('total', 0):.5f}")
    return ""


def render(obs: list[dict]) -> str:
    """The trace as an indented tree, children in the order they started."""
    kids: dict[str | None, list[dict]] = {}
    for o in obs:
        kids.setdefault(o["parent_observation_id"], []).append(o)
    lines: list[str] = []

    def walk(parent: str | None, depth: int) -> None:
        for o in sorted(kids.get(parent, []), key=lambda o: (o["start_time"], ORDER.get(o["name"], 9))):
            flag = f"  [{o['level']}: {_short(o['status_message'], 60)}]" if o["level"] not in ("DEFAULT", None) else ""
            lines.append(f"{'  ' * depth}{o['name']:<{12 - 2 * depth if depth < 5 else 2}} "
                         f"{o['latency'] or 0:6.2f}s  {describe(o)}{flag}")
            walk(o["id"], depth + 1)

    walk(None, 0)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Print a trace from Langfuse as a tree.")
    ap.add_argument("trace_id", nargs="?", help="the trace_id a /chat response carried")
    ap.add_argument("--session", help="every turn of this conversation instead")
    args = ap.parse_args()
    if not tracing.enabled:
        raise SystemExit("tracing is off: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env")
    ids = session(args.session) if args.session else [args.trace_id]
    if not ids or ids == [None]:
        raise SystemExit("give a trace ID, or --session with a conversation ID that has traces")
    for tid in ids:
        print(render(load(tid)))
        print(f"  {tracing.url(tid)}\n")


if __name__ == "__main__":
    main()
