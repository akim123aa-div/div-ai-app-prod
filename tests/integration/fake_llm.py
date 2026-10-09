"""A fake language model behind the same protocol as the real one.

The API reaches its model through `LLMClient`, which posts JSON to
`{OPENAI_BASE}/chat/completions` and reads JSON or server-sent events back. It
has never imported a vendor's SDK (GenAI Lesson 7), so anything that answers that
one route is a model as far as the application knows. Lesson 7 pointed the base
URL at Ollama; the integration test points it here.

This model reads no weights. It answers with the first sentence of context block
[1] and a citation marker, which is what a perfectly obedient model would do on
an easy question. That makes the test free, fast, and the same every time, and it
tests everything around the model: upload, ingestion, retrieval, the prompt, the
citation parser, the stream, the database. What it cannot test is whether a real
model obeys the prompt. That is the golden set's job.

Every request it receives is kept in `calls`, so a test can read the prompt the
API built.
"""

from __future__ import annotations

import json
import re
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

BLOCK_1 = re.compile(r'<document index="1"[^>]*>\n(.*?)\n</document>|\[1\] [^\n]*\n(.*?)(?:\n\n\[2\]|\Z)', re.S)


def reply_to(messages: list[dict]) -> str:
    """The first sentence of block [1], cited. No context, or no block: a refusal."""
    m = BLOCK_1.search(messages[-1]["content"])
    if not m:
        return "NOT_IN_CONTEXT"
    block = " ".join((m.group(1) or m.group(2)).split())
    return f"{re.split(r'(?<=[.!?]) ', block)[0].rstrip('.')} [1]."


def make_app(calls: list[dict]) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        body = await request.json()
        calls.append(body)
        text = reply_to(body["messages"])
        usage = {"prompt_tokens": sum(len(m["content"]) for m in body["messages"]) // 4,
                 "completion_tokens": len(text.split())}
        if not body.get("stream"):
            return JSONResponse({"model": body["model"], "usage": usage, "choices": [
                {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})

        def events():
            for word in text.split(" "):
                delta = {"choices": [{"delta": {"content": word + " "}, "finish_reason": None}]}
                yield f"data: {json.dumps({'model': body['model'], **delta})}\n\n"
            yield f"data: {json.dumps({'model': body['model'], 'choices': [], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    return app


class FakeLLM:
    """The fake, running in a thread of the test process on `port`."""

    def __init__(self, port: int):
        self.calls: list[dict] = []
        self.url = f"http://127.0.0.1:{port}/v1"
        self.server = uvicorn.Server(uvicorn.Config(
            make_app(self.calls), host="127.0.0.1", port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> FakeLLM:
        self.thread.start()
        while not self.server.started:
            time.sleep(0.05)
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)
