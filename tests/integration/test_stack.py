"""One test through the running stack: upload a document, ask about it, get a cited answer.

Every layer of Lessons 2 to 5 is on this path: the upload route and its background
job, the chunker, Postgres, the embedder and Qdrant, BM25, the reranker and its gate,
the prompt, the model call, the citation parser, the saved turn, the SSE stream. The
fake model makes it free and repeatable. If any layer breaks, this fails, and the
unit tests say which layer.
"""

import time

from pdf import write_pdf

PAGES = [["Halvard Point Lighthouse",
          "The Halvard Point lighthouse was first lit on 14 March 1871, and its first keeper "
          "was Ingrid Solberg.",
          "The tower is 31 metres tall, and its light can be seen 22 nautical miles out to sea."],
         ["Visiting the lighthouse",
          "The lighthouse has been run by the Northcoast Maritime Trust since it was "
          "automated in 1987."]]
QUESTION = "When was the Halvard Point lighthouse first lit?"


def events(response) -> list[str]:
    """The event names of a server-sent event stream, in order."""
    return [line[len("event: "):] for line in response.iter_lines() if line.startswith("event: ")]


def test_upload_ask_and_get_a_cited_answer(api, fake_llm, tmp_path):
    pdf = write_pdf(tmp_path / "pytest-halvard-point.pdf", PAGES)
    r = api.post("/documents", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
    assert r.status_code == 202
    doc = r.json()["id"]
    try:
        for _ in range(120):                    # the job pattern: poll until it is ready
            d = api.get(f"/documents/{doc}").json()
            if d["status"] != "pending":
                break
            time.sleep(0.5)
        assert d["status"] == "ready", d["error"]

        a = api.post("/chat", json={"question": QUESTION}, headers={"X-User-Id": "pytest"}).json()
        assert not a["refused"], a["reason"]
        assert "14 March 1871" in a["answer"]
        assert a["citations"] and a["citations"][0]["doc"] == doc
        assert a["sources"][0]["doc"] == doc and a["usage"]["usd"] == 0
        prompt = fake_llm.calls[-1]["messages"][-1]["content"]
        assert "first lit on 14 March 1871" in prompt          # retrieval put it in the prompt

        with api.stream("POST", "/chat/stream", json={"question": QUESTION},
                        headers={"X-User-Id": "pytest"}) as s:
            names = events(s)
        assert names[0] == "delta" and names[-3:] == ["citations", "usage", "done"]
    finally:
        api.delete(f"/documents/{doc}")
