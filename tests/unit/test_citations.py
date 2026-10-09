"""The citation parser: markers the model wrote, turned into sources it could not invent."""

from app.generation import REFUSAL, finish, resolve_citations
from app.llm import Completion
from app.retrieval import Hit


def hit(doc: str, page: int) -> Hit:
    chunk = {"id": f"{doc}#p{page}#0", "doc": doc, "page": page, "title": doc.title(), "text": "..."}
    return Hit(chunk=chunk, score=0.9, fused=0.03)


HITS = [hit("albany", 85), hit("aurora", 9), hit("kelly", 12)]


def test_markers_map_to_their_blocks():
    cites = resolve_citations("The rate was 26.9% [1]. Patents: 1300 [2].", HITS)
    assert [(c["marker"], c["doc"], c["page"]) for c in cites] == [(1, "albany", 85), (2, "aurora", 9)]


def test_invented_and_repeated_markers_are_dropped():
    cites = resolve_citations("A [3]. B [3]. C [7]. D [0].", HITS)
    assert [c["marker"] for c in cites] == [3]  # [7] and [0] point at no block


def test_a_refusal_carries_no_citations():
    c = Completion(text=f"{REFUSAL} [1]", finish="stop", model="m", n_in=1, n_out=1,
                   seconds=0, usd=0)
    a = finish("q", HITS, c, t0=0)
    assert a.refused and a.citations == []
