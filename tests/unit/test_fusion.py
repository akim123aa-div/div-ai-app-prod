"""Reciprocal rank fusion: ranks combine, raw scores never meet."""

from app.retrieval import Retriever


def test_agreement_beats_one_strong_vote():
    dense = {7: 1, 3: 2}        # chunk position -> rank
    lexical = {9: 1, 3: 2}
    fused = Retriever.fuse(dense, lexical)
    assert max(fused, key=fused.get) == 3       # second twice beats first once


def test_a_chunk_one_side_missed_still_counts():
    fused = Retriever.fuse({1: 1}, {2: 1})
    assert set(fused) == {1, 2}
    assert fused[1] == fused[2] == 1 / 61       # k = 60, rank 1


def test_k_sets_how_much_the_top_rank_counts():
    # A small k makes rank 1 tower over rank 10; the usual 60 flattens the curve, so
    # one retriever's first place cannot outvote the other retriever's whole list.
    sharp, flat = Retriever.fuse({1: 1, 2: 10}, k=1), Retriever.fuse({1: 1, 2: 10}, k=60)
    assert sharp[1] / sharp[2] > 5 > flat[1] / flat[2]
