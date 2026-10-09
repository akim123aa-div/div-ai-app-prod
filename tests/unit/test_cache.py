"""The cache key: every input that shapes an answer is in it.

The regression test for the bug Lesson 8 found through a trace. The key had no
retrieval settings in it, so turning the reranker off changed nothing a user saw:
every answer came from the cache. A test like this is written the day a bug is
found, so that it is found only once.
"""

from types import SimpleNamespace

import pytest

from app import cache
from app.config import settings


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    # key_parts asks the retriever for its fingerprint and the client for its model.
    # Stand-ins with those two attributes keep this test away from Postgres and OpenAI.
    monkeypatch.setattr(cache, "get_retriever", lambda: SimpleNamespace(fingerprint="corpus1"))
    monkeypatch.setattr(cache, "get_client", lambda: SimpleNamespace(model="model1"))


def key(query: str = "What was the tax rate?") -> str:
    return cache.make_key(cache.key_parts(query))


def test_the_same_question_asked_differently_is_one_key():
    assert key("What was the tax rate?") == key("  what was the TAX rate ")


@pytest.mark.parametrize("name, value", [("rerank", False), ("shortlist", 50),
                                         ("gate", 0.5), ("top_k", 3)])
def test_a_retrieval_setting_changes_the_key(monkeypatch, name, value):
    before = key()
    monkeypatch.setattr(settings, name, value)
    assert key() != before
