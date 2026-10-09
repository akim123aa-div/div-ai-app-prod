"""The retry decision: which failures are worth trying again."""

import pytest

from app.llm import should_retry


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_failures_are_retried(status):
    assert should_retry(status)


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_the_callers_mistakes_are_not(status):
    assert not should_retry(status)


def test_two_429s_with_opposite_answers():
    # A rate limit clears in seconds. An exhausted quota does not clear today.
    assert should_retry(429, "rate_limit_exceeded")
    assert not should_retry(429, "insufficient_quota")
