"""The token budget on a conversation's history: whole turns, newest first."""

from app.memory import newest_turns


def msg(role: str, n: int) -> dict:
    return {"role": role, "n": n}


TURNS = [msg("user", 10), msg("assistant", 90),     # turn 1: 100 tokens
         msg("user", 10), msg("assistant", 40),     # turn 2:  50
         msg("user", 10), msg("assistant", 20)]     # turn 3:  30


def test_keeps_the_newest_turns_that_fit():
    assert newest_turns(TURNS, limit=80) == TURNS[2:]       # 30 + 50 fit, 100 more does not


def test_a_turn_is_kept_or_dropped_whole():
    kept = newest_turns(TURNS, limit=60)
    assert kept == TURNS[4:]                                # never an answer without its question
    assert kept[0]["role"] == "user"


def test_the_newest_turn_survives_any_budget():
    assert newest_turns(TURNS, limit=1) == TURNS[4:]
