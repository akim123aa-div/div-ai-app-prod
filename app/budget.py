"""Per-user token budgets. Lesson 4.

A served chatbot spends money on behalf of whoever sends it a request. Without a
limit, one user with a loop spends everyone's month. So each user has a budget
of USER_DAILY_TOKENS, tokens in plus out, over a rolling 24 hours, and a request
from a user who has spent it is refused with 429 before any work starts.

Three choices, each one simpler than production:

* The user is whoever the X-User-Id header names. That is a label, not an
  identity: anyone can send any header. Real authentication puts a verified
  user ID in the same place, a token checked by the API, and this module would
  not change. It is named here and left.
* The count is tokens, read from `messages`, not dollars. Tokens are what the
  provider meters, and the whole turn is counted: condensing, summarising and
  answering. A cached answer costs nothing and counts nothing.
* The check happens before the turn and the cost lands after it, so the last
  request under the line can carry a user over it by one turn. Production
  reserves an estimate up front and settles afterwards. Here one turn is a few
  thousand tokens, and the overshoot is shown rather than prevented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app import db
from app.config import settings

WINDOW_HOURS = 24
ANONYMOUS = "anonymous"


@dataclass
class Budget:
    user_id: str
    limit: int
    used: int
    reset_seconds: float      # until the oldest counted turn leaves the window

    @property
    def remaining(self) -> int:
        return max(self.limit - self.used, 0)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def to_dict(self) -> dict:
        return {**asdict(self), "remaining": self.remaining, "window_hours": WINDOW_HOURS}


def check(user_id: str | None) -> Budget:
    """What this user has spent in the window, and what is left."""
    user_id = user_id or ANONYMOUS
    used, reset = db.tokens_used(user_id, WINDOW_HOURS)
    return Budget(user_id=user_id, limit=settings.user_daily_tokens, used=used,
                  reset_seconds=round(reset))
