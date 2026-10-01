"""/usage: what the caller has spent against their budget. Lesson 4.

The same numbers the chat dependency checks, for a client to show before the
user hits the limit rather than after. Lesson 5's UI puts them under the input.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app import budget
from app.api.chat import user
from app.api.schemas import BudgetOut

router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("", response_model=BudgetOut)
def get_usage(user_id: Annotated[str, Depends(user)]) -> dict:
    """Tokens used in the last 24 hours by whoever X-User-Id names."""
    return budget.check(user_id).to_dict()
