"""docchat - the reference application for Module 9.

One layer per lesson, each one a git tag:

    m9-l1  this package, run from the command line
    m9-l2  Postgres as the source of truth, Qdrant derived from it
    m9-l3  a FastAPI service in front of it
    m9-l4  conversation memory, budgets, a cache, a fallback, an injection guard
    m9-l5  a Streamlit client in ui/, which talks to the API and imports none of this
    ...

Nothing in here is production-ready, and the notebooks say where the gaps are.
"""

__version__ = "0.1.0"
