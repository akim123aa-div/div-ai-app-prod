"""The HTTP layer, Lesson 3. Everything below `app/api/` is transport.

    main.py           the app: startup, the health check, error handling
    schemas.py        every request and response body, as a Pydantic model
    documents.py      upload, list, poll, delete         /documents
    chat.py           one question, whole or streamed     /chat, /chat/stream
    conversations.py  what was said, read back            /conversations
    usage.py          what a user has spent of a budget   /usage            (Lesson 4)
    chunks.py         the passage behind a citation       /chunks           (Lesson 5)

The rule for this package is the one `db.py` and `index.py` already follow, one
layer up: no retrieval, no prompts and no SQL in here. A handler parses the
request, calls a function the command line could have called, and shapes the
result. If a handler grows logic, that logic belongs in the pipeline, where the
CLI, the golden set and Lesson 8's tests can reach it too.
"""
