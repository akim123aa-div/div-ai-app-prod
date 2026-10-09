"""Shared by every test. Runs before any test module imports `app`.

Tests do not write traces. The settings are read once, at the first import of
`app.config`, so the switch is set here, before that can happen: an empty key in
the environment wins over the one in .env, and tracing is off for the whole run.
"""

import os

os.environ["LANGFUSE_PUBLIC_KEY"] = ""
