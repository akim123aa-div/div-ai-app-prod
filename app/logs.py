"""Logging, at the depth of "levels exist and you set one".

`print` writes to stdout with no level, no timestamp and no way to turn it down.
That is fine in a notebook, where you are the only reader and the cell output is
the point. It stops being fine the moment the code runs somewhere you are not.

A logger gives you three things print does not: a level you can raise in
production and lower when something breaks, the module the line came from, and
one place to change the format for every line in the application.
"""

import logging
import sys

from app.config import settings

_configured = False


def setup_logging(level: str | None = None) -> None:
    """Call once, at the top of an entry point. Calling it twice is harmless."""
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=(level or settings.log_level).upper(),
        format="%(asctime)s %(levelname)-7s %(name)-16s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,          # stdout belongs to the answer, stderr to the machinery
    )
    # These two are chatty at INFO and have nothing to say that we want.
    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("sentence_transformers").setLevel("WARNING")
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
