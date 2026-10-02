# The API image: python -m app.serve, and everything it imports.
#
#   docker build -t docchat-api .          # or let compose do it: docker compose build api
#
# Read it as three parts, ordered from what changes least to what changes most.
# Docker caches each instruction as a layer and reuses it until something it
# depends on changes; after that, every later layer is rebuilt. So the dependencies,
# which change once a month, are installed before the code, which changes every
# hour. A code change then rebuilds one small layer, and the 1 GB below it is reused.
#
# What is not here: the keys (they arrive from .env when the container starts), the
# corpus (mounted), uploaded PDFs and model weights (volumes). See compose.yml.

# 1. A base: Debian with Python 3.12, the version in .python-version. Pinned, because
#    `latest` is "works on my machine" again, a few months later.
FROM python:3.12-slim

# uv, copied out of its own image as a single binary, at the version that wrote uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.8.19 /uv /bin/uv

# copy:     the files go into .venv, not as links to a cache that is not in the image
# compile:  .pyc files now, at build time, rather than on every container's first start
# downloads=0: use this image's Python; never fetch another one
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=0 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 2. The dependencies, from the lockfile alone. --locked fails the build if uv.lock
#    and pyproject.toml disagree, so the image gets exactly the versions on your
#    machine. --no-install-project: the code is not here yet, and that is the point.
#    No dependency groups are installed: jupyter and streamlit stay out of this image.
#    The --mount gives uv its download cache from the build machine, outside the
#    image: without it, every wheel would be stored twice in this layer, once in
#    the cache and once installed, and the image would be a gigabyte bigger.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# 3. The code, and the project itself installed into the environment (a few KB).
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# The port the API listens on inside the container. EXPOSE is documentation; compose
# decides what is published to the host.
EXPOSE 8000

# The same entry point as on the host. Where it binds, and where the databases are,
# is configuration, and compose sets it.
CMD ["python", "-m", "app.serve"]
