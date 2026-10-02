# The UI image: the Streamlit page, and only what the page imports.
#
#   docker build -f ui/Dockerfile -t docchat-ui .     # the context is the repository root
#
# The same three parts as the API's Dockerfile, with one difference that is the
# whole point of Lesson 5: the page is a client of the API and imports nothing
# from `app`, so this image installs the `ui` dependency group and nothing else.
# No torch, no models, no database driver, no keys. Compare the two image sizes.

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.19 /uv /bin/uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=0 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --only-group ui: Streamlit, httpx and python-dotenv, without the project or its
# dependencies. The lockfile is the same one the API image reads. The cache mount
# keeps uv's downloads out of the image, as in the API's Dockerfile.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --only-group ui --no-install-project

COPY ui ./ui

# The page imports `ui.client`, so /app has to be on the import path. On the host it
# is there by accident: `uv sync` installs the project in editable mode, and the .pth
# file that writes puts the repository root on sys.path. This image installs no
# project, and Streamlit adds only the script's own folder, /app/ui. Without this
# line the server starts, its healthcheck passes, and the page fails on its first
# import. That is the clean-clone test's reason to render the page, not ping it.
ENV PYTHONPATH=/app

EXPOSE 8501

# 0.0.0.0, not localhost: inside a container, localhost is the container itself, and
# a server bound there cannot be reached from anywhere else. Headless: no browser to
# open in here, and no usage statistics sent home.
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
