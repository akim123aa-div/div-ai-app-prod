# docchat

The reference application for **Module 9, Shipping an AI Application**. It is the
GenAI Weeks 5 to 7 retriever with a codebase around it, and it grows one layer
per lesson:

| tag | what it adds |
|---|---|
| `m9-l1` | the package: config, logging, entry points, a scorecard |
| `m9-l2` | Postgres as the source of truth, Qdrant derived from it |
| `m9-l3` | a FastAPI service, background ingestion, streamed answers |
| `m9-l4` | conversation memory, budgets, a cache, a fallback, guardrails |
| `m9-l5` | a Streamlit client |
| `m9-l6` | Dockerfiles and a full-stack compose file |
| `m9-l7` | a model you run yourself, through Ollama |
| `m9-l8` | tests and traces |

Fall behind and you check out the tag for the lesson you are on. Nothing later
depends on your copy having survived.

**This is not production-ready and is not trying to be.** Every simplification is
named in the module handbook next to what a production system would use instead.

## Setup

Prerequisites: [uv](https://docs.astral.sh/uv/), Docker, and an OpenAI API key.

```bash
git clone <this repo> && cd reference_app
git checkout m9-l6

cp .env.example .env          # then put your key in it
uv sync --all-groups          # creates .venv from pyproject.toml
source .venv/bin/activate

docker compose up -d postgres qdrant
```

If port 5432 or 6333 is already taken on your machine, change `POSTGRES_PORT`
and `QDRANT_PORT` in `.env`. From `m9-l6` the two URLs are built from them, so
that is the whole change. Nothing in the code needs to know.

Or skip the virtual environment altogether and run everything in containers: see
[Containers](#containers).

## Running it

```bash
python -m app.ingest                       # parse, chunk, write Postgres, index Qdrant. A few minutes.
python -m app.ask "How many patents did Aurora Innovation hold at year end?"
python -m app.ask --conversation 12 "And the year before?"   # a follow-up, condensed against the history
python -m app.reindex                      # drop the vector index, rebuild it from Postgres
python scripts/eval_golden.py --label m9-l6
```

Run these from the repository root. The first `ingest` downloads two small
models from Hugging Face, about 200 MB, and then never touches the network
again except to call the generator.

Postgres is the source of truth: documents, chunks, conversations, messages.
Qdrant holds vectors and a chunk ID per point, nothing it cannot recompute, so
`reindex` rebuilds it without opening a PDF. `ask` saves each question and
answer as a conversation, with citations, tokens and cost.

Coming from `m9-l1`? Run `uv sync --all-groups` (two new dependencies) and then
`python -m app.ingest` once. Ingestion now writes somewhere new.

## The API

```bash
python -m app.serve                        # http://localhost:8000, about 15s to start
open http://localhost:8000/docs            # every route, with a "Try it out" button
```

| route | what it does |
|---|---|
| `POST /documents` | upload a PDF; answers `202` with a `pending` document and ingests it in the background |
| `GET /documents`, `GET /documents/{id}` | list, or poll one until `ready` or `failed` |
| `DELETE /documents/{id}` | remove it from Postgres, Qdrant and the running retriever |
| `POST /chat` | one question, the whole answer as JSON |
| `POST /chat/stream` | the same answer as server-sent events: `delta`*, `citations`, `usage`, `done`, or `error` |
| `GET /conversations`, `GET /conversations/{id}` | the caller's conversations, from Postgres; another user's is a 404 (Lesson 5) |
| `GET /usage` | what the caller has spent of their token budget (Lesson 4) |
| `GET /chunks/{id}` | the passage behind a citation; encode the `#` in the ID as `%23` (Lesson 5) |
| `GET /health` | what is being searched, what startup cost, the prompt version and the fallback |

Uploaded PDFs are kept in `data/uploads/` and get their ID from the filename, so
uploading the same file again replaces it. The embedder, the reranker and BM25 are
built once at startup. A job running when the server stops is lost, and the next
start marks it `failed`, because background tasks live in the API process.

Coming from `m9-l2`? Run `uv sync --all-groups` (three new dependencies). The
database needs nothing: the tables did not change.

## Conversations and guardrails

From `m9-l4` a chat request is one turn of a conversation, and `app/conversation.py`
runs it: load the window, condense the follow-up, look in the cache, answer, record.
Every request names its user in an `X-User-Id` header (missing means `anonymous`).
That header is a label, not authentication.

| control | where | setting | what you see |
|---|---|---|---|
| conversation memory | `memory.py` | `HISTORY_TOKENS`, `SUMMARY_TOKENS` | older turns folded into a summary in `summaries`; `window` in every response |
| per-user budget | `budget.py` | `USER_DAILY_TOKENS` | `429` with `Retry-After` before any work; `GET /usage` |
| response cache | `cache.py` | `CACHE_ANSWERS` | `cached: true`, no tokens, no retrieval; rows in `answer_cache` |
| fallback provider | `llm.py` | `FALLBACK_BASE`, `FALLBACK_MODEL`, `FALLBACK_API_KEY`, `PRIMARY_TIMEOUT`, `PRIMARY_RETRIES` | `usage.model` names the fallback; a warning in the log |
| injection guard | `generation.py`, `prompts/guard.md` | `GUARD_CONTEXT` | context blocks in `<document>` tags; a new `prompt_version` |

The cache key is the standalone query, the retriever's corpus fingerprint, a hash of
the prompt files, the model and `top_k`. Answers from the fallback and refusals from
the retrieval gate are never cached. A conversation started by one user is a 404 to
any other.

The fallback is Gemini through its OpenAI-shaped endpoint, the GenAI Lesson 7 second
provider. Leave `FALLBACK_API_KEY` empty to run without one. Lesson 7 points it at
Ollama instead.

The guard stops a plain instruction planted in an uploaded PDF and does not stop a
better-written one; the Lesson 4 notebook measures both. It is the minimum, not a
defence you can rely on.

Coming from `m9-l3`? No new dependencies. Copy the new block from `.env.example`
into your `.env` and add a Gemini key. The two new tables are created when the
server starts. Postgres needs nothing else, and the corpus does not need ingesting
again: the chunker fix in this tag only changes documents of one or two pages.

## The UI

```bash
python -m app.serve                        # the API, in one terminal
streamlit run ui/app.py                    # the page, in another: http://localhost:8501
```

From `m9-l5` the application has a face: a Streamlit chat page in `ui/`. It is a
client of the API and nothing more. It sends HTTP to `API_URL`, draws what comes back,
and imports nothing from `app/`. It needs only the `ui` dependency group (Streamlit,
httpx, python-dotenv), which is what lets Lesson 6 build its image without torch.

| the page | what it calls |
|---|---|
| the chat, streamed | `POST /chat/stream`, events drawn as they arrive |
| Sources under each answer | `GET /chunks/{id}` for each citation, cached for five minutes |
| the sidebar: usage, conversations, documents | `GET /usage`, `GET /conversations`, `GET /documents`, on every rerun |
| an upload and its status | `POST /documents`, then `GET /documents/{id}` once a second from a fragment |

Refusals, a spent budget, someone else's conversation, a provider failing mid-answer
and an API that is down each have a state of their own on the page. The user in the
sidebar is sent as `X-User-Id`, and it is still a label, not a login.

To watch a request go through the stack, give each process a terminal:

```bash
python -m app.serve                                    # each step of a turn, each ingestion stage
streamlit run ui/app.py                                # each rerun, each call the page makes
docker compose logs -f --since 1s qdrant               # each vector search and write
docker compose exec -T postgres psql -U docchat < scripts/watch.sql
```

The last one refreshes the newest rows of `messages`, `documents` and `answer_cache`
every second, because Postgres logs no queries by default. Uploads have no terminal of
their own: they are ingested inside the API process, so they log to the first one.

Coming from `m9-l4`? Run `uv sync --all-groups` (Streamlit is new). Nothing else: no
tables changed, and the corpus does not need ingesting again.

## Containers

```bash
cp .env.example .env                       # then put your key in it
docker compose up -d --build               # postgres, qdrant, api, ui; the first build takes a few minutes
open http://localhost:8501                 # the page; the API is still on http://localhost:8000
```

From `m9-l6` the whole application starts with one command. Postgres and Qdrant
are the same two services as before; `api` and `ui` are built from this
repository. Use one way of running the app at a time, because both want ports 8000
and 8501.

| file | what it is |
|---|---|
| `Dockerfile` | the API image: `python:3.12-slim`, uv, the locked dependencies, then the code. About 1.6 GB, almost all of it torch |
| `ui/Dockerfile` | the UI image: the `ui` dependency group only. About 0.6 GB: no torch, no models, no keys |
| `.dockerignore` | keeps `.venv`, `.env` and `data/` out of the build context |
| `compose.yml` | the four services, their healthchecks, and four volumes |

Inside compose, only configuration differs from the host. The API gets `.env` as its
environment with a few values on top, set in `compose.yml`. The UI gets no `.env`
at all, only `API_URL`, because it needs no key:

| setting | on the host (`.env`) | inside compose |
|---|---|---|
| `DATABASE_URL` | `...@localhost:${POSTGRES_PORT}/...` | `...@postgres:5432/...` |
| `QDRANT_URL` | `http://localhost:${QDRANT_PORT}` | `http://qdrant:6333` |
| `API_HOST` | `127.0.0.1` | `0.0.0.0`, or nothing outside the container could reach it |
| `API_URL`, for the UI | `http://localhost:8000` | `http://api:8000` |
| `HF_HOME` | `~/.cache/huggingface` | `/models`, a volume |

Data lives on volumes, not in images: `pgdata` and `qdrantdata` as before, `uploads`
for uploaded PDFs, and `models` for the embedder and reranker weights, downloaded on
the first start. The corpus in `data/pdfs` is mounted read-only. The API waits for
Postgres and Qdrant to be healthy, and the UI waits for the API.

A new stack starts empty. Load the ten reports with a one-off container, or upload a
PDF from the page:

```bash
docker compose run --rm api python -m app.ingest      # a new container, same network and volumes, removed after
docker compose logs -f api ui                         # Lesson 5's terminals, from every service at once
docker compose stop api ui                            # back to the app on the host; the databases keep running
docker compose down                                   # remove containers and the network; the volumes stay
python scripts/clean_clone.py                         # the clean-clone test, below
```

The golden set runs inside the API's image the same way, with the script, the golden set
and the runs directory mounted. `--user` makes the scorecard yours rather than root's:

```bash
docker compose run --rm --no-deps --user "$(id -u):$(id -g)" -e HF_HUB_OFFLINE=1 \
  -v ./scripts:/app/scripts:ro -v ./data/golden:/app/data/golden:ro -v ./data/runs:/app/data/runs \
  api python scripts/eval_golden.py --label m9-l6
```

The project name in `compose.yml` is still `m9-infra`, so a stack you ran in Lessons
1 to 5 keeps its volumes and its corpus. Volume names come from the project name, so
renaming it would start you with an empty stack.

**The clean-clone test** clones the committed repository into a temporary directory,
copies `.env`, runs `docker compose up` under a separate project name and free ports,
uploads a PDF, asks about it, checks for a cited answer and a page, and removes
everything it made. Uncommitted changes are not tested. The final project has to
pass it.

Coming from `m9-l5`? No new Python dependencies. Docker and about 3 GB of disk for
the two images. Your `.env` still works as it is; the new `.env.example` writes the
Postgres credentials once and builds both URLs from them, if you want to copy that.

## Layout

```
app/
  config.py       every value that differs between machines, and nothing else
  logs.py         one logging setup, called by the entry points
  llm.py          the GenAI Lesson 7 provider client, as a module
  models.py       the embedder and the reranker, in this process, on the CPU
  chunking.py     PDF -> pages -> sections -> token windows  (GenAI Lesson 13)
  db.py           the only module that knows Postgres exists: four tables, all the SQL
  index.py        the only module that knows Qdrant exists
  ingestion.py    PDFs -> Postgres rows -> Qdrant points
  retrieval.py    dense + BM25 + fusion + rerank          (GenAI Lesson 15)
  generation.py   context blocks, citations, refusal      (GenAI Lesson 16)
  memory.py       the conversation window: summary + recent turns, under a budget  (Lesson 4)
  budget.py       per-user token budgets                  (Lesson 4)
  cache.py        the response cache and its key          (Lesson 4)
  conversation.py one turn: window, condense, cache, answer, record                 (Lesson 4)
  prompts/        prompts are files, not string literals: answer, condense, guard, summarise
  api/            the HTTP layer: routers, schemas, startup  (Lesson 3)
    main.py         the app, lifespan, /health, error mapping
    schemas.py      every request and response body, and the SSE events
    documents.py    upload -> background job -> poll
    chat.py         /chat and /chat/stream
    conversations.py
    usage.py        what a user has spent              (Lesson 4)
    chunks.py       the passage behind a citation      (Lesson 5)
  ingest.py       entry point: python -m app.ingest
  ask.py          entry point: python -m app.ask "..."
  reindex.py      entry point: python -m app.reindex
  serve.py        entry point: python -m app.serve
ui/               the Streamlit client: HTTP only, no `app` imports   (Lesson 5)
  client.py       every URL, header and status code the page uses, and an SSE parser
  app.py          the page: session state, the chat, sources, upload status, the sidebar
scripts/
  eval_golden.py  the golden set, run as a script, writing a scorecard
  watch.sql       the newest rows, every second, for a psql terminal   (Lesson 5)
  clean_clone.py  fresh clone, copy .env, compose up, upload, ask, tear down   (Lesson 6)
data/
  pdfs/           the corpus: ten annual reports
  uploads/        PDFs uploaded through the API, gitignored
  corpus.json     which PDFs, and what each is called
  golden/         the 40-question golden set from GenAI Lesson 17
  runs/           scorecards, gitignored, one per run
compose.yml       the whole stack: postgres, qdrant, api, ui, and their volumes   (Lesson 6)
Dockerfile        the API image                                                    (Lesson 6)
ui/Dockerfile     the UI image: the `ui` dependency group, nothing else            (Lesson 6)
.dockerignore     what never reaches a build: .venv, .env, data/                   (Lesson 6)

Postgres tables, created by `create_all` on first run:
  documents       one row per PDF; pending -> ready | failed
  chunks          the text that is searched and cited; id = "doc#pN#i"
  conversations   a thread of turns, owned by the X-User-Id that started it
  messages        each turn, with citations (JSON), tokens and cost of the whole turn
  summaries       the older turns of a conversation, folded   (Lesson 4, derived)
  answer_cache    answers already paid for, by key           (Lesson 4, derived)
```

## The scorecard

Forty questions, twenty-seven of them answerable, ten documents.

| metric | `m9-l1` | `m9-l2` | `m9-l3` | `m9-l4` | `m9-l5` | `m9-l6` |
|---|---|---|---|---|---|---|
| hit@5 | 0.963 | 0.963 | 0.963 | 0.963 | 0.963 | 0.963 |
| answer accuracy | 0.852 | 0.852 | 0.852 | 0.852 | 0.852 | 0.852 |
| false answers on the unanswerable | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| answers carrying a citation | 0.889 | 0.889 | 0.889 | 0.926 | 0.926 | 0.926 |
| cost for the whole set | $0.024 | $0.010 | $0.010 | $0.011 | $0.011 | $0.011 |

`m9-l1` is the baseline. Every later lesson changes something underneath it,
and the question each time is whether these numbers moved. At `m9-l2` the
quality numbers did not. The cost did, because `m9-l1` over-counted it: the
golden set runs four questions in parallel on one client, and each question's
cost was that client's running total before and after, which included its
neighbours' calls. `m9-l2` reads the cost from each call.

At `m9-l3` nothing moved, question by question. The API is a transport; the
pipeline behind it is the same functions, and the golden set still calls them
in-process. Lesson 8 runs it over HTTP.

At `m9-l4` the golden set still asks one question at a time, so memory and the
cache are not on its path. The guard is: every prompt is about a hundred tokens
longer and the context is in `<document>` tags. Accuracy, retrieval and refusals
held. One question gained a citation (its answer is still wrong), and the cost
rose by about a tenth.

At `m9-l5` nothing moved, question by question. No line of the pipeline changed:
the UI is a client of the API, and the golden set does not go through either.

At `m9-l6` the golden set ran inside the API's image, as a one-off container, against
the same Postgres and Qdrant: a different Linux, Python 3.12.13 instead of 3.12.11, and
every package installed fresh from the lockfile. Nothing moved, question by question.
