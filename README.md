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

Prerequisites: [uv](https://docs.astral.sh/uv/), Docker, and an OpenAI API key. From
`m9-l7`, about 11 GB of disk for the model you run yourself: see
[A model you run yourself](#a-model-you-run-yourself). From `m9-l8`, about 4 GB more
for Langfuse: see [Tests and traces](#tests-and-traces).

```bash
git clone <this repo> && cd reference_app
git checkout m9-l8

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
python scripts/eval_golden.py --label m9-l8 --api http://localhost:8000   # the same, through the API
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
| fallback provider | `llm.py` | `FALLBACK_BASE`, `FALLBACK_MODEL`, `FALLBACK_API_KEY`, `FALLBACK_TIMEOUT`, `PRIMARY_TIMEOUT`, `PRIMARY_RETRIES` | `usage.model` names the fallback; a warning in the log |
| injection guard | `generation.py`, `prompts/guard.md` | `GUARD_CONTEXT` | context blocks in `<document>` tags; a new `prompt_version` |

The cache key is the standalone query, the retriever's corpus fingerprint, a hash of
the prompt files, the model and `top_k`. Answers from the fallback and refusals from
the retrieval gate are never cached. A conversation started by one user is a 404 to
any other.

At `m9-l4` the fallback was Gemini through its OpenAI-shaped endpoint, the GenAI Lesson 7
second provider. From `m9-l7` it is a model in Ollama, in the same three settings. Leave
`FALLBACK_API_KEY` empty to run without one.

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
docker compose up -d --build               # postgres, qdrant, api, ui, from m9-l7 ollama, from m9-l8 langfuse; minutes the first time
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
| `compose.yml` | the services, their healthchecks, and their volumes; `ollama` from Lesson 7, Langfuse from Lesson 8 |
| `compose.gpu.yml` | an override that gives `ollama` an NVIDIA GPU, used through `COMPOSE_FILE` in `.env` |

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

## A model you run yourself

```bash
docker compose up -d --build                     # the first start also pulls OLLAMA_MODEL, about 2.5 GB
docker compose exec ollama ollama pull qwen3:4b-instruct-2507-q8_0   # the 8-bit copy, for the Lesson 7 notebook
docker compose exec ollama ollama list           # what is in the ollama volume
docker compose logs ollama-pull                  # the download, if the API is still waiting for it
```

From `m9-l7` the stack runs a language model of its own: Qwen3 4B Instruct, at 4 bits, in
[Ollama](https://ollama.com). It is the Lesson 4 fallback. When OpenAI fails, the API asks
`http://ollama:11434/v1` instead, with the same client, and nothing leaves the machine.

| service | what it is |
|---|---|
| `ollama` | the server: llama.cpp behind an OpenAI-shaped endpoint. Its weights are on the `ollama` volume. On the host it answers on `OLLAMA_PORT` |
| `ollama-pull` | a one-shot job that downloads `OLLAMA_MODEL` and exits. The API waits for it to succeed, so a first start takes as long as the download |

| setting | default | what it does |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:4b-instruct-2507-q4_K_M` | what `ollama-pull` downloads, and the fallback model inside compose. Empty means no download and no fallback |
| `OLLAMA_PORT` | `11434` | the host port. Change it if an Ollama installed natively already has 11434 |
| `OLLAMA_CONTEXT_LENGTH` | `4096` | the longest prompt Ollama reads. A longer one is cut from the front, silently |
| `FALLBACK_TIMEOUT` | `300` | how long the fallback may take. On a CPU a RAG answer takes a minute or more |

The container runs on the CPU. Expect a minute or more for an answer on a laptop, almost all of it
spent reading the 1,600-to-2,000-token prompt, not writing the answer. On a GPU it takes seconds. On a Mac, Docker cannot reach the GPU:
install Ollama natively, which uses Metal, and change `FALLBACK_BASE` in the `api` service of
`compose.yml` to `http://host.docker.internal:11434/v1`. With an NVIDIA GPU and the NVIDIA
container toolkit, add `COMPOSE_FILE=compose.yml:compose.gpu.yml` to `.env` and run
`docker compose up -d`: the override hands the card to the `ollama` service.
`docker compose logs ollama | grep "inference compute"` names the GPU when it worked. If it
still says `cpu`, write the toolkit's device list once with
`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`.

To answer with nothing leaving the machine at all, make the local model the primary. That is
configuration too:

```bash
docker compose run --rm --no-deps -e OPENAI_BASE=http://ollama:11434/v1 -e OPENAI_API_KEY=ollama \
  -e GEN_MODEL=qwen3:4b-instruct-2507-q4_K_M -e FALLBACK_API_KEY= \
  api python -m app.ask --no-save "What was Albany International's effective tax rate for 2022?"
```

The clean-clone test sets `OLLAMA_MODEL` to nothing, so it starts the `ollama` service and
downloads no model. The Ollama image is 4 GB on its own, because it carries GPU libraries
whether or not it uses them.

Coming from `m9-l6`? No new Python dependencies. Copy the two Lesson 7 blocks from
`.env.example` into your `.env`. They replace the Gemini fallback, so keep your Gemini lines
commented out if you want them back. Then run `docker compose up -d --build`. One Python setting
is new, `FALLBACK_TIMEOUT`, and the pipeline did not change.

## Tests and traces

```bash
uv sync --all-groups                             # pytest is new, in the `test` group
pytest tests/unit                                # about a second; nothing needs to be running
docker compose up -d postgres qdrant
pytest tests/integration                         # about 20 seconds: the real API, a fake model
pytest                                           # both

docker compose up -d langfuse-web                # the traces, at http://localhost:3000
python scripts/show_trace.py <trace_id>          # one trace as a tree, in the terminal
python scripts/show_trace.py --session 812       # every turn of conversation 812
```

From `m9-l8` the application has tests and leaves traces.

**The unit tests** cover the pieces whose output is decided by their input alone: the
chunker, reciprocal rank fusion, the citation parser, the history budget, the retry
decision, and the cache key. None of them touches a database, a model or the network.
What they do not test is the model's wording, which no assertion can pin down.

**The integration test** starts `python -m app.serve` on a free port against the real
Postgres and Qdrant, with `OPENAI_BASE` pointed at a fake model in
`tests/integration/fake_llm.py`. The fake answers the OpenAI-shaped route with the
first sentence of context block [1], cited, so the test is free and the same every
time. It uploads a PDF the test writes by hand, waits for `ready`, asks about it, and
checks for a cited answer from that document, in JSON and as a stream. Then it deletes
the document. Nothing in `app/` knows it is being tested.

**The golden set over HTTP.** `--api` sends each question to `POST /chat`, as a new
conversation from a user of its own (`golden-<label>-<id>`), so memory and the daily
budget stay out of it. Each row of the scorecard keeps the `conversation_id` and the
`trace_id` of its answer.

**Traces** go to Langfuse, which runs in compose: the web app on `LANGFUSE_PORT`, a
worker, ClickHouse, Redis and MinIO, and a database of its own on our Postgres, made by
the one-shot `langfuse-db`. Its first start creates the project and the keys in
`.env`, and a user to log in as (`LANGFUSE_USER_EMAIL`, `LANGFUSE_USER_PASSWORD`).
Every turn is a trace whose steps are `window`, `condense`, `cache`, `answer`,
`retrieve`, `rerank` and each `llm` call with its prompt, tokens and cost. It is named
after the `X-User-Id` and the conversation, and `/chat` returns its `trace_id`.
Leave `LANGFUSE_PUBLIC_KEY` empty and every decorator is a no-op; the tests run that
way. The API does not wait for Langfuse: a tracer that is down loses traces, never
answers.

| setting | default | what it does |
|---|---|---|
| `RERANK` | `true` | the cross-encoder over the shortlist, and the refusal gate on its score. Off, the fused order stands and the gate is skipped |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | `pk-lf-docchat`, `sk-lf-docchat` | the project's keys; Langfuse is created with them, the API sends with them |
| `LANGFUSE_PORT` | `3000` | Langfuse's page on the host |
| `LANGFUSE_BASE_URL` | `http://localhost:${LANGFUSE_PORT}` | where the app on the host sends traces; inside compose, `http://langfuse-web:3000` |

The defaults for Langfuse's keys, passwords and secrets are for your own machine.
Change every one of them before it runs anywhere else.

**The bug Lesson 8 found.** Up to `m9-l7` the response cache's key held the query, the
corpus, the prompt, the model and `top_k`, and nothing about retrieval. With `RERANK`
off, the golden set over HTTP did not move, because every answer came from the cache;
the trace of any one of them showed `cache: hit` and no `retrieve`. The key now holds
the retrieval settings, and `tests/unit/test_cache.py` fails if one is left out. The
commit before the tag, `m9-l8~1`, has the old key, and the Lesson 8 notebook runs it
once to show the bug.

Coming from `m9-l7`? Run `uv sync --all-groups` (langfuse, and pytest in a new `test`
group). Copy the Lesson 8 blocks of `.env.example` into your `.env`: `RERANK`, and the
`LANGFUSE_` lines. Then `docker compose up -d --build`, which pulls about 4 GB of
images for Langfuse. The database needs nothing, and every cached answer is retired
once, because the cache key gained a part.

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
  tracing.py      Langfuse: @observe, and what each step reports      (Lesson 8)
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
  eval_golden.py  the golden set, run as a script, writing a scorecard; --api over HTTP (Lesson 8)
  show_trace.py   a trace from Langfuse as a tree                      (Lesson 8)
  watch.sql       the newest rows, every second, for a psql terminal   (Lesson 5)
  clean_clone.py  fresh clone, copy .env, compose up, upload, ask, tear down   (Lesson 6)
tests/                                                                 (Lesson 8)
  unit/           the deterministic pieces: chunker, fusion, citations, history, retries, cache key
  integration/    the real API, Postgres and Qdrant, a fake model: upload, ask, a cited answer
data/
  pdfs/           the corpus: ten annual reports
  uploads/        PDFs uploaded through the API, gitignored
  corpus.json     which PDFs, and what each is called
  golden/         the 40-question golden set from GenAI Lesson 17
  runs/           scorecards, gitignored, one per run
compose.yml       the whole stack: postgres, qdrant, api, ui, and their volumes   (Lesson 6)
                  and ollama, a model of our own, with a job that pulls it       (Lesson 7)
                  and Langfuse, with ClickHouse, Redis and MinIO behind it       (Lesson 8)
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

| metric | `m9-l1` | `m9-l2` | `m9-l3` | `m9-l4` | `m9-l5` | `m9-l6` | `m9-l7` | `m9-l8` |
|---|---|---|---|---|---|---|---|---|
| hit@5 | 0.963 | 0.963 | 0.963 | 0.963 | 0.963 | 0.963 | 0.963 | 0.963 |
| answer accuracy | 0.852 | 0.852 | 0.852 | 0.852 | 0.852 | 0.852 | 0.852 | 0.852 |
| false answers on the unanswerable | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| answers carrying a citation | 0.889 | 0.889 | 0.889 | 0.926 | 0.926 | 0.926 | 0.926 | 0.926 |
| cost for the whole set | $0.024 | $0.010 | $0.010 | $0.011 | $0.011 | $0.011 | $0.011 | $0.011 |

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

At `m9-l7` nothing moved, question by question. The primary is still gpt-4o-mini, and it did
not fail during the run, so the new fallback was never asked. What an outage would cost is a
separate scorecard: the same forty questions with Qwen3 4B in Ollama as the primary, and no
fallback.

| metric | gpt-4o-mini (`m9-l7`) | Qwen3 4B, 8-bit | Qwen3 4B, 4-bit |
|---|---|---|---|
| hit@5 | 0.963 | 0.963 | 0.963 |
| answer accuracy | 0.852 | 0.889 | 0.778 |
| false answers on the unanswerable | 0.000 | 0.077 | 0.000 |
| answers carrying a citation | 0.926 | 0.926 | 0.889 |
| cost for the whole set | $0.011 | $0 | $0 |

Retrieval does not change, so hit@5 does not either. The 4-bit model, the one in compose, gets
three answers fewer than gpt-4o-mini and makes no false answers. The 8-bit model gets one more
answer than gpt-4o-mini and answers one question it should have refused. Twenty-seven
answerable questions cannot rank the three more finely than that. These two runs are in
`data/golden/m9-l7-local-*-scorecard.json`. They ran on a GPU, because on a laptop CPU each
answer takes a minute or more.

At `m9-l8` the golden set ran through the API for the first time: forty `POST /chat`
requests to the running stack, each a conversation of its own, through memory, the
cache and the fallback. Nothing moved, question by question, against `m9-l7` in
process. No answer came from the cache, because the cache key gained a part in this
tag and every key was new.

With `RERANK=false` the same run gives the number Lesson 8 goes looking for:

| metric | `m9-l8` | `m9-l8`, reranker off |
|---|---|---|
| hit@5 | 0.963 | 0.852 |
| answer accuracy | 0.852 | 0.778 |
| false answers on the unanswerable | 0.000 | 0.000 |
| answers carrying a citation | 0.926 | 0.889 |
| cost for the whole set | $0.011 | $0.011 |

Four questions lose the page their answer is on from the top five, and the model,
still told to answer only from the context, refuses or answers with less. No false
answers appear, though the gate is off with the reranker: on these forty questions
the prompt's refusal clause holds on its own.
