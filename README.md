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
git checkout m9-l2

cp .env.example .env          # then put your key in it
uv sync --all-groups          # creates .venv from pyproject.toml
source .venv/bin/activate

docker compose up -d postgres qdrant
```

If port 5432 or 6333 is already taken on your machine, change `POSTGRES_PORT`
and `QDRANT_PORT` in `.env` and point `DATABASE_URL` and `QDRANT_URL` at the
new ones. Nothing in the code needs to know.

## Running it

```bash
python -m app.ingest                       # parse, chunk, write Postgres, index Qdrant. A few minutes.
python -m app.ask "How many patents did Aurora Innovation hold at year end?"
python -m app.reindex                      # drop the vector index, rebuild it from Postgres
python scripts/eval_golden.py --label m9-l2
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
  prompts/        prompts are files, not string literals
  ingest.py       entry point: python -m app.ingest
  ask.py          entry point: python -m app.ask "..."
  reindex.py      entry point: python -m app.reindex
scripts/
  eval_golden.py  the golden set, run as a script, writing a scorecard
data/
  pdfs/           the corpus: ten annual reports
  corpus.json     which PDFs, and what each is called
  golden/         the 40-question golden set from GenAI Lesson 17
  runs/           scorecards, gitignored, one per run
compose.yml       Postgres and Qdrant. Lesson 6 adds the rest of the stack.

Postgres tables, created by `create_all` on first run:
  documents       one row per PDF; status is always "ready" until Lesson 3
  chunks          the text that is searched and cited; id = "doc#pN#i"
  conversations   a thread of turns; user_id is empty until Lesson 4
  messages        each turn, with citations (JSON), tokens and cost
```

## The scorecard

Forty questions, twenty-seven of them answerable, ten documents.

| metric | `m9-l1` | `m9-l2` |
|---|---|---|
| hit@5 | 0.963 | 0.963 |
| answer accuracy | 0.852 | 0.852 |
| false answers on the unanswerable | 0.000 | 0.000 |
| answers carrying a citation | 0.889 | 0.889 |
| cost for the whole set | $0.024 | $0.010 |

`m9-l1` is the baseline. Every later lesson changes something underneath it,
and the question each time is whether these numbers moved. At `m9-l2` the
quality numbers did not. The cost did, because `m9-l1` over-counted it: the
golden set runs four questions in parallel on one client, and each question's
cost was that client's running total before and after, which included its
neighbours' calls. `m9-l2` reads the cost from each call.
