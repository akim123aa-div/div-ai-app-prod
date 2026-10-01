"""Every value that changes between one machine and another, in one object.

The rule this module exists to enforce: configuration is data, code is code.
No hostname, port, model name, threshold or key appears anywhere else in the
package. That line is drawn here, in Lesson 1, because it is what makes Lesson 6
a configuration change rather than a rewrite -- `localhost:5432` becomes
`postgres:5432` in a file, and no Python moves.

Read from the environment, and the environment is loaded from `.env`, which is
in `.gitignore`. A secret that is never in the code cannot be pushed by accident.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed settings. A missing key fails at import, not at 3 a.m. in a request."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # the model provider, as built in GenAI Lesson 7
    openai_api_key: str = ""
    openai_base: str = "https://api.openai.com/v1"
    gen_model: str = "gpt-4o-mini"

    # the stores. Postgres holds the truth, Qdrant an index derived from it.
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chunks"
    database_url: str = "postgresql+psycopg2://docchat:docchat@localhost:5432/docchat"

    # models that run in this process, on the CPU
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # retrieval, the Lesson 15 and 16 numbers
    shortlist: int = 20
    top_k: int = 5
    gate: float = 0.15
    chunk_tokens: int = 400
    chunk_overlap: int = 60

    # the API (Lesson 3). The server binds host:port; clients call api_url.
    # They are different values because in Lesson 6 the server binds 0.0.0.0
    # inside a container and the UI reaches it as http://api:8000.
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_url: str = "http://localhost:8000"
    max_upload_mb: int = 50

    # the conversation (Lesson 4). Tokens of past turns allowed back into the
    # window, and a cap on the summary that stands in for the turns that left it.
    # The summary also never takes more than a quarter of HISTORY_TOKENS.
    history_tokens: int = 1200
    summary_tokens: int = 200

    # what one user may spend, in tokens in + out, over a rolling 24 hours.
    # Keyed on the X-User-Id header, which is a name, not a proof of identity.
    user_daily_tokens: int = 50_000

    # the response cache, and the injection defence on retrieved text
    cache_answers: bool = True
    guard_context: bool = True

    # the fallback provider, tried when the primary fails or times out. Any
    # OpenAI-shaped endpoint will do; Lesson 7 points it at Ollama. Empty key and
    # empty model mean no fallback. The primary gets a short leash so that an
    # outage costs seconds before the fallback runs, not minutes of retries.
    fallback_base: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    fallback_model: str = "gemini-2.5-flash-lite"
    fallback_api_key: str = ""
    primary_timeout: float = 20.0
    primary_retries: int = 1

    # paths and noise
    corpus_dir: str = "data/pdfs"
    upload_dir: str = "data/uploads"
    log_level: str = "INFO"

    @property
    def corpus_path(self) -> Path:
        """Relative paths resolve against the repository, not the shell's cwd."""
        p = Path(self.corpus_dir)
        return p if p.is_absolute() else ROOT / p

    @property
    def upload_path(self) -> Path:
        """Where uploaded PDFs are kept. A directory here, a volume in Lesson 6,
        object storage in production."""
        p = Path(self.upload_dir)
        return p if p.is_absolute() else ROOT / p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """One instance per process. Import this, never the class."""
    return Settings()


settings = get_settings()
