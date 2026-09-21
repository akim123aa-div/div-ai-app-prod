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

    # the stores. Lesson 2 starts using database_url; Lesson 1 only needs Qdrant.
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

    # paths and noise
    corpus_dir: str = "data/pdfs"
    log_level: str = "INFO"

    @property
    def corpus_path(self) -> Path:
        """Relative paths resolve against the repository, not the shell's cwd."""
        p = Path(self.corpus_dir)
        return p if p.is_absolute() else ROOT / p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """One instance per process. Import this, never the class."""
    return Settings()


settings = get_settings()
