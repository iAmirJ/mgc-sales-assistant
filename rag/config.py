"""Central configuration for the MGC Sales Assistant.

All environment-dependent settings live in a `.env` file next to this project
and are loaded here via python-dotenv. Keeping configuration here means the
rest of the code never has to know where a value comes from, and no API key is
ever hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root is one level above this file (mgc-ai-task/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load variables from the .env file in the project root, if present.
# The .env file must never be committed (see .gitignore).
load_dotenv(PROJECT_ROOT / ".env")


def _get(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


class Config:
    """Typed access to application configuration."""

    # --- Google / Gemini ---
    gemini_api_key: str | None = _get("GEMINI_API_KEY")
    gemini_model: str = _get("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_embedding_model: str = _get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    # --- Retrieval ---
    top_k: int = int(_get("TOP_K", "5"))
    # Minimum cosine similarity (best retrieved chunk) required to answer.
    min_relevance_score: float = float(_get("MIN_RELEVANCE_SCORE", "0.60"))

    # --- Paths ---
    docs_dir: Path = PROJECT_ROOT / (_get("DOCS_DIR", "docs"))
    vectorstore_dir: Path = PROJECT_ROOT / (_get("VECTORSTORE_DIR", "vectorstore"))
    faiss_index_file: str = _get("FAISS_INDEX_FILE", "index.faiss")

    @property
    def faiss_index_path(self) -> Path:
        return self.vectorstore_dir / self.faiss_index_file


config = Config()
