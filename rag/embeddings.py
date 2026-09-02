"""Embeddings: build a Gemini text-embedding client.

We use Google's Gemini text embedding model via LangChain's
`GoogleGenerativeAIEmbeddings`. The task only contains Markdown text
documents, so a text embedding model is the correct and sufficient choice.

The integration supports a `task_type` so the model can optimise its
representation for document indexing vs. query retrieval. We expose two
instances (one for documents, one for queries) but share the same underlying
model.
"""

from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from rag.config import config


def _require_api_key() -> str:
    if not config.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "Google AI Studio API key before building the index."
        )
    return config.gemini_api_key


def build_document_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Embeddings client tuned for indexing document chunks."""
    return GoogleGenerativeAIEmbeddings(
        model=config.gemini_embedding_model,
        google_api_key=_require_api_key(),
        task_type="RETRIEVAL_DOCUMENT",
    )


def build_query_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Embeddings client tuned for embedding a query at retrieval time."""
    return GoogleGenerativeAIEmbeddings(
        model=config.gemini_embedding_model,
        google_api_key=_require_api_key(),
        task_type="RETRIEVAL_QUERY",
    )
