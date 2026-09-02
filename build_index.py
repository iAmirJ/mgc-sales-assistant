"""Build (or rebuild) the FAISS index over the MGC documents.

Usage:
    python build_index.py

Requires a GEMINI_API_KEY in .env (embeddings call the Gemini API). The index
is persisted under vectorstore/index.faiss so the app can load it later without
re-embedding.
"""

from __future__ import annotations

import sys

from rag.config import config
from rag.embeddings import build_document_embeddings
from rag.ingestion import load_and_chunk_documents
from rag.vectorstore import build_index


def main() -> int:
    print(f"Loading Markdown documents from: {config.docs_dir}")
    documents = load_and_chunk_documents(config.docs_dir)
    print(f"Loaded {len(documents)} chunks from {len(set(d.metadata['source_file'] for d in documents))} documents")

    print(f"Building embeddings with model: {config.gemini_embedding_model}")
    embeddings = build_document_embeddings()

    print(f"Building FAISS index and saving to: {config.faiss_index_path}")
    build_index(documents, embeddings)

    print("Index built and saved successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())