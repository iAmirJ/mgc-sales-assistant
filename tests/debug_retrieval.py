"""Dev-only: inspect exactly what the retriever returns for each test question.

Usage (requires the built index + GEMINI_API_KEY for query embeddings):
    python tests\\debug_retrieval.py

Shows, for each question:
- the raw vector candidates (true cosine scores) BEFORE selection,
- the final selected evidence (what the assistant/UI will actually use),
including which chunks were selected and why.

This is a developer tool ONLY; it is not part of the final Streamlit UI.
The Streamlit app exposes the same detail behind a Sidebar "Developer mode"
toggle instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.embeddings import build_query_embeddings
from rag.retriever import Retriever, detect_intents, detect_block
from rag.vectorstore import load_index

QUESTIONS = [
    "Base price of a 2-bed in Block B?",
    "Total for a Margalla-facing corner unit, floor 15, 2-bed Block B?",
    "What's the transfer fee?",
    "Rental yield on a 1-bed?",
    "Who is the anchor tenant?",
]


def _preview(text: str, width: int = 100) -> str:
    text = " ".join(text.split())
    return text[:width] + ("..." if len(text) > width else "")


def _describe(doc) -> str:
    meta = doc.metadata
    g = meta.get("grounded_score", "")
    return (
        f"{meta.get('source_file')} | {meta.get('section')} | "
        f"type={meta.get('document_type')} | "
        f"cos={meta.get('score', 0.0):+.3f}"
        + (f" | grounded={g:+.3f}" if isinstance(g, float) else "")
    )


def main() -> None:
    embeddings = build_query_embeddings()
    vs = load_index(embeddings)
    if vs is None:
        print("No index found. Run: python build_index.py")
        return

    retriever = Retriever(vs)

    for q in QUESTIONS:
        print("=" * 100)
        print(f"QUERY: {q}")
        print(f"  intents={detect_intents(q)} block={detect_block(q)}")
        print("  --- raw vector candidates (top_k pool) ---")
        for i, doc in enumerate(retriever.candidates(q, k=retriever.pool_k), 1):
            print(f"  [{i}] {_describe(doc)}")
            print(f"        {_preview(doc.page_content)}")
        print("  --- final SELECTED evidence (what the assistant uses) ---")
        selected = retriever.retrieve(q)
        if not selected:
            print("  (none)")
        for doc in selected:
            print(f"  * {_describe(doc)}")
            print(f"        {_preview(doc.page_content)}")
        print()

    print("NOTE: cos = raw cosine similarity (gate uses this); grounded = cos + metadata affinity boosts.")


if __name__ == "__main__":
    main()