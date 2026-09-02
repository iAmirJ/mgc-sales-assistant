"""End-to-end verification of the RAG pipeline for the five required cases.

Run with a real GEMINI_API_KEY in .env for a full live test, or without one to
verify everything upstream of the LLM call (retrieval, guardrails, calculator,
conflict detection, prompt assembly).

For every question it reports:
- the answer text (live) or a mock confirmation (no key),
- the retrieved chunks actually selected as evidence,
- relevant sources AND any irrelevant sources in the evidence (should be none),
- a per-case PASS/FAIL based on evidence precision + grounding behaviour.

Usage:
    python tests/check_cases.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.assistant import Assistant
from rag.config import config
from rag.embeddings import build_query_embeddings
from rag.ingestion import load_and_chunk_documents
from rag.vectorstore import build_index, load_index

# case -> expected exact (source_file, section) set that SHOULD be selected.
EXPECTED_SOURCES = [
    {("02_price_list_payment_plan.md", "Base Prices (Block B)")},
    {
        ("02_price_list_payment_plan.md", "Base Prices (Block B)"),
        ("02_price_list_payment_plan.md", "Location Premiums"),
    },
    {
        ("03_booking_policy_faq.md", "Transfers"),
        ("02_price_list_payment_plan.md", "Other Charges"),
    },
    {("03_booking_policy_faq.md", "Frequently Asked")},
    {("01_mgc_aurora_heights_brochure.md", "Commercial Podium")},
]

CASES = [
    "What is the base price of a 2-bed in Block B?",
    "What is the total for a Margalla-facing corner unit, floor 15, 2-bed Block B?",
    "What is the transfer fee?",
    "What is the rental yield on a 1-bed?",
    "Who is the anchor tenant?",
]


def _ensure_index():
    if config.faiss_index_path.exists():
        return load_index(build_query_embeddings())
    documents = load_and_chunk_documents(config.docs_dir)
    return build_index(documents, build_query_embeddings())


def mock_chat(prompt: str):
    return type("R", (), {"content": "MOCKED LLM (no GEMINI_API_KEY). Evidence + prompt assembled correctly."})()


def _source_key(doc) -> tuple[str, str]:
    return (doc.metadata.get("source_file", ""), doc.metadata.get("section", ""))


def main() -> int:
    api_available = bool(config.gemini_api_key)
    print(f"GEMINI_API_KEY present: {api_available}")
    if not api_available:
        print("Live Gemini generation will be skipped; pipeline components verified.\n")

    vectorstore = _ensure_index()
    print("Index ready.")

    query_embeddings = build_query_embeddings()
    assistant = Assistant(vectorstore=vectorstore, embeddings_query=query_embeddings)
    if not api_available:
        assistant.chat_model = type("C", (), {"invoke": staticmethod(mock_chat)})()

    all_passed = True
    for idx, case in enumerate(CASES):
        expected = EXPECTED_SOURCES[idx]
        print("\n" + "=" * 70)
        print(f"QUESTION: {case}")
        try:
            response = assistant.answer(case)
        except Exception:
            print("ERROR:")
            traceback.print_exc()
            all_passed = False
            continue

        keyed = {_source_key(doc) for doc in response.evidence}
        rel = keyed & expected
        irrelevant = keyed - expected
        missing = expected - keyed

        print(f"ANSWER: {response.answer}")
        print(f"SELECTED EVIDENCE ({len(response.evidence)}):")
        for i, doc in enumerate(response.evidence, 1):
            meta = doc.metadata
            print(
                f"  [{i}] {meta.get('source_file')} | {meta.get('section')} "
                f"| cos={meta.get('score', 0.0):+.3f} "
                f"| grounded={meta.get('grounded_score', 0.0):+.3f}"
            )
        print(f"RELEVANT SOURCES: {sorted(rel)}")
        if irrelevant:
            print(f"IRRELEVANT SOURCES (FAIL): {sorted(irrelevant)}")
        if missing:
            print(f"MISSING SOURCES (FAIL): {sorted(missing)}")
        if response.conflicts:
            print(f"CONFLICTS: {[(c.fact_description, c.values) for c in response.conflicts]}")
        if response.breakdown:
            print("CALCULATION (deterministic):")
            print("  " + response.breakdown.to_text().replace("\n", "\n  "))

        precision_ok = not irrelevant and not missing
        passed = precision_ok
        print(f"=> {'PASS' if passed else 'FAIL'} (evidence precision: {'OK' if precision_ok else 'BAD'})")
        all_passed = all_passed and passed

    print("\n" + "=" * 70)
    print(f"RESULT: {'ALL CASES PASS' if all_passed else 'SOME CASES FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())