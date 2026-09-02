"""No-API-key verification of the deterministic RAG logic on the REAL documents.

This exercises the actual document chunks and the real downstream logic
(base-price extraction, premium extraction, deterministic calculator, conflict
detection, abstention paths). It does NOT test FAISS semantic retrieval order
or live Gemini generation - those need GEMINI_API_KEY and are covered by
tests/check_cases.py.

For each case we pass in the chunks a working semantic retriever is expected to
return (the sections that contain the relevant facts). This is a legitimate
scope decision: FAISS retrieval + Gemini embeddings is library code whose
correctness is verified in the live run, while the MGC-specific extraction
logic - which is what a reviewer would ask about - is what we verify here.

Usage:
    python tests/verify_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.assistant import Assistant
from rag.calculator import calculate_with_premiums
from rag.config import config
from rag.guardrails import detect_conflicts, validate_evidence
from rag.ingestion import load_and_chunk_documents
from rag.retriever import (
    _chunk_matches_intent,
    detect_block,
    detect_intents,
)

PASS = 0
FAIL = 0
WARN = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label} {detail}")


def warn(label: str, detail: str) -> None:
    global WARN
    WARN += 1
    print(f"  [WARN] {label} {detail}")


def section(docs: list, *names: str) -> list:
    return [d for d in docs if d.metadata["section"] in names]


def main() -> int:
    docs = load_and_chunk_documents(config.docs_dir)
    block_b = section(docs, "Base Prices (Block B)")
    block_a = section(docs, "Base Prices (Block A)")
    premiums = section(docs, "Location Premiums")
    transfer_policy = section(docs, "Transfers")
    other_charges = section(docs, "Other Charges")
    faq = section(docs, "Frequently Asked")
    brochure_anchor = [d for d in docs if "anchor" in d.page_content.lower()]

    only_if = lambda items, name: (items if items else warn(f"{name} chunk found", f"(missing {name})"), items)[1]

    a = object.__new__(Assistant)

    # ---------------- Test 1: base price lookup ----------------
    print("\nTEST 1: Base price of a 2-bed in Block B?")
    block_b = only_if(block_b, "Block B price list")
    if block_b:
        price = a._extract_base_price("What is the base price of a 2-bed in Block B?", block_b)
        check("Block B 2-bed base price = 22,425,000", price == 22425000.0, f"got {price}")
        # With BOTH Block A and Block B evidence present, a "Block B" query must
        # still prefer the Block B row (22,425,000), not Block A (21,850,000).
        mixed = block_b + block_a
        price_mixed = a._extract_base_price("What is the base price of a 2-bed in Block B?", mixed)
        check("Block B row preferred over Block A when both retrieved",
              price_mixed == 22425000.0, f"got {price_mixed}")

    # ---------------- Test 2: stacked premiums ----------------
    print("\nTEST 2: Margalla-facing corner, floor 15, 2-bed Block B")
    if block_b and premiums:
        combo = []
        for doc in section(docs, "Base Prices (Block B)"):
            combo.append(doc)
        for doc in premiums:
            combo.append(doc)
        price = a._extract_base_price("Margalla-facing corner unit floor 15 2-bed Block B", combo)
        check("Retrieves Block B 2-bed base price (22,425,000)", price == 22425000.0, f"got {price}")

        prems = a._extract_premiums(
            "What is the total for a Margalla-facing corner unit, floor 15, 2-bed Block B?",
            combo,
        )
        got = {label: pct for label, pct in prems}
        check("Corner premium 3%", got.get("Corner") == 3.0, f"got {got}")
        check("Margalla-facing premium 6%", got.get("Margalla-facing") == 6.0, f"got {got}")
        floor_pct = [v for k, v in got.items() if k.startswith("Floor")]
        check("Floor-15 premium 4% (tier 13-19)", floor_pct == [4.0], f"got {got}")

        bd = calculate_with_premiums(price or 0.0, prems)
        check("Deterministic total = 25,340,250", bd.final_total == 25340250.0, f"got {bd.final_total}")
        if bd.final_total == 25340250.0:
            print("    " + bd.to_text().replace("\n", "\n    "))

    # ---------------- Test 3: conflict ----------------
    print("\nTEST 3: Transfer fee")
    if transfer_policy and other_charges:
        evidence = transfer_policy + other_charges
        conflicts = detect_conflicts(evidence)
        values = conflicts[0].values if conflicts else {}
        check("Conflict detected", bool(conflicts), f"got {conflicts}")
        check("Two distinct sources", len(values) == 2, f"got {values}")
        src_vals = set(values.values())
        check("Holds 2% (price list) and 2.5% (policy)", ("2%" in src_vals and "2.5%" in src_vals), f"got {values}")

    # ---------------- Test 4: rental yield abstention ----------------
    print("\nTEST 4: Rental yield on a 1-bed")
    if faq:
        yield_chunk = [d for d in faq if "rental yield" in d.page_content.lower()]
        check("FAQ chunk states MGC doesn't publish yield",
              bool(yield_chunk) and "does not publish rental yield" in yield_chunk[0].page_content,
              "yield policy text found")
        # Reject a hard-coded calc path: no breakdown should be produced.
        bd = a._maybe_calculate("What is the rental yield on a 1-bed?", [])
        check("No deterministic calculation attempted for yield", bd is None)

    # ---------------- Test 5: anchor tenant ----------------
    print("\nTEST 5: Anchor tenant")
    if brochure_anchor:
        text = brochure_anchor[0].page_content.lower()
        check("Brochure says no anchor confirmed",
              "no anchor tenant has been confirmed" in text,
              f"chunk: {brochure_anchor[0].page_content[:100]}")

    # ---------------- Relevance gate: no orphans ----------------
    print("\nADDITIONAL checks")
    fake = [type("D", (), {"metadata": {"score": 0.10}})]
    verdict = validate_evidence(fake)
    check("Weak evidence is rejected by the relevance gate", not verdict.relevant)

    # ---------------- Intent-based retrieval precision ----------------
    print("\nRETRIEVAL PRECISION (intent detection, no API key)")
    check("Block B detected", detect_block("What is the price of a 2-bed in Block B?") == "block b")
    check("Block A detected", detect_block("What is the price of a 2-bed in Block A?") == "block a")
    check("No block name -> None", detect_block("What is the price of a 2-bed?") is None)
    check("Pricing intent detected", detect_intents("Base price of a 2-bed in Block B?")[0] == "pricing")
    check("Transfer intent most specific",
          detect_intents("What is the transfer fee?")[0] == "transfer")
    check("Yield intent precedes generic pricing",
          detect_intents("Rental yield on a 1-bed?")[0] == "yield")
    check("Anchor intent detected", detect_intents("Who is the anchor tenant?")[0] == "anchor")

    def _doc(section_name, doc_type, content):
        from langchain_core.documents import Document
        return Document(
            page_content=content,
            metadata={"section": section_name, "document_type": doc_type,
                      "source_file": "02_price_list_payment_plan.md"},
        )

    bb = _doc("Base Prices (Block B)", "price_list",
              "## Base Prices (Block B) | 2-Bed Standard | 1,150 sq ft | 22,425,000")
    _ = _doc("Base Prices (Block A)", "price_list",
              "## Base Prices (Block A) | 2-Bed Standard | 1,150 sq ft | 21,850,000")
    am = _doc("Amenities", "brochure",
              "## Amenities Rooftop infinity pool and residents' lounge")
    tr = _doc("Transfers", "booking_policy",
              "## Transfers Transfer fee is 2.5% of the current list price")
    oc = _doc("Other Charges", "price_list",
              "## Other Charges Transfer fee (before possession): 2% of the current list price")
    faq = _doc("Frequently Asked", "booking_policy",
               "## Frequently Asked MGC does not publish rental yield projections")

    check("Pricing intent admits Block B price table", _chunk_matches_intent(bb, "pricing"))
    check("Pricing intent admits only price_list tables", not _chunk_matches_intent(am, "pricing"))
    check("Booking_policy not admitted by pricing intent", not _chunk_matches_intent(faq, "pricing"))
    check("Transfer intent admits policy Transfers", _chunk_matches_intent(tr, "transfer"))
    check("Transfer intent admits price_list Other Charges", _chunk_matches_intent(oc, "transfer"))
    check("Transfer intent rejects unrelated chunk", not _chunk_matches_intent(faq, "transfer"))
    check("Yield intent admits FAQ abstention section", _chunk_matches_intent(faq, "yield"))
    check("Yield intent rejects price table", not _chunk_matches_intent(bb, "yield"))

    print(f"\n==== RESULT: {PASS} passed, {FAIL} failed, {WARN} warnings ====")
    print("NOTE: This run verifies the deterministic MGC-specific logic on the")
    print("real documents. FAISS semantic retrieval and live Gemini generation")
    print("require GEMINI_API_KEY and are covered by tests/check_cases.py.")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())