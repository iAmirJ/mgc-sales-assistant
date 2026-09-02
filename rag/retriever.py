"""Retriever: turn a natural-language question into a precise subset of evidence.

Two-stage, metadata-aware selection so a sales question surfaces only the
chunks that actually contribute to the answer (not every near-threshold
neighbour in a small corpus):

1. Vector candidate pool: embed the query, fetch ``pool_k`` candidates with
   cosine similarity (MAX_INNER_PRODUCT over unit vectors).
2. Intent-aware evidence selection:
   - Detect the question's topic with generic lexical rules (pricing,
     transfer fee, rental yield, anchor tenant, amenities, payment, ...).
   - For a detected intent, keep only candidates whose document type / section
     content matches that intent (metadata filtering / keyword affinity).
   - Re-order the surviving pool by a grounded score
     (cosine + small metadata affinity boost + block-entity boost) and cut off
     once a chunk falls more than ``margin`` below the best, so sibling tables
     (e.g. Block A when the user asked Block B) are not displayed.

No answer values are hard-coded here: prices, percentages and totals always
come from the retrieved document text. The raw cosine score (used by the
relevance gate) is preserved unchanged in each Document's ``score`` metadata.

This module is UI-agnostic so it can be validated independently.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from rag.config import config

# ---------------------------------------------------------------------------
# Intent rules: generic keyword -> which document type(s)/section names matter.
# The values are document categories, NOT answers. They keep the evidence
# focused on the part of the corpus the question is actually about.
# ---------------------------------------------------------------------------

_INTENT_RULES: dict[str, dict] = {
    "transfer": {
        "types": {"price_list", "booking_policy"},
        "keywords": ("transfer", "fee"),  # all must appear in the chunk
    },
    "yield": {
        "types": {"booking_policy"},
        "keywords": ("rental", "yield"),
    },
    "anchor": {
        "types": {"brochure"},
        "keywords": ("anchor", "tenant"),
    },
    "amenity": {
        "types": {"brochure"},
        "keywords": ("amenit",),
    },
    "pricing": {
        "types": {"price_list"},
        "sections": ("base prices", "location premiums"),
    },
    "payment": {
        "types": {"price_list", "booking_policy"},
        "sections": ("payment plan", "discounts", "book"),
    },
    "possession": {
        "types": {"booking_policy", "price_list", "brochure"},
        "sections": ("possession", "handover"),
    },
    "approval": {
        "types": {"brochure", "booking_policy"},
        "sections": ("approval", "noc", "cda"),
    },
}

# (intent, trigger substrings) - evaluated in order, most specific first.
_INTENT_TRIGGERS: list[tuple[str, tuple[str, ...]]] = [
    ("transfer", ("transfer fee", "transfer fee",)),
    ("yield", ("rental", "yield",)),
    ("anchor", ("anchor", "tenant",)),
    ("amenity", ("amenit",)),
    ("possession", ("possession", "handover", "completion",)),
    ("approval", ("noc", "approval", "cda", "environmental clearance",)),
    ("payment", ("payment", "plan", "instalment", "installment", "discount",
                 "book", "token", "cash", "schedule",)),
    ("pricing", ("base price", "price", "cost", "how much", "total", "premium",
                 "corner", "margalla", "floor", "2-bed", "1-bed", "3-bed",
                 "studio", "penthouse",)),
]


def detect_intents(question: str) -> list[str]:
    """Return detected intents, ordered most-specific first, no duplicates."""
    q = question.lower()
    found: list[str] = []
    for intent, triggers in _INTENT_TRIGGERS:
        if intent in found:
            continue
        if any(t in q for t in triggers):
            found.append(intent)
    return found


def detect_block(question: str) -> Optional[str]:
    """Return 'block b' / 'block a' when the question names one, else None."""
    q = question.lower()
    if "block b" in q:
        return "block b"
    if "block a" in q:
        return "block a"
    return None


def _query_terms(question: str) -> set[str]:
    """Lower-cased alphanumeric terms >= 3 chars from the question."""
    return {t for t in re.findall(r"[a-z0-9]{3,}", question.lower())}


def _chunk_matches_intent(doc: Document, intent: str) -> bool:
    """Keyword / section affinity for one chunk against one intent rule."""
    rule = _INTENT_RULES.get(intent)
    if rule is None:
        return True
    if doc.metadata.get("document_type") not in rule.get("types", set()):
        return False
    content_low = doc.page_content.lower()
    sections = rule.get("sections", ())
    if sections and any(s in content_low for s in sections):
        return True
    keywords = rule.get("keywords", ())
    return bool(keywords) and all(k in content_low for k in keywords)


def _section_affinity(query_terms: Iterable[str], doc: Document) -> float:
    """Small per-section-term boost: the section header is authoritative."""
    section_low = doc.metadata.get("section", "").lower()
    hits = sum(1 for t in set(query_terms) if t in section_low)
    return min(hits, 4) * 0.02


def _content_affinity(query_terms: Iterable[str], doc: Document) -> float:
    """Small per-content-term boost, capped so it never dominates cosine."""
    content_low = doc.page_content.lower()
    hits = sum(1 for t in set(query_terms) if t in content_low)
    return min(hits, 4) * 0.015


def _block_boost(block: Optional[str], doc: Document) -> float:
    """Strong boost when the asked block actually appears in the section."""
    if not block:
        return 0.0
    section_low = doc.metadata.get("section", "").lower()
    return 0.06 if block in section_low else 0.0


class Retriever:
    """Precise, metadata-aware retrieval over a FAISS vector store."""

    def __init__(
        self,
        vectorstore: FAISS,
        top_k: Optional[int] = None,
        pool_k: Optional[int] = None,
        margin: Optional[float] = None,
    ):
        self.vectorstore = vectorstore
        self.top_k = top_k or config.top_k
        self.pool_k = pool_k or max(self.top_k * 2, 10)
        self.margin = margin if margin is not None else 0.05

    def candidates(self, query: str, k: Optional[int] = None) -> list[Document]:
        """Raw vector candidates with TRUE cosine scores (debugging/tests)."""
        k = k or self.pool_k
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        docs: list[Document] = []
        for doc, score in results:
            meta = dict(doc.metadata)
            meta["score"] = float(score)
            docs.append(Document(page_content=doc.page_content, metadata=meta))
        return docs

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[Document]:
        """Return the evidence chunks for ``query`` (final selection)."""
        k = top_k or self.top_k
        intents = detect_intents(query)
        block = detect_block(query)
        terms = _query_terms(query)

        def grounded(doc: Document) -> float:
            cos = doc.metadata.get("score", 0.0)
            return (
                cos
                + _section_affinity(terms, doc)
                + _content_affinity(terms, doc)
                + _block_boost(block, doc)
            )

        pool = self.candidates(query, k=self.pool_k)
        scored = sorted(pool, key=grounded, reverse=True)

        chosen = scored
        if intents:
            focused = [d for d in scored if _chunk_matches_intent(d, intents[0])]
            if focused:
                chosen = focused

        stable = sorted(chosen, key=grounded, reverse=True)
        if not stable:
            return []

        best = grounded(stable[0])
        picked: list[Document] = []
        for i, doc in enumerate(stable):
            g = grounded(doc)
            if g < best - self.margin:
                continue
            if doc.metadata.get("score", 0.0) < config.min_relevance_score:
                continue
            meta = dict(doc.metadata)
            meta["grounded_score"] = round(float(g), 4)
            meta["selection_rank"] = i
            picked.append(Document(page_content=doc.page_content, metadata=meta))
            if len(picked) >= k:
                break
        return picked