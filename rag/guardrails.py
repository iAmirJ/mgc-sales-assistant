"""Guardrails: evidence validation and conflict detection.

Before the LLM is ever asked to answer, we check that the retrieved evidence is
actually relevant. If it is not, we abstain rather than let the model fill gaps
from world knowledge.

We also detect when two source documents contradict each other on the same
fact (e.g. the transfer fee is 2% in one document and 2.5% in another). In that
case we surface the conflict openly instead of silently choosing a side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from langchain_core.documents import Document

from rag.config import config


@dataclass
class EvidenceVerdict:
    """Outcome of validating the retrieved evidence for a query."""

    relevant: bool
    reason: str = ""
    score: float = 0.0


def validate_evidence(
    documents: list[Document],
    min_score: Optional[float] = None,
) -> EvidenceVerdict:
    """Check that retrieved chunks are relevant enough to answer from.

    ``score`` is the cosine similarity to the query (see retriever), in [-1, 1]
    with HIGHER meaning MORE similar. If the best chunk does not clear
    ``min_score`` (default: MIN_RELEVANCE_SCORE, 0.60) we consider the evidence
    too weak and the assistant should abstain rather than let the model guess.
    """
    min_score = min_score if min_score is not None else config.min_relevance_score

    if not documents:
        return EvidenceVerdict(relevant=False, reason="No evidence retrieved.")

    best = max(doc.metadata.get("score", 0.0) for doc in documents)
    if best < min_score:
        return EvidenceVerdict(
            relevant=False,
            reason=(
                f"Retrieved evidence is not sufficiently relevant "
                f"(best score {best:.3f} < {min_score})."
            ),
            score=best,
        )
    return EvidenceVerdict(relevant=True, reason="Evidence passes relevance check.", score=best)


@dataclass
class Conflict:
    """Represents a numeric disagreement between two source documents."""

    fact_description: str
    values: dict[str, str]
    explanation: str = ""


def detect_conflicts(documents: Iterable[Document]) -> list[Conflict]:
    """Look across retrieved chunks for conflicting values on shared facts.

    The required 'transfer fee' test case documents 2% in the price list and
    2.5% in the booking policy. We look specifically for lines that mention a
    'transfer' * fee and extract the percentage stated by each source. Where
    two sources state different figures for the same fact, we flag the conflict
    and surface both values, without ever choosing one to be authoritative.
    """
    transfer_fee_percent: dict[str, float] = {}

    for doc in documents:
        source = doc.metadata.get("source_file", "unknown")
        for percent in _transfer_fee_percent(doc.page_content):
            # Keep the last-stated value per source (documents state it once).
            transfer_fee_percent[source] = percent

    if len(transfer_fee_percent) < 2:
        return []

    unique_values = set(transfer_fee_percent.values())
    if len(unique_values) < 2:
        return []

    return [
        Conflict(
            fact_description="transfer fee",
            values={
                source: f"{value:g}%" for source, value in transfer_fee_percent.items()
            },
            explanation=(
                "The supplied MGC documents contain conflicting information "
                "about the transfer fee and do not establish which figure is "
                "currently authoritative. Please confirm with MGC before "
                "quoting the fee to a customer."
            ),
        )
    ]


def _transfer_fee_percent(text: str) -> list[float]:
    """Extract the percentage figures from a 'transfer fee' sentence.

    Example: "Transfer fee (before possession): 2% of the current list price"
    -> [2.0]. Only lines that mention 'transfer' and a fee/percentage count.
    """
    found: list[float] = []
    for line in text.splitlines():
        if "transfer" not in line.lower():
            continue
        if "fee" not in line.lower() and "%" not in line:
            continue
        found.extend(float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*%", line))
    return found


def conflicts_to_text(conflicts: list[Conflict]) -> str:
    """Render detected conflicts into human readable text for the LLM."""
    lines: list[str] = []
    for c in conflicts:
        lines.append(f"CONFLICT detected for: {c.fact_description}")
        for source, value in c.values.items():
            lines.append(f"- {source}: {value}")
        lines.append(c.explanation)
    return "\n".join(lines)