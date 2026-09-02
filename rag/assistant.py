"""Assistant: orchestrate retrieval, grounding and answer generation.

This is the single service the Streamlit UI calls. It:

1. Embeds the user's question and retrieves relevant chunks (Retriever).
2. Validates the evidence (Guardrails) and abstains if it is too weak.
3. Runs conflict detection (Guardrails).
4. Runs deterministic pricing calculations when the question requests one.
5. Builds a grounding-aware system prompt and asks Gemini to answer strictly
   from the retrieved evidence.
6. Returns a structured response (answer, sources, warnings, breakdown).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.calculator import PriceBreakdown, calculate_with_premiums
from rag.config import config
from rag.guardrails import (
    Conflict,
    conflicts_to_text,
    detect_conflicts,
    validate_evidence,
)
from rag.ingestion import format_evidence
from rag.retriever import Retriever

# Zone of questions we treat as pricing requests for deterministic calculation.
_PRICING_SIGNALS = (
    "total",
    "price for",
    "cost",
    "how much",
    "margalla",
    "corner",
    "floor premium",
    "total price",
    "final price",
)


def _looks_like_pricing_query(question: str) -> bool:
    q = question.lower()
    return any(signal in q for signal in _PRICING_SIGNALS)


class Assistant:
    """Grounded Q&A service over the MGC documents."""

    def __init__(
        self,
        vectorstore: FAISS,
        embeddings_query,
        chat_model: Optional[ChatGoogleGenerativeAI] = None,
    ):
        self.retriever = Retriever(vectorstore)
        self._query_embeddings = embeddings_query
        self.chat_model = chat_model or self._default_chat_model()

    @staticmethod
    def _default_chat_model() -> ChatGoogleGenerativeAI:
        if not config.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add "
                "your Google AI Studio API key."
            )
        return ChatGoogleGenerativeAI(
            model=config.gemini_model,
            google_api_key=config.gemini_api_key,
            temperature=0.0,
        )

    def answer(self, question: str) -> "AssistantResponse":
        """Answer ``question`` grounded only in the retrieved MGC evidence."""
        # 1. Retrieve evidence.
        evidence = self.retriever.retrieve(question)
        if not evidence:
            return self._build_abstention(
                question, "No evidence could be retrieved from the MGC documents."
            )

        # 2. Validate evidence relevance.
        verdict = validate_evidence(evidence)
        if not verdict.relevant:
            return self._build_abstention(
                question,
                "I don't have enough information in the provided MGC documents "
                "to answer this reliably.",
                evidence=evidence,
            )

        # 3. Detect conflicts (e.g. transfer fee 2% vs 2.5%).
        conflicts = detect_conflicts(evidence)

        # 4. Deterministic calculation when the question calls for one.
        breakdown = self._maybe_calculate(question, evidence)

        # 5. Grounding-aware generation.
        prompt = self._build_prompt(question, evidence, conflicts, breakdown)
        answer = self._generate(prompt)

        return AssistantResponse(
            question=question,
            answer=answer,
            sources=self._collect_sources(evidence),
            evidence=evidence,
            conflicts=conflicts,
            breakdown=breakdown,
        )

    # --- helpers -----------------------------------------------------------

    def _maybe_calculate(self, question: str, evidence: list) -> Optional[PriceBreakdown]:
        """Run deterministic pricing if the question looks like a pricing query.

        Rather than parsing arbitrary LLM text, we extract the verified base
        price and premium percentages from the retrieved chunks (all values
        originate in the documents). If any required piece is missing, we
        return None and let the LLM answer without a breakdown.
        """
        if not _looks_like_pricing_query(question):
            return None

        base_price = self._extract_base_price(question, evidence)
        if base_price is None:
            return None

        premiums = self._extract_premiums(question, evidence)
        if not premiums:
            return None

        try:
            return calculate_with_premiums(base_price, premiums)
        except (TypeError, ValueError):
            return None

    def _extract_base_price(self, question: str, evidence: list) -> Optional[float]:
        """Find the base price matching the requested unit in the evidence.

        When the question names a block ("Block A"/"Block B"), we prefer rows
        from a chunk whose section/heading matches that block, so a Block B
        query never silently picks up the Block A 2-bed price.
        """
        q = question.lower()
        block_phrase = None
        if "block b" in q:
            block_phrase = "block b"
        elif "block a" in q:
            block_phrase = "block a"
        unit_quality = None
        if "2-bed" in q or "2 bed" in q or "two-bed" in q:
            unit_quality = "2bed"
        elif "1-bed" in q or "1 bed" in q:
            unit_quality = "1bed"
        elif "corner" in q:
            unit_quality = "corner"

        if unit_quality is None:
            return None

        # Order evidence: chunks whose section mentions the requested block
        # first when the block is specified.
        def _block_rank(doc) -> int:
            if block_phrase is None:
                return 0
            section_text = f"{doc.metadata.get('section', '')} {doc.metadata.get('document_name', '')}"
            return 0 if block_phrase in section_text.lower() else 1

        ordered = sorted(evidence, key=_block_rank)

        for doc in ordered:
            for line in doc.page_content.splitlines():
                if "sq ft" not in line and not line.startswith("|"):
                    continue
                if self._row_matches(line, unit_quality):
                    return self._price_from_row(line)
        return None

    @staticmethod
    def _row_matches(line: str, unit_quality: str) -> bool:
        low = line.lower()
        if unit_quality == "2bed":
            return "2-bed" in low or "2 bed" in low
        if unit_quality == "1bed":
            return "1-bed" in low or "1 bed" in low
        if unit_quality == "corner":
            return "corner" in low
        return False

    @staticmethod
    def _price_from_row(line: str) -> Optional[float]:
        """Extract the base price (3rd column) from a pipe-table row."""
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) >= 3:
            try:
                return float(str(cells[2]).replace(",", ""))
            except ValueError:
                return None
        return None

    def _extract_premiums(self, question: str, evidence: list) -> list[tuple[str, float]]:
        """Extract applicable premium percentages from the evidence chunk.

        We rely on the documents' explicit statement that premiums are
        cumulative, and pick the premium rows that apply to the unit (floor
        tier, corner, facing). This logic contains NO hard-coded answer values;
        the percentages come from the retrieved document text itself.
        """
        q = question.lower()
        premiums: list[tuple[str, float]] = []

        # Applicable unit features derived from the question.
        wants_corner = "corner" in q
        wants_margalla = "margalla" in q
        floor_num = self._extract_floor(question)
        floor_tier = self._floor_tier(floor_num) if floor_num is not None else None

        for doc in evidence:
            for line in doc.page_content.splitlines():
                low = line.lower().strip()

                # Only the bullet definitions carry a premium value.
                if "corner" in low and wants_corner:
                    value = self._percent_in_line(line)
                    if value is not None:
                        premiums.append(("Corner", value))

                if "margalla-facing" in low and wants_margalla:
                    value = self._percent_in_line(line)
                    if value is not None:
                        premiums.append(("Margalla-facing", value))

                if floor_tier and self._tier_in_line(low, floor_tier):
                    value = self._percent_in_line(line)
                    if value is not None:
                        premiums.append((f"Floor {floor_tier}", value))

        # De-duplicate by label, preserving order.
        seen: set[str] = set()
        unique: list[tuple[str, float]] = []
        for label, value in premiums:
            if label not in seen:
                unique.append((label, value))
                seen.add(label)
        return unique

    @staticmethod
    def _tier_in_line(line: str, tier: str) -> bool:
        """Match a floor tier label like 'floors 13-19' in a line."""
        digits = re.findall(r"\d+", line)
        tier_digits = {str(n) for n in re.findall(r"\d+", tier)}
        return "floors" in line and tier_digits.issubset(set(digits))

    @staticmethod
    def _percent_in_line(line: str) -> Optional[float]:
        """Return the first percentage figure on a line, e.g. '+4%' -> 4.0."""
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
        return float(m.group(1)) if m else None

    @staticmethod
    def _extract_floor(question: str) -> Optional[int]:
        m = re.search(r"floor\s*(\d{1,2})", question.lower())
        if m:
            return int(m.group(1))
        m = re.search(r"(\d{1,2})\s*(?:st|nd|rd|th)?\s*floor", question.lower())
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _floor_tier(floor: int) -> Optional[str]:
        if 13 <= floor <= 19:
            return "floors 13-19"
        if 20 <= floor <= 22:
            return "floors 20-22"
        return None

    def _build_prompt(
        self,
        question: str,
        evidence: list,
        conflicts: list[Conflict],
        breakdown: Optional[PriceBreakdown],
    ) -> str:
        evidence_text = format_evidence(evidence)
        conflict_text = conflicts_to_text(conflicts) if conflicts else ""
        calc_text = breakdown.to_text() if breakdown else ""

        system = SYSTEM_PROMPT
        user = f"""QUESTION:
{question}

GROUNDING RULES:
{system}

EVIDENCE (retrieved from MGC documents only - do NOT use any other knowledge):
{evidence_text}

{conflict_text}

{'CALCULATION BREAKDOWN (deterministic, verified against documents):' + calc_text if calc_text else ''}

Now answer the QUESTION using ONLY the EVIDENCE above. Follow the GROUNDING RULES strictly.
"""
        return user

    def _generate(self, prompt: str) -> str:
        try:
            response = self.chat_model.invoke(prompt)
            return response.content.strip() if response else ""
        except Exception as exc:  # surface API or model errors clearly
            raise RuntimeError(
                f"Gemini generation failed: {exc}"
            ) from exc

    @staticmethod
    def _collect_sources(evidence: list) -> list[dict]:
        sources: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for doc in evidence:
            key = (doc.metadata.get("source_file", ""), doc.metadata.get("section", ""))
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "source_file": doc.metadata.get("source_file", ""),
                    "section": doc.metadata.get("section", ""),
                }
            )
        return sources

    @staticmethod
    def _build_abstention(
        question: str,
        reason: str,
        evidence: Optional[list] = None,
    ) -> "AssistantResponse":
        return AssistantResponse(
            question=question,
            answer=reason,
            sources=(
                [
                    {
                        "source_file": d.metadata.get("source_file", ""),
                        "section": d.metadata.get("section", ""),
                    }
                    for d in evidence
                ]
                if evidence
                else []
            ),
        )


@dataclass
class AssistantResponse:
    """Structured result returned to the UI."""

    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    evidence: list = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    breakdown: Optional[PriceBreakdown] = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "conflicts": [c.fact_description for c in self.conflicts],
            "breakdown": self.breakdown.to_dict() if self.breakdown else None,
        }


SYSTEM_PROMPT = """You are the MGC Sales Assistant. A MGC salesperson asks you a question about the MGC Aurora Heights project in Islamabad. Answer in plain, concise, professional language suitable for a salesperson.

STRICT GROUNDING RULES:
1. Answer ONLY from the retrieved MGC document evidence provided below.
2. Never invent facts. If the documents do not contain the answer, say so.
3. Never use general or world knowledge to fill gaps; the answer must be fully supported by the supplied MGC documents.
4. If the evidence is insufficient or the answer is not in the documents, state: "I don't have enough information in the provided MGC documents to answer this reliably."
5. If the retrieved documents disagree (a conflict is flagged above), do NOT silently pick one side. Explicitly present each conflicting value and its source, and say the documents do not establish which is currently authoritative and to confirm with MGC.
6. Cite the relevant source document and section for each claim you make.
7. Be concise and useful for a salesperson. Use PKR formatting with thousands separators.
8. If a deterministic CALCULATION BREAKDOWN was provided above, use those exact numbers (they are verified and correct). If no breakdown is given, do NOT attempt to compute prices yourself; instead state the base price and any applicable premiums from the documents without calculating a total.
9. Never present an unsupported value as if it were in the documents.
10. If the documents explicitly say something is 'not confirmed' or 'no anchor tenant has been confirmed', repeat that status; do not invent a name or a confirmation.
"""