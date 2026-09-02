"""Deterministic price calculator.

Financial arithmetic must NOT be delegated to the LLM. Instead:

1. Retrieval finds the relevant pricing facts (base price, applicable
   premiums) from the MGC documents.
2. The verified values are passed into this Python calculator.
3. The calculator returns a transparent breakdown.

The calculator contains NO hard-coded MGC answers. It implements *generic*
logic: give it a base price and a list of premium percentages and it sums them
deterministically, mirroring how the price list describes premiums as
"cumulative" percentages added on top of the base price.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PriceBreakdown:
    """A transparent, inspectable price calculation."""

    base_price: float
    premium_lines: list[tuple[str, float]] = field(default_factory=list)
    final_total: float = 0.0

    def to_dict(self) -> dict:
        """A serialisable representation for the UI / prompt."""
        return {
            "base_price": self.base_price,
            "premiums": [
                {"label": label, "percent": percent}
                for label, percent in self.premium_lines
            ],
            "final_total": self.final_total,
        }

    def to_text(self) -> str:
        """A readable breakdown used as evidence for the LLM."""
        lines = [f"Base price: PKR {self.base_price:,.2f}"]
        total_premium = 0.0
        for label, percent in self.premium_lines:
            amount = self.base_price * (percent / 100.0)
            total_premium += amount
            lines.append(
                f"- {label}: +{percent:g}% = +PKR {amount:,.2f}"
            )
        calced = self.base_price + total_premium
        lines.append(f"Subtotal of premiums: PKR {total_premium:,.2f}")
        lines.append(f"Final total: PKR {calced:,.2f}")
        return "\n".join(lines)


def calculate_with_premiums(
    base_price: float,
    premiums: list[tuple[str, float]],
) -> PriceBreakdown:
    """Sum ``base_price`` plus cumulative percentage premiums.

    ``premiums`` is a list of (label, percent) pairs. The premium is applied as
    a percentage of the (flat) base price and summed, matching the price list's
    description that premiums are cumulative (added together) on top of base.
    """
    total_premium = 0.0
    for _, percent in premiums:
        total_premium += base_price * (percent / 100.0)

    breakdown = PriceBreakdown(
        base_price=base_price,
        premium_lines=list(premiums),
        final_total=base_price + total_premium,
    )
    return breakdown
