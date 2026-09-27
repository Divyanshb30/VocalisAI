"""Result types for the rights engine."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field


class Regime(StrEnum):
    UK261 = "UK261"
    EU261 = "EU261"
    DGCA = "DGCA"
    GCAA = "GCAA"
    MONTREAL = "MONTREAL"


class EntitlementKind(StrEnum):
    REFUND = "refund"
    REROUTING = "rerouting"
    COMPENSATION = "compensation"
    CARE = "care"
    HOTEL = "hotel"
    DAMAGES_CLAIM = "damages_claim"


class Citation(BaseModel):
    regime: Regime
    clause: str
    url: str


class Entitlement(BaseModel):
    kind: EntitlementKind
    description: str
    amount: Decimal | None = None
    currency: str | None = None
    condition: str | None = Field(
        default=None, description="What must hold for this to be owed, if not certain"
    )
    citation: Citation

    def spoken(self) -> str:
        """Short, speakable form for the agent."""
        money = f" of {self.amount:,.0f} {self.currency}" if self.amount is not None else ""
        cond = f" ({self.condition})" if self.condition else ""
        return f"{self.description}{money}{cond}, under {self.citation.regime} {self.citation.clause}"


class RegimeAssessment(BaseModel):
    regime: Regime
    applies: bool
    why: str
    entitlements: list[Entitlement] = Field(default_factory=list)


class RightsAssessment(BaseModel):
    case_id: str
    regimes: list[RegimeAssessment]

    @property
    def applicable(self) -> list[RegimeAssessment]:
        return [r for r in self.regimes if r.applies]

    @property
    def entitlements(self) -> list[Entitlement]:
        return [e for r in self.applicable for e in r.entitlements]

    def best_compensation(self) -> Entitlement | None:
        comps = [
            e for e in self.entitlements if e.kind == EntitlementKind.COMPENSATION and e.amount is not None
        ]
        return max(comps, key=lambda e: e.amount or Decimal(0), default=None)

    def summary_lines(self) -> list[str]:
        return [e.spoken() for e in self.entitlements]
