"""Everything the agent knows going into a call, rendered compactly for small context windows."""

from __future__ import annotations

from dataclasses import dataclass, field

from vocalis.core.geo import airport, carrier_name
from vocalis.core.models import Case, DisruptionType, Mandate, OutcomeType
from vocalis.guards.output_guard import OutputGuard
from vocalis.guards.vault import Vault
from vocalis.rights.models import RightsAssessment

NATO = {
    "A": "Alpha",
    "B": "Bravo",
    "C": "Charlie",
    "D": "Delta",
    "E": "Echo",
    "F": "Foxtrot",
    "G": "Golf",
    "H": "Hotel",
    "I": "India",
    "J": "Juliett",
    "K": "Kilo",
    "L": "Lima",
    "M": "Mike",
    "N": "November",
    "O": "Oscar",
    "P": "Papa",
    "Q": "Quebec",
    "R": "Romeo",
    "S": "Sierra",
    "T": "Tango",
    "U": "Uniform",
    "V": "Victor",
    "W": "Whiskey",
    "X": "X-ray",
    "Y": "Yankee",
    "Z": "Zulu",
    "0": "Zero",
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
}

OUTCOME_WORDS = {
    OutcomeType.CASH_REFUND: "a full cash refund to the original payment method",
    OutcomeType.COMPENSATION: "the compensation owed",
    OutcomeType.REFUND_AND_COMPENSATION: "a full refund plus the compensation owed",
    OutcomeType.REBOOKING: "rebooking on the next suitable flight at no cost",
    OutcomeType.VOUCHER: "a travel voucher",
    OutcomeType.CALLBACK: "a callback",
    OutcomeType.NOTHING: "nothing",
}


def nato(code: str) -> str:
    return " ".join(NATO.get(ch.upper(), ch) for ch in code if not ch.isspace())


@dataclass
class Briefing:
    case: Case
    rights: RightsAssessment
    mandate: Mandate
    vault: Vault
    facts: str = field(init=False)
    entitlements: str = field(init=False)
    mandate_text: str = field(init=False)

    def __post_init__(self) -> None:
        c = self.case
        seg = c.first_segment
        d = c.disruption
        o, dst = airport(seg.origin), airport(c.last_segment.destination)
        lines = [
            f"Passenger: {c.passenger.full_name}",
            f"Booking reference: {c.booking_reference} (spell it as: {nato(c.booking_reference)})",
            f"Flight: {carrier_name(seg.carrier)} {seg.designator}, {o.city} ({seg.origin}) to {dst.city} ({dst.iata}), "
            f"scheduled {seg.scheduled_departure:%d %B %Y at %H:%M}",
        ]
        if d.type is not DisruptionType.NONE:
            detail = d.type.value.replace("_", " ")
            if d.notice_hours is not None:
                detail += f", notified {d.notice_hours:.0f} hours before departure"
            if d.arrival_delay_minutes:
                detail += f", arrived {d.arrival_delay_minutes // 60}h{d.arrival_delay_minutes % 60:02d} late"
            if d.reason_given:
                detail += f", reason given: {d.reason_given}"
            lines.append(f"What happened: {detail}")
        if c.fare:
            lines.append(f"Fare paid: {c.fare.total:,.0f} {c.fare.currency}")
        if c.booking_email:
            lines.append(f"Booking email: {c.booking_email}")
        self.facts = "\n".join(lines)
        ents = self.rights.summary_lines()
        self.entitlements = (
            "\n".join(f"- {e}" for e in ents)
            or "- No regulation clearly applies; rely on the airline's own policy."
        )
        m = self.mandate
        acc = ", ".join(OUTCOME_WORDS[o] for o in m.acceptable) or "nothing else"
        forb = ", ".join(OUTCOME_WORDS[o] for o in m.forbidden) or "nothing"
        self.mandate_text = (
            f"Target: {OUTCOME_WORDS[m.target]}. Acceptable fallback: {acc}. "
            f"Do NOT accept: {forb}."
            + (f" Minimum amount: {m.min_amount} {m.currency}." if m.min_amount else "")
            + (f" Notes: {m.notes}" if m.notes else "")
        )

    def output_guard(self, enabled: bool = True) -> OutputGuard:
        g = OutputGuard(secrets=self.vault.secret_values(), enabled=enabled, mandate=self.mandate)
        allowed = [self.facts, self.entitlements, self.mandate_text]
        g.allow_text("\n".join(allowed))
        for v in self.vault.allowed_values():
            g.allow_text(v)
        g.secrets = {k: v for k, v in g.secrets.items() if v not in self.vault.allowed_values()}
        return g

    def disclosure(self, rep_name: str | None = None) -> str:
        hello = f"Hi {rep_name}," if rep_name else "Hi,"
        seg = self.case.first_segment
        return (
            f"{hello} I'm an AI assistant calling on behalf of the passenger, {self.case.passenger.full_name}, "
            f"and this call may be recorded. I'm calling about booking reference "
            f"{nato(self.case.booking_reference)}, flight {seg.designator}."
        )
