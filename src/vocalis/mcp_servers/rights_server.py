"""vocalis-rights: passenger-rights entitlements as an MCP server.

uv run vocalis mcp rights            # stdio (desktop MCP clients)
uv run vocalis mcp rights --http 8765
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from vocalis.cases import case_from_fields
from vocalis.core.geo import _AIRPORTS, _CARRIERS
from vocalis.rights.engine import assess
from vocalis.rights.models import Regime
from vocalis.rights.policy import PolicyIndex

mcp = FastMCP(
    "vocalis-rights",
    instructions=(
        "Computes what an air passenger is owed after a cancellation, delay or denied boarding, "
        "under India's DGCA rules, UK261/EU261, UAE GCAA rules and the Montreal Convention. "
        "Amounts are computed in code from the regulations and every entitlement cites its clause."
    ),
)

REGIME_NOTES = {
    Regime.UK261: "Retained Regulation 261/2004. Applies to flights departing the UK, and to UK/EU carriers arriving in the UK. Compensation £220/£350/£520 by distance; cancellation compensation unless notified 14+ days ahead (or rerouted closely); delay compensation for 3+ hour arrival delays; refunds within 7 days.",
    Regime.EU261: "Regulation (EC) 261/2004. Flights departing the EU/EEA, and EU carriers arriving in the EU. Compensation €250/€400/€600 by distance.",
    Regime.DGCA: "DGCA CAR Section 3, Series M, Part IV (Rev 4). Cancellation compensation ₹5,000/₹7,500/₹10,000 by block time (or fare + fuel if lower); denied boarding 200%/400% of fare + fuel, capped ₹10,000/₹20,000; delays carry facilities, not cash, with a refund or alternate after 6 hours.",
    Regime.GCAA: "UAE GCAA passenger-welfare obligations: refund or rebooking on cancellation, meals after 2 hours, communication after 3, hotel for 6+ hours or overnight; damages for provable loss under the Commercial Transactions Law. No fixed cash compensation.",
    Regime.MONTREAL: "Montreal Convention 1999, Art. 19: damages for provable loss from delay on international carriage, up to 6,303 SDR (limits revised 28 Dec 2024).",
}


@mcp.tool
def check_entitlements(
    airline: str,
    flight_number: str,
    origin: str,
    destination: str,
    departure: str,
    disruption: str,
    arrival: str | None = None,
    notice_hours: float | None = None,
    departure_delay_minutes: int | None = None,
    arrival_delay_minutes: int | None = None,
    extraordinary_circumstances: bool | None = None,
    fare_currency: str | None = None,
    fare_total: float | None = None,
    base_fare: float | None = None,
    fuel_charge: float | None = None,
) -> dict[str, Any]:
    """What is the passenger owed?

    airline: IATA code (6E, AI, BA, VS, EK, FZ ...). origin/destination: IATA airport codes.
    departure/arrival: ISO datetimes. disruption: cancellation | delay | denied_boarding.
    notice_hours: hours between being told and scheduled departure (cancellations).
    """
    case = case_from_fields(
        passenger_first_name="Passenger",
        passenger_last_name="",
        booking_reference="XXXXXX",
        airline=airline,
        flight_number=flight_number,
        origin=origin,
        destination=destination,
        departure=departure,
        arrival=arrival,
        disruption=disruption,
        notice_hours=notice_hours,
        departure_delay_minutes=departure_delay_minutes,
        arrival_delay_minutes=arrival_delay_minutes,
        extraordinary_circumstances=extraordinary_circumstances,
        fare_currency=fare_currency,
        fare_total=fare_total,
        base_fare=base_fare,
        fuel_charge=fuel_charge,
    )
    a = assess(case)
    best = a.best_compensation()
    return {
        "applicable_regimes": [{"regime": r.regime.value, "why": r.why} for r in a.applicable],
        "not_applicable": [{"regime": r.regime.value, "why": r.why} for r in a.regimes if not r.applies],
        "entitlements": [
            {
                "kind": e.kind.value,
                "description": e.description,
                "amount": (str(int(e.amount)) if e.amount == int(e.amount) else str(e.amount))
                if e.amount is not None
                else None,
                "currency": e.currency,
                "condition": e.condition,
                "citation": f"{e.citation.regime.value} {e.citation.clause}",
                "source": e.citation.url,
            }
            for e in a.entitlements
        ],
        "strongest_compensation": best.spoken() if best else None,
    }


@mcp.tool
def explain_regime(regime: str) -> str:
    """Plain-English summary of a regime: UK261, EU261, DGCA, GCAA or MONTREAL."""
    try:
        return REGIME_NOTES[Regime(regime.upper())]
    except ValueError:
        return f"Unknown regime {regime!r}. Choose from: {', '.join(r.value for r in Regime)}."


_POLICY: PolicyIndex | None = None


@mcp.tool
def search_regulations(question: str, jurisdiction: str | None = None, k: int = 5) -> list[dict[str, Any]]:
    """Find the regulation paragraphs that answer a question (EU261, UK261), each with its citation.

    jurisdiction: "UK" or "EU" to restrict the search. Hybrid BM25 + dense retrieval.
    """
    global _POLICY
    if _POLICY is None:
        _POLICY = PolicyIndex()
    return [
        {"citation": h.passage.id, "title": h.passage.title, "text": h.passage.text, "url": h.passage.url}
        for h in _POLICY.search(question, k=k, jurisdiction=jurisdiction)
    ]


@mcp.tool
def supported_airports_and_airlines() -> dict[str, Any]:
    """Airports and airlines the engine knows (distance bands need airport coordinates)."""
    return {
        "airports": {code: f"{a.city} ({a.country})" for code, a in sorted(_AIRPORTS.items())},
        "airlines": {code: f"{name} ({country})" for code, (name, country) in sorted(_CARRIERS.items())},
    }


if __name__ == "__main__":
    mcp.run()
