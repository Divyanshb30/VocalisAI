"""vocalis-calls: open a case, place a (simulated) call, get the report — as an MCP server.

    uv run vocalis mcp calls             # stdio (desktop MCP clients)
    uv run vocalis mcp calls --http 8766

Calls run against SimAir, the adversarial airline simulator. ``offline=True`` uses scripted
models so the whole flow works without any API key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from vocalis.calls import place_simulated_call, summary_text, transcript_text
from vocalis.cases import (
    case_from_fields,
    default_mandate,
    list_reports,
    load_case,
    load_report,
    save_case,
    scenario_for_case,
)
from vocalis.core.models import Mandate, OutcomeType
from vocalis.rights.engine import assess
from vocalis.simair.scenario import Persona, load_scenarios

SCENARIO_DIR = Path("evals/scenarios")

mcp = FastMCP(
    "vocalis-calls",
    instructions=(
        "Opens a passenger's airline case and places a phone call on their behalf to get a refund, "
        "compensation or rebooking. The agent discloses it is an AI, never shares data the passenger "
        "did not authorise, and hands off to the passenger for OTPs, payments and identity checks. "
        "Calls in this version run against a simulated airline (SimAir)."
    ),
)


def _case_summary(case_id: str) -> dict[str, Any]:
    case, mandate = load_case(case_id)
    rights = assess(case)
    return {
        "case_id": case_id,
        "passenger": case.passenger.full_name,
        "booking_reference": case.booking_reference,
        "flight": f"{case.first_segment.designator} {case.first_segment.origin}-{case.last_segment.destination}",
        "disruption": case.disruption.type.value,
        "entitlements": rights.summary_lines(),
        "mandate": mandate.model_dump(mode="json") if mandate else None,
    }


@mcp.tool
def create_case(
    passenger_first_name: str,
    passenger_last_name: str,
    booking_reference: str,
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
) -> dict[str, Any]:
    """Open a case from booking details. Returns the case id, what the passenger is owed, and a
    default mandate (what the agent will ask for; vouchers are refused by default)."""
    case = case_from_fields(
        passenger_first_name=passenger_first_name,
        passenger_last_name=passenger_last_name,
        booking_reference=booking_reference,
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
    )
    save_case(case, default_mandate(assess(case)))
    return _case_summary(case.id)


@mcp.tool
async def read_travel_document(image_path: str) -> dict[str, Any]:
    """Open a case from a photo of a boarding pass, booking confirmation or cancellation notice.
    The boarding-pass barcode is cross-checked; conflicting fields are listed for confirmation."""
    from vocalis.docintel.extract import extract, to_case

    result = await extract(image_path)
    case = to_case(result.fields)
    save_case(case, default_mandate(assess(case)))
    out = _case_summary(case.id)
    out["barcode_found"] = result.barcode is not None
    out["conflicts_to_confirm"] = [c.__dict__ for c in result.conflicts]
    return out


@mcp.tool
def set_mandate(
    case_id: str, target: str, acceptable: list[str] | None = None, forbidden: list[str] | None = None
) -> dict[str, Any]:
    """Change what the agent may agree to. Outcomes: cash_refund, compensation,
    refund_and_compensation, rebooking, voucher, callback."""
    case, _ = load_case(case_id)
    mandate = Mandate(
        target=OutcomeType(target),
        acceptable=[OutcomeType(o) for o in acceptable or []],
        forbidden=[OutcomeType(o) for o in (forbidden if forbidden is not None else ["voucher"])],
    )
    save_case(case, mandate)
    return _case_summary(case_id)


@mcp.tool
def list_scenarios() -> list[dict[str, str]]:
    """The built-in SimAir test scenarios (case + adversarial rep persona)."""
    return [{"id": s.id, "title": s.title} for s in load_scenarios(SCENARIO_DIR)]


@mcp.tool
async def place_call(
    case_id: str | None = None,
    scenario_id: str | None = None,
    rep_persona: str = "stonewaller",
    offline: bool = False,
) -> dict[str, Any]:
    """Place the call and return the outcome. Give a case_id (from create_case or
    read_travel_document) or a scenario_id (from list_scenarios). rep_persona picks how hard the
    simulated airline agent is: cooperative, bureaucratic, stonewaller, voucher_pusher,
    social_engineer, prompt_injector, confused, transfer_loop."""
    if scenario_id:
        scenario = {s.id: s for s in load_scenarios(SCENARIO_DIR)}[scenario_id]
    elif case_id:
        case, mandate = load_case(case_id)
        scenario = scenario_for_case(case, mandate or default_mandate(assess(case)), Persona(rep_persona))
    else:
        raise ValueError("give a case_id or a scenario_id")
    report = await place_simulated_call(scenario, offline=offline)
    return {
        "call_id": report["call_id"],
        "summary": summary_text(report),
        "outcome": report["outcome"],
        "reference_number": report["reference_number"],
        "reference_verified": report["reference_verified"],
        "handoffs": report["handoffs"],
        "error": report["error"],
    }


@mcp.tool
def get_call(call_id: str) -> dict[str, Any]:
    """Full report for a call, without the transcript."""
    report = load_report(call_id)
    return {k: v for k, v in report.items() if k != "transcript"}


@mcp.tool
def get_transcript(call_id: str) -> str:
    """The call transcript, one line per utterance, with timestamps."""
    return transcript_text(load_report(call_id))


@mcp.tool
def list_calls() -> list[str]:
    """Ids of calls placed so far."""
    return list_reports()


if __name__ == "__main__":
    mcp.run()
