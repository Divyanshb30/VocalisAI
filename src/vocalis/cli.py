"""`vocalis` command line.

vocalis scenarios                                  list SimAir scenarios
vocalis call <scenario_id> [--offline]             run one simulated call, print transcript
vocalis case --scenario-case <id> | --document P   run the LangGraph case workflow
vocalis rights --airline 6E --origin DEL ...       what is the passenger owed?
vocalis mcp rights|calls [--http PORT]             start an MCP server
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from loguru import logger

SCENARIO_DIR = Path("evals/scenarios")


def _quiet(verbose: bool) -> None:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")


def cmd_scenarios(_: argparse.Namespace) -> None:
    from vocalis.simair.scenario import load_scenarios

    for s in load_scenarios(SCENARIO_DIR):
        print(f"{s.id:<55} {s.title}")


def cmd_call(a: argparse.Namespace) -> None:
    from vocalis.calls import place_simulated_call, summary_text, transcript_text
    from vocalis.simair.scenario import load_scenarios

    scenario = {s.id: s for s in load_scenarios(SCENARIO_DIR)}[a.scenario_id]
    report = asyncio.run(place_simulated_call(scenario, offline=a.offline, seed=a.seed))
    print(transcript_text(report))
    print("\n" + summary_text(report))


def cmd_case(a: argparse.Namespace) -> None:
    from langgraph.types import Command

    from vocalis.orchestrator.graph import build_graph
    from vocalis.simair.scenario import load_scenarios

    graph = build_graph()
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    state: dict = {"rep_persona": a.persona, "offline": a.offline}
    if a.document:
        state["document_path"] = a.document
    else:
        sc = {s.id: s for s in load_scenarios(SCENARIO_DIR)}[a.scenario_case]
        state["case"] = sc.case.model_dump(mode="json")

    async def run() -> None:
        result = await graph.ainvoke(state, config)
        pending = result.get("__interrupt__")
        if pending:
            ask = pending[0].value
            c = ask["case"]
            print(
                f"\nCase: {c['passenger']['first_name']} {c['passenger']['last_name']}, booking {c['booking_reference']}, "
                f"{c['segments'][0]['carrier']}{c['segments'][0]['flight_number']} "
                f"{c['segments'][0]['origin']}-{c['segments'][-1]['destination']}, {c['disruption']['type']}"
            )
            for conflict in ask.get("conflicts_to_confirm", []):
                print(
                    f"  ! check {conflict['field']}: photo says {conflict['vision']!r}, barcode says {conflict['barcode']!r}"
                )
            print("Entitlements:")
            for e in ask["entitlements"]:
                print(f"  - {e}")
            m = ask["mandate"]
            print(f"Mandate: target {m['target']}, acceptable {m['acceptable']}, never {m['forbidden']}")
            approved = a.yes or input("\nApprove and place the call? [y/N] ").strip().lower() == "y"
            result = await graph.ainvoke(Command(resume={"approved": approved}), config)
        if result.get("summary"):
            print("\n" + result["summary"])
        else:
            print("\nNo call placed.")

    asyncio.run(run())


def cmd_rights(a: argparse.Namespace) -> None:
    from vocalis.mcp_servers.rights_server import check_entitlements

    fn = getattr(check_entitlements, "fn", check_entitlements)
    out = fn(
        airline=a.airline,
        flight_number=a.flight,
        origin=a.origin,
        destination=a.destination,
        departure=a.departure,
        arrival=a.arrival,
        disruption=a.disruption,
        notice_hours=a.notice_hours,
        departure_delay_minutes=a.delay,
        arrival_delay_minutes=a.arrival_delay,
        extraordinary_circumstances=a.extraordinary,
        fare_currency=a.currency,
        fare_total=a.fare,
        base_fare=a.base_fare,
        fuel_charge=a.fuel,
    )
    print(json.dumps(out, indent=1, ensure_ascii=False))


def cmd_mcp(a: argparse.Namespace) -> None:
    if a.server == "rights":
        from vocalis.mcp_servers.rights_server import mcp
    else:
        from vocalis.mcp_servers.calls_server import mcp
    if a.http:
        mcp.run(transport="http", host="127.0.0.1", port=a.http)
    else:
        mcp.run()


def main() -> None:
    ap = argparse.ArgumentParser(prog="vocalis")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scenarios").set_defaults(fn=cmd_scenarios)

    p = sub.add_parser("call", help="run one simulated call")
    p.add_argument("scenario_id")
    p.add_argument("--offline", action="store_true", help="scripted models, no API keys needed")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_call)

    p = sub.add_parser("case", help="run the LangGraph case workflow")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--document", help="photo of a boarding pass / cancellation notice (needs GOOGLE_API_KEY)"
    )
    src.add_argument("--scenario-case", help="use the case from a SimAir scenario")
    p.add_argument("--persona", default="stonewaller")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--yes", action="store_true", help="approve the mandate without asking")
    p.set_defaults(fn=cmd_case)

    p = sub.add_parser("rights", help="what is the passenger owed?")
    p.add_argument("--airline", required=True)
    p.add_argument("--flight", default="0")
    p.add_argument("--origin", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--departure", required=True, help="ISO datetime, e.g. 2026-10-12T09:30")
    p.add_argument("--arrival")
    p.add_argument("--disruption", default="cancellation")
    p.add_argument("--notice-hours", type=float)
    p.add_argument("--delay", type=int, help="departure delay, minutes")
    p.add_argument("--arrival-delay", type=int)
    p.add_argument("--extraordinary", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--currency")
    p.add_argument("--fare", type=float)
    p.add_argument("--base-fare", type=float)
    p.add_argument("--fuel", type=float)
    p.set_defaults(fn=cmd_rights)

    p = sub.add_parser("mcp", help="start an MCP server")
    p.add_argument("server", choices=["rights", "calls"])
    p.add_argument("--http", type=int, help="serve streamable HTTP on this port instead of stdio")
    p.set_defaults(fn=cmd_mcp)

    a = ap.parse_args()
    if a.cmd != "mcp":
        _quiet(a.verbose)
    a.fn(a)


if __name__ == "__main__":
    main()
