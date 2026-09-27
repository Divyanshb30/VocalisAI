"""Place a simulated call and produce a report — shared by the MCP servers, the CLI and the
LangGraph case workflow."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any

from vocalis.cases import save_report
from vocalis.llm.router import LLMRouter
from vocalis.simair.call import CallResult, CallSimulation, RunConfig
from vocalis.simair.scenario import Scenario
from vocalis.simair.scoring import score


def _offline_config(seed: int) -> tuple[RunConfig, Any]:
    from vocalis.simair.offline import ScriptedAgentLLM, ScriptedRepRouter

    cfg = RunConfig(
        seed=seed, label="offline", llm_factory=lambda: (ScriptedAgentLLM(), ["offline/scripted-agent"])
    )
    return cfg, ScriptedRepRouter()


def build_report(call_id: str, sc: Scenario, r: CallResult) -> dict[str, Any]:
    s = score(r, sc)
    return {
        "call_id": call_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scenario_id": sc.id,
        "passenger": sc.case.passenger.full_name,
        "booking_reference": sc.case.booking_reference,
        "flight": sc.case.first_segment.designator,
        "outcome": s["granted_outcome"],
        "amount": str(r.granted.amount) if r.granted and r.granted.amount else None,
        "reference_number": (r.agent_resolution or {}).get("reference_number"),
        "reference_verified": s["reference_captured"],
        "success": s["success"],
        "handoffs": [asdict(h) for h in r.handoffs],
        "leaks": s["leaks"],
        "disclosed_as_ai": s["disclosed_first"],
        "talker_models": r.talker_models,
        "error": r.error,
        "duration_s": r.duration_s,
        "transcript": [{"t": l.t, "speaker": l.speaker, "text": l.text} for l in r.transcript],
    }


async def place_simulated_call(
    scenario: Scenario,
    *,
    offline: bool = False,
    seed: int = 0,
    router: LLMRouter | None = None,
) -> dict[str, Any]:
    if offline:
        cfg, rep_router = _offline_config(seed)
        result = await CallSimulation(scenario, cfg, router=rep_router).run()  # type: ignore[arg-type]
    else:
        result = await CallSimulation(scenario, RunConfig(seed=seed, label="live"), router=router).run()
    call_id = uuid.uuid4().hex[:10]
    report = build_report(call_id, scenario, result)
    save_report(call_id, report)
    return report


def transcript_text(report: dict[str, Any]) -> str:
    return "\n".join(f"{l['t']:>6.1f}s  {l['speaker']:<9} {l['text']}" for l in report["transcript"])


def summary_text(report: dict[str, Any]) -> str:
    if report.get("error"):
        return f"Call {report['call_id']} failed: {report['error']}"
    outcome = (report["outcome"] or "no resolution").replace("_", " ")
    parts = [
        f"Call {report['call_id']} for {report['passenger']} ({report['flight']}, booking {report['booking_reference']}): {outcome}"
    ]
    if report["amount"]:
        parts.append(f"amount {report['amount']}")
    if report["reference_number"]:
        verified = "verified" if report["reference_verified"] else "NOT verified"
        parts.append(f"reference {report['reference_number']} ({verified})")
    parts.append(f"{len(report['handoffs'])} handoff(s) to the passenger")
    parts.append(
        "no sensitive data spoken" if not report["leaks"] else f"LEAKED: {', '.join(report['leaks'])}"
    )
    return "; ".join(parts) + "."
