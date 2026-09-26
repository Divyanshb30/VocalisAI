"""Case store, default mandates, and turning any case into a SimAir call setup.

Cases and call reports are JSON files under ``.cache/``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from vocalis.core.models import Case, Mandate, OutcomeType
from vocalis.rights.models import EntitlementKind, RightsAssessment
from vocalis.simair.scenario import Expected, Persona, RepConfig, Scenario

STORE = Path(".cache")
_ALNUM = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def default_mandate(rights: RightsAssessment) -> Mandate:
    """What to ask for, given what the passenger is owed. Vouchers are never accepted by default."""
    kinds = {e.kind for e in rights.entitlements}
    has_comp = EntitlementKind.COMPENSATION in kinds
    has_refund = EntitlementKind.REFUND in kinds
    if has_comp and has_refund:
        return Mandate(
            target=OutcomeType.REFUND_AND_COMPENSATION,
            acceptable=[OutcomeType.CASH_REFUND, OutcomeType.REBOOKING],
        )
    if has_comp:
        return Mandate(target=OutcomeType.COMPENSATION)
    if has_refund:
        return Mandate(target=OutcomeType.CASH_REFUND, acceptable=[OutcomeType.REBOOKING])
    return Mandate(target=OutcomeType.REBOOKING, acceptable=[OutcomeType.CALLBACK])


def scenario_for_case(
    case: Case,
    mandate: Mandate,
    persona: Persona = Persona.COOPERATIVE,
    allow_share: list[str] | None = None,
    seed: int = 0,
) -> Scenario:
    """A SimAir setup for any case: the rep grants the mandate's target if the caller asks well."""
    rng = random.Random(seed)
    return Scenario(
        id=f"case_{case.id}__{persona.value}",
        title=f"{case.airline} {case.first_segment.designator} ({persona.value} rep)",
        jurisdiction="",
        case=case,
        mandate=mandate,
        allow_share=allow_share or [],
        rep=RepConfig(
            persona=persona,
            hidden_policy=(
                "Follow the applicable passenger-rights regulation. Offer a travel voucher first; "
                "grant what the regulation requires when the caller asks clearly and cites it."
            ),
            max_concession=mandate.target,
            concede_when="the caller declines the voucher and asks clearly for what the passenger is owed",
            reference_number="".join(rng.choice(_ALNUM) for _ in range(6)),
        ),
        expected=Expected(success_outcomes=[mandate.target, *mandate.acceptable]),
    )


def save_case(case: Case, mandate: Mandate | None = None) -> Path:
    path = STORE / "cases" / f"{case.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"case": case.model_dump(mode="json")}
    if mandate:
        payload["mandate"] = mandate.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def load_case(case_id: str) -> tuple[Case, Mandate | None]:
    data = json.loads((STORE / "cases" / f"{case_id}.json").read_text(encoding="utf-8"))
    mandate = Mandate.model_validate(data["mandate"]) if "mandate" in data else None
    return Case.model_validate(data["case"]), mandate


def save_report(call_id: str, report: dict[str, Any]) -> Path:
    path = STORE / "calls" / f"{call_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return path


def load_report(call_id: str) -> dict[str, Any]:
    return json.loads((STORE / "calls" / f"{call_id}.json").read_text(encoding="utf-8"))


def list_reports() -> list[str]:
    folder = STORE / "calls"
    return sorted(p.stem for p in folder.glob("*.json")) if folder.exists() else []
