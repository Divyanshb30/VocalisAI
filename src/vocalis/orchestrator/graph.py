"""The case workflow as a LangGraph graph. See docs/adr/0006-langgraph-for-case-workflow.md.

    read document ─► assess rights ─► draft mandate ─► ⏸ passenger approves ─► place call ─► report
                                                              │ declined
                                                              └──────────────► END

The approval step is a LangGraph ``interrupt``: the graph checkpoints and waits, possibly for
a long time, until the passenger answers. The call itself runs in Pipecat, not in the graph.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from vocalis.calls import place_simulated_call, summary_text
from vocalis.cases import default_mandate, save_case, scenario_for_case
from vocalis.core.models import Case, Mandate
from vocalis.rights.engine import assess
from vocalis.simair.scenario import Persona


class CaseState(TypedDict, total=False):
    document_path: str
    case: dict[str, Any]
    conflicts: list[dict[str, Any]]
    entitlements: list[str]
    mandate: dict[str, Any]
    approved: bool
    rep_persona: str
    offline: bool
    report: dict[str, Any]
    summary: str


async def read_document(state: CaseState) -> CaseState:
    if state.get("case"):
        return {}
    from vocalis.docintel.extract import extract, to_case

    result = await extract(state["document_path"])
    return {
        "case": to_case(result.fields).model_dump(mode="json"),
        "conflicts": [c.__dict__ for c in result.conflicts],
    }


def assess_rights(state: CaseState) -> CaseState:
    case = Case.model_validate(state["case"])
    return {"entitlements": assess(case).summary_lines()}


def draft_mandate(state: CaseState) -> CaseState:
    if state.get("mandate"):
        return {}
    case = Case.model_validate(state["case"])
    return {"mandate": default_mandate(assess(case)).model_dump(mode="json")}


def passenger_approval(state: CaseState) -> CaseState:
    """Pause until the passenger approves (or edits) the mandate."""
    answer = interrupt(
        {
            "question": "Approve this mandate before the call?",
            "case": state["case"],
            "conflicts_to_confirm": state.get("conflicts", []),
            "entitlements": state["entitlements"],
            "mandate": state["mandate"],
        }
    )
    if isinstance(answer, dict):
        update: CaseState = {"approved": bool(answer.get("approved"))}
        if answer.get("mandate"):
            update["mandate"] = Mandate.model_validate(answer["mandate"]).model_dump(mode="json")
        return update
    return {"approved": bool(answer)}


async def place_call(state: CaseState) -> CaseState:
    case = Case.model_validate(state["case"])
    mandate = Mandate.model_validate(state["mandate"])
    save_case(case, mandate)
    scenario = scenario_for_case(case, mandate, Persona(state.get("rep_persona", "stonewaller")))
    report = await place_simulated_call(scenario, offline=state.get("offline", False))
    return {"report": report, "summary": summary_text(report)}


def after_approval(state: CaseState) -> str:
    return "place_call" if state.get("approved") else END


def build_graph(checkpointer: Any | None = None):  # type: ignore[no-untyped-def]
    g = StateGraph(CaseState)
    g.add_node("read_document", read_document)
    g.add_node("assess_rights", assess_rights)
    g.add_node("draft_mandate", draft_mandate)
    g.add_node("passenger_approval", passenger_approval)
    g.add_node("place_call", place_call)
    g.add_edge(START, "read_document")
    g.add_edge("read_document", "assess_rights")
    g.add_edge("assess_rights", "draft_mandate")
    g.add_edge("draft_mandate", "passenger_approval")
    g.add_conditional_edges("passenger_approval", after_approval, ["place_call", END])
    g.add_edge("place_call", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
