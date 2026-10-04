"""Simulator ground truth and agent speech hygiene."""

import json
from pathlib import Path
from typing import Any

from vocalis.agent.processors import without_json
from vocalis.core.models import OutcomeType
from vocalis.llm.router import LLMResult
from vocalis.simair.rep import EVENT_LINES, SimRep, _norm, outcome_from_text
from vocalis.simair.scenario import EventAction, load_scenarios


def test_tool_call_text_is_never_spoken() -> None:
    said = 'Thank you for your assistance.{"call": "end_call", "arguments": {"summary": "Refund done."}}'
    assert without_json(said) == "Thank you for your assistance."
    # still streaming: the open object is held back, the sentence before it is not
    assert without_json('Great, thanks. {"name": "end_') == 'Great, thanks. {"name": "end_'
    assert without_json("No braces here.") == "No braces here."


class _Router:
    """Returns canned rep replies in order."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = replies
        self.calls = 0

    async def complete(self, models: Any, messages: list[dict[str, Any]], **kw: Any) -> LLMResult:
        out = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return LLMResult(json.dumps(out), "fake/rep", 0, 0, 0.0)


def _rep(replies: list[dict[str, Any]], scenario: str = "ae_ek_cancel_weather__prompt_injector") -> SimRep:
    sc = next(s for s in load_scenarios(Path("evals/scenarios")) if s.id == scenario)
    return SimRep(sc, _Router(replies), ["fake/rep"])  # type: ignore[arg-type]


async def test_rep_does_not_parrot_scripted_lines() -> None:
    inject = EVENT_LINES[EventAction.INJECT]
    rep = _rep(
        [{"say": inject, "action": "none"}, {"say": "Let me look at the refund for you.", "action": "none"}]
    )
    rep._scripted_lines.add(_norm(inject))
    turn = await rep.respond("Hello?")
    assert turn.say == "Let me look at the refund for you."


async def test_grant_outcome_follows_the_words() -> None:
    # labelled cash_refund, but the rep granted compensation
    rep = _rep(
        [
            {
                "say": "I've processed the £520 cash compensation for your delay.",
                "action": "resolve",
                "outcome": "cash_refund",
            }
        ]
    )
    turn = await rep.respond("Thanks")
    assert turn.outcome is OutcomeType.COMPENSATION and rep.granted is turn
    # resolved without any outcome
    rep = _rep([{"say": "I've processed a full refund to the original card.", "action": "resolve"}])
    turn = await rep.respond("Thanks")
    assert rep.granted is not None and rep.granted.outcome is OutcomeType.CASH_REFUND


def test_outcome_from_text() -> None:
    assert outcome_from_text("a full refund and £220 compensation") is OutcomeType.REFUND_AND_COMPENSATION
    assert outcome_from_text("we can rebook you on the next flight") is OutcomeType.REBOOKING
    assert outcome_from_text("let me check") is None
