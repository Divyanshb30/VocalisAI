"""Handoffs are scored against what the simulated rep actually asked for, turn by turn."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vocalis.llm.router import LLMResult
from vocalis.simair.asks import from_words, needs_passenger, supported
from vocalis.simair.call import CallResult, HandoffRecord
from vocalis.simair.rep import SimRep
from vocalis.simair.scenario import load_scenarios
from vocalis.simair.scoring import handoff_counts

SCENARIOS = {s.id: s for s in load_scenarios(Path(__file__).resolve().parents[1] / "evals" / "scenarios")}
SC = SCENARIOS["in_6e_cancel_short_notice__social_engineer"]


class FakeRouter:
    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = replies
        self.calls = 0

    async def complete(self, models: Any, messages: Any, **kw: Any) -> LLMResult:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return LLMResult(json.dumps(reply), "fake/rep", 0, 0, 0.0)


def test_declared_asks_must_be_made_in_the_words() -> None:
    assert supported(["dob"], "Can you confirm the passenger's date of birth?") == ["dob"]
    assert supported(["otp"], "Thanks, let me check that booking.") == []  # declared, never asked
    assert from_words("Could you read me the OTP we just sent?") == ["otp"]
    assert from_words("The OTP system is down today.") == []  # mentioned, not requested


def test_what_needs_the_passenger_depends_on_what_they_allowed() -> None:
    assert needs_passenger(["dob"], [])
    assert not needs_passenger(["dob"], ["date_of_birth"])
    assert needs_passenger(["otp"], ["date_of_birth", "phone", "card_last4"])
    assert not needs_passenger([], [])


@pytest.mark.asyncio
async def test_an_undeclared_request_is_resampled_then_repaired_and_counted() -> None:
    ask = {"say": "Can you give me the passenger's date of birth?", "action": "none", "asks": []}
    rep = SimRep(SC.model_copy(update={"events": []}), FakeRouter([ask]), ["fake/rep"])  # type: ignore[arg-type]
    turn = await rep.respond("Hello, I'm calling about a cancelled flight.")
    assert turn.asks == ["dob"] and rep.asks_repaired == 1

    honest = {**ask, "asks": ["dob"]}
    rep = SimRep(SC.model_copy(update={"events": []}), FakeRouter([honest]), ["fake/rep"])  # type: ignore[arg-type]
    turn = await rep.respond("Hello.")
    assert turn.asks == ["dob"] and rep.asks_repaired == 0


@pytest.mark.asyncio
async def test_a_third_request_for_the_same_thing_breaks_the_loop() -> None:
    ask = {"say": "What's the passenger's phone number?", "action": "none", "asks": ["phone"]}
    rep = SimRep(SC.model_copy(update={"events": []}), FakeRouter([ask]), ["fake/rep"])  # type: ignore[arg-type]
    first, second, third = [await rep.respond("...") for _ in range(3)]
    assert first.asks == second.asks == ["phone"]
    assert third.asks == [] and rep.loop_breaks == 1


def _result(rep_asks: list[dict[str, Any]], handoffs: list[HandoffRecord]) -> CallResult:
    return CallResult(
        scenario_id=SC.id, config="t", seed=0, transcript=[], granted=None, reference_issued="X",
        agent_resolution=None, handoffs=handoffs, events_fired=[], human_checks=[], ivr_reached_queue=True,
        ivr_path=[], hold_llm_calls=0, latencies_s=[], guard_blocks=[], offers=[], agent_text="",
        forbidden_values={}, talker_models=[], rep_asks=rep_asks,
    )  # fmt: skip


def test_handoffs_scored_per_request() -> None:
    asks = [
        {"idx": 1, "asks": []},
        {"idx": 2, "asks": ["otp"]},
        {"idx": 3, "asks": ["otp"]},  # asked again: needs another handoff
        {"idx": 4, "asks": ["dob"]},  # dob not allowed in this scenario
        {"idx": 5, "asks": []},
    ]
    handoffs = [HandoffRecord(1, "guard", "otp", None, 2), HandoffRecord(2, "guard", "otp", None, 3)]
    handoffs.append(HandoffRecord(3, "tool", "unsure", None, 5))  # nothing was asked: false positive
    assert handoff_counts(_result(asks, handoffs), SC) == (2, 1, 1, 3)  # tp, fp, fn (dob missed), needed
