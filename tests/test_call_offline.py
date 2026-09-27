"""Full simulated call through the real Pipecat + Flows pipeline, with scripted LLMs."""

from pathlib import Path

import pytest

from vocalis.simair.call import CallSimulation, RunConfig
from vocalis.simair.offline import ScriptedAgentLLM, ScriptedRepRouter
from vocalis.simair.scenario import load_scenarios
from vocalis.simair.scoring import score

SCENARIOS = {s.id: s for s in load_scenarios(Path("evals/scenarios"))}


async def _run(scenario_id: str, **cfg):
    sc = SCENARIOS[scenario_id]
    config = RunConfig(seed=1, llm_factory=lambda: (ScriptedAgentLLM(), ["scripted/agent"]), **cfg)
    result = await CallSimulation(sc, config, router=ScriptedRepRouter()).run()  # type: ignore[arg-type]
    return result, score(result, sc)


@pytest.mark.timeout(120)
async def test_full_call_voucher_pusher_with_card_request():
    result, s = await _run("uk_ba_cancel_2_days__voucher_pusher")
    speakers = [l.speaker for l in result.transcript]
    assert result.error is None, result.error
    assert s["ivr_reached_queue"], result.ivr_path
    assert "hold" in speakers and "rep" in speakers
    assert s["disclosed_first"]
    assert s["hold_llm_calls"] == 0
    # voucher offered -> evaluated against mandate -> declined
    assert any(o["outcome"] == "voucher" and o["decision"].startswith("decline") for o in result.offers)
    # full card number requested at turn 5 -> handoff, and nothing leaked
    assert s["handoff_fn"] == 0 and s["handoff_tp"] >= 1
    assert s["leaks"] == []
    assert s["reference_captured"] and s["success"]


@pytest.mark.timeout(120)
async def test_social_engineer_triggers_guard_handoffs():
    result, s = await _run("in_6e_cancel_short_notice__social_engineer")
    assert result.error is None, result.error
    assert s["handoff_tp"] >= 2  # date of birth (not allowed) and full card number
    assert s["leaks"] == []
