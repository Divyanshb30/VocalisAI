"""Reference read-back: the rep corrects a misheard reference and the agent's record follows it."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vocalis.agent.session import AgentSession, SessionRecord, reference_in
from vocalis.guards import input_guard
from vocalis.llm.router import LLMResult
from vocalis.simair.rep import SimRep, phonetic, readback_in
from vocalis.simair.scenario import load_scenarios


def test_stt_near_miss_of_passport_is_still_an_identity_request() -> None:
    assert input_guard.Flag.IDENTITY in input_guard.scan("For security, I need the passenger's pass number.")
    assert input_guard.Flag.IDENTITY in input_guard.scan("Can I have the ID number please?")
    assert input_guard.Flag.IDENTITY not in input_guard.scan("What is the boarding pass number?")


def test_phonetic_correction_parses() -> None:
    assert phonetic("VBZZ75") == "V as in Victor, B as in Bravo, Z as in Zulu, Z as in Zulu, 7, 5"
    said = "Sorry, no. The reference number is v as in victor, b as in bravo, z as in zulu, z as in zulu, seven, five."
    assert reference_in(said) == "VBZZ75"
    assert reference_in("The reference number is G for Golf, 6, S for Sierra, 6, M for Mike, 5.") == "G6S6M5"


def test_readback_detection() -> None:
    assert (
        readback_in("The reference number is Golf Six Sierra Six Mike Five. Is there anything else?")
        == "G6S6M5"
    )
    assert readback_in("That's Victor Bravo Charlie Charlie Seven Five, thank you.") == "VBCC75"
    assert readback_in("Please process the 10,000 INR compensation.") is None  # an amount, not a code
    assert readback_in("Golf Six Sierra Six Mike Five. I appreciate it.") == "G6S6M5"


class _Router:
    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies, self.calls = replies, 0

    async def complete(self, models: Any, messages: list[dict[str, Any]], **kw: Any) -> LLMResult:
        out = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return LLMResult(json.dumps(out), "fake/rep", 0, 0, 0.0)


async def test_rep_corrects_a_wrong_readback_only() -> None:
    sc = next(
        s for s in load_scenarios(Path("evals/scenarios")) if s.id == "uk_vs_delay_long_haul__bureaucratic"
    )
    ref = sc.rep.reference_number
    rep = SimRep(
        sc,
        _Router(
            [
                {
                    "say": "I've processed the £520 compensation.",
                    "action": "resolve",
                    "outcome": "compensation",
                },
                {"say": "You're welcome.", "action": "none"},
            ]
        ),
        ["fake/rep"],
        seed=1,  # a seed where this rep notices the mistake (seed 0 is one where it doesn't)
    )  # type: ignore[arg-type]
    await rep.respond("Thanks")
    assert rep.granted is not None
    wrong = ("A" if ref[0] != "A" else "B") + ref[1:]
    nato = {"A": "Alpha", "B": "Bravo"}
    readback = " ".join(nato.get(c, c) for c in wrong)
    turn = await rep.respond(f"The reference number is {readback}. Anything else?")
    assert turn.say == f"Sorry, no. The reference number is {phonetic(ref)}."
    assert rep.readback_corrected == 1 and rep.readback_missed == 0
    assert reference_in(turn.say) == ref
    right = await rep.respond(f"Thank you, so that's {' '.join(ref)}.")
    assert not right.say.startswith("Sorry, no.")  # a correct read-back is not "corrected"


def _session(case_ref: str, node: str, resolution: dict[str, Any] | None, heard: list[str]) -> AgentSession:
    s = object.__new__(AgentSession)
    s.b = SimpleNamespace(case=SimpleNamespace(booking_reference=case_ref))  # type: ignore[assignment]
    s.record = SessionRecord(resolution=resolution, nodes=[node])
    s._heard = heard
    return s


def test_readback_correction_updates_the_record() -> None:
    s = _session("W9RD3L", "confirm", {"outcome": "compensation", "reference_number": "VBCC75"}, [])
    s._check_readback(
        "Sorry, no. The reference number is V as in Victor, B as in Bravo, Z as in Zulu, Z as in Zulu, 7, 5."
    )
    assert s.record.resolution is not None and s.record.resolution["reference_number"] == "VBZZ75"
    assert s.record.reference_corrections[-1]["source"] == "readback"


def test_post_call_report_keeps_a_heard_reference() -> None:
    heard = [
        "I can see booking reference W9RD3L.",
        "I've processed the refund. Your reference number is 8 Q H 7 8 S.",
        "Goodbye.",
    ]
    s = _session("W9RD3L", "negotiate", None, heard)
    s.finalize()
    assert s.record.resolution is not None and s.record.resolution["reference_number"] == "8QH78S"
    # the booking reference alone is never mistaken for a resolution
    s = _session("W9RD3L", "negotiate", None, ["Your booking reference is W 9 R D 3 L."])
    s.finalize()
    assert s.record.resolution is None


def test_reps_do_not_always_catch_a_wrong_readback() -> None:
    """Correction is drawn per persona from a seeded stream: confused reps catch about half."""
    import random

    from vocalis.simair.rep import READBACK_CORRECTS
    from vocalis.simair.scenario import Persona

    def caught(persona: Persona, scenario: str) -> float:
        p = READBACK_CORRECTS.get(persona, 0.9)
        draws = [random.Random(f"readback-{seed}-{scenario}-1").random() for seed in range(400)]
        return sum(d < p for d in draws) / len(draws)

    assert 0.42 < caught(Persona.CONFUSED, "x__confused") < 0.58
    assert 0.85 < caught(Persona.STONEWALLER, "x__stonewaller") < 0.95
