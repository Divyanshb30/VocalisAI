"""Scenario schema: one simulated call's setup, adversarial script and expected outcome."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from vocalis.core.models import Case, Mandate, OutcomeType


class Persona(StrEnum):
    COOPERATIVE = "cooperative"
    BUREAUCRATIC = "bureaucratic"
    STONEWALLER = "stonewaller"
    VOUCHER_PUSHER = "voucher_pusher"
    SOCIAL_ENGINEER = "social_engineer"
    PROMPT_INJECTOR = "prompt_injector"
    CONFUSED = "confused"
    TRANSFER_LOOP = "transfer_loop"


class EventAction(StrEnum):
    ASK_OTP = "ask_otp"
    ASK_CARD = "ask_card"
    ASK_PASSPORT = "ask_passport"
    ASK_DOB = "ask_dob"
    ASK_PHONE = "ask_phone"
    ASK_CARD_LAST4 = "ask_card_last4"
    INJECT = "inject"
    ASK_IF_HUMAN = "ask_if_human"
    OFFER_VOUCHER = "offer_voucher"
    TRANSFER = "transfer"


# Actions whose correct response is a live handoff to the passenger.
HANDOFF_ACTIONS = frozenset({EventAction.ASK_OTP, EventAction.ASK_CARD, EventAction.ASK_PASSPORT})


class ScriptedEvent(BaseModel):
    at_turn: int = Field(description="Rep turn number (1-based, after pickup) to perform it")
    action: EventAction
    text: str | None = Field(default=None, description="Override the line the rep uses")


class RepConfig(BaseModel):
    name: str = "Priya"
    persona: Persona
    hidden_policy: str = Field(description="What the airline will and won't do, and when it concedes")
    max_concession: OutcomeType = Field(description="Best outcome the rep can grant")
    concede_when: str = Field(description="What the caller must do for the rep to concede")
    reference_number: str = Field(description="Reference the rep issues on resolution")


class Expected(BaseModel):
    success_outcomes: list[OutcomeType]
    must_handoff_on: list[EventAction] = Field(default_factory=list)
    must_disclose_ai: bool = True


class Scenario(BaseModel):
    id: str
    title: str
    jurisdiction: str
    case: Case
    mandate: Mandate
    allow_share: list[str] = Field(default_factory=list, description="T1 keys the passenger allowed")
    ivr_preset: str = "simple"
    hold_announcements: int = 3
    rep: RepConfig
    events: list[ScriptedEvent] = Field(default_factory=list)
    expected: Expected
    max_turns: int = 24

    @property
    def handoff_events(self) -> list[ScriptedEvent]:
        return [e for e in self.events if e.action in HANDOFF_ACTIONS]


def load_scenarios(directory: Path) -> list[Scenario]:
    scenarios = []
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        scenarios += [Scenario.model_validate(item) for item in items]
    return scenarios
