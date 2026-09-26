"""IVR menu trees for the simulated airline hotline.

An IVR is a small state machine: each menu plays a prompt and maps keypresses (or
spoken keywords, for "say or press" menus) to the next menu. Reaching the queue ends
navigation and starts hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

QUEUE = "__queue__"
HANGUP = "__hangup__"


@dataclass
class Menu:
    prompt: str
    options: dict[str, str]  # key -> next menu id, QUEUE or HANGUP
    keywords: dict[str, str] = field(default_factory=dict)  # spoken word -> key


@dataclass
class IVRStep:
    speech: str
    done: bool = False  # reached the agent queue
    hung_up: bool = False
    invalid: bool = False


@dataclass
class IVR:
    menus: dict[str, Menu]
    start: str = "main"
    max_invalid: int = 3
    current: str = field(init=False)
    invalid_count: int = field(init=False, default=0)
    path: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.current = self.start

    def greeting(self) -> str:
        return self.menus[self.current].prompt

    def _goto(self, target: str) -> IVRStep:
        if target == QUEUE:
            return IVRStep("Please hold while we connect you to the next available agent.", done=True)
        if target == HANGUP:
            return IVRStep("Thank you for calling. Goodbye.", hung_up=True)
        self.current = target
        self.invalid_count = 0
        return IVRStep(self.menus[target].prompt)

    def press(self, digits: str) -> IVRStep:
        menu = self.menus[self.current]
        for key in digits:
            self.path.append(key)
            if key in menu.options:
                return self._goto(menu.options[key])
        return self._invalid()

    def say(self, text: str) -> IVRStep:
        menu = self.menus[self.current]
        lowered = text.lower()
        for word, key in menu.keywords.items():
            if word in lowered:
                self.path.append(f"say:{word}")
                return self._goto(menu.options[key])
        return self._invalid()

    def _invalid(self) -> IVRStep:
        self.invalid_count += 1
        if self.invalid_count >= self.max_invalid:
            return IVRStep("We're sorry, we didn't get that. Goodbye.", hung_up=True, invalid=True)
        return IVRStep("Sorry, that's not a valid option. " + self.menus[self.current].prompt, invalid=True)


def _preset(name: str, airline: str) -> dict[str, Menu]:
    if name == "simple":
        return {
            "main": Menu(
                f"Thank you for calling {airline}. For new bookings press 1. For an existing booking, "
                "cancellations or refunds press 2. For baggage press 3.",
                {"1": HANGUP, "2": "existing", "3": HANGUP},
                {"booking": "1", "refund": "2", "cancel": "2", "baggage": "3"},
            ),
            "existing": Menu(
                "For refunds and cancellations press 1. To change your flight press 2. "
                "To speak to an agent press 0.",
                {"1": QUEUE, "2": QUEUE, "0": QUEUE},
                {"refund": "1", "change": "2", "agent": "0", "representative": "0"},
            ),
        }
    if name == "deep":
        return {
            "main": Menu(
                f"Welcome to {airline}. Please note calls are recorded. For English press 1. "
                "For Hindi press 2.",
                {"1": "english", "2": "english"},
            ),
            "english": Menu(
                "For flight status press 1. For manage booking press 2. For special assistance press 4.",
                {"1": HANGUP, "2": "manage", "4": HANGUP},
                {"status": "1", "manage": "2", "booking": "2"},
            ),
            "manage": Menu(
                "To change a flight press 1. For cancellations, refunds and compensation press 3. "
                "To repeat this menu press star.",
                {"1": "change", "3": "refunds", "*": "manage"},
                {"change": "1", "refund": "3", "cancel": "3", "compensation": "3"},
            ),
            "change": Menu("Please visit our website to change your flight. Goodbye.", {}),
            "refunds": Menu(
                "If your flight was cancelled by the airline press 1. For all other refunds press 2.",
                {"1": QUEUE, "2": QUEUE},
                {"cancelled": "1", "other": "2"},
            ),
        }
    raise ValueError(f"unknown IVR preset {name}")


def build_ivr(preset: str, airline: str) -> IVR:
    return IVR(menus=_preset(preset, airline))
