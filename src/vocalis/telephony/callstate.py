"""Which state is the far end in: IVR menu, hold, live human, or voicemail?

S0 uses transcript heuristics. The audio classifier (fine-tuning track B) replaces this
with an acoustic model fused with the same cues; the interface stays the same.
"""

from __future__ import annotations

import re
from enum import StrEnum


class CallState(StrEnum):
    IVR = "ivr"
    HOLD = "hold"
    HUMAN = "human"
    VOICEMAIL = "voicemail"
    UNKNOWN = "unknown"


_IVR = re.compile(r"\b(press|for [a-z ]+ press|say or press|to repeat this menu|main menu)\b", re.I)
_HOLD = re.compile(
    r"(\[hold music\]|call is important|please (continue to )?hold|estimated wait|"
    r"all (of )?our (agents|advisors|representatives) are (currently )?busy|you are (caller )?number|"
    r"thank you for your patience|your call will be answered)",
    re.I,
)
_VOICEMAIL = re.compile(r"(leave (a|your) message|after the (tone|beep)|mailbox|not available to take)", re.I)
_HUMAN = re.compile(
    r"(my name is|this is [A-Z][a-z]+|speaking|how (can|may) i help|you'?re through to|"
    r"thanks? for holding|good (morning|afternoon|evening))",
    re.I,
)


def classify(text: str) -> CallState:
    if _VOICEMAIL.search(text):
        return CallState.VOICEMAIL
    human = bool(_HUMAN.search(text))
    if _HOLD.search(text) and not human:
        return CallState.HOLD
    if _IVR.search(text) and not human:
        return CallState.IVR
    if human:
        return CallState.HUMAN
    return CallState.UNKNOWN
