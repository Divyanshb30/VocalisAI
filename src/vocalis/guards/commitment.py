"""Stop the agent agreeing out loud to an offer the passenger has not authorised.

The rep's latest words are checked for an offer; if the mandate does not permit it (and the passenger
has not approved it), an agent sentence that accepts it is blocked before it is spoken.
"""

from __future__ import annotations

import re

from vocalis.core.models import Mandate, OutcomeType

_OFFER_CUE = re.compile(
    r"\b(offer|can (issue|give|provide|do|arrange|process)|best (i|we) can|shall i|should i|would you like|"
    r"would you accept|how about|instead|only option|all (i|we) can|what i can do|i('ll| will| can) (issue|put|process|arrange|give|provide))\b",
    re.I,
)
_DENIAL = re.compile(
    r"\b(no|not|cannot|can't|unable|don't|won't|declin\w*|forgo|waive|instead of|rather than)\b[^.?!]{0,20}$",
    re.I,
)
_ACCEPT_START = re.compile(
    r"^\W*(yes|yeah|yep|sure|okay|ok|alright|all right|fine|great|perfect|deal|agreed|sounds good|"
    r"that works|that'?s fine|that'?s great|that'?s acceptable|go ahead|please (do|go ahead|issue|process|proceed))\b",
    re.I,
)
_ACCEPT_VERB = re.compile(
    r"\b(i|we)('ll| will|'d| would| can)?\s+(happily |gladly |be happy to )?(accept|take|go with|agree to)\b|"
    r"\b(please|could you|can you) (\w+ )?(issue|process|proceed with|go ahead with|arrange) (that|it|this)\b|"
    r"\bplease (go ahead|proceed)\b",
    re.I,
)
_RESIST = re.compile(
    r"\b(no|not|cannot|declin\w*|refus\w*|rather|instead|but|however|unfortunately|entitled|"
    r"unable|only|insist\w*)\b|n't\b",
    re.I,
)
_OUTCOME_WORDS: list[tuple[OutcomeType, re.Pattern[str]]] = [
    (
        OutcomeType.VOUCHER,
        re.compile(r"\b(voucher|credit shell|travel credit|future credit|e-?credit)s?\b", re.I),
    ),
    (OutcomeType.CALLBACK, re.compile(r"\b(call (you|them|her|him) back|callback|call back)\b", re.I)),
    (OutcomeType.REBOOKING, re.compile(r"\b(rebook\w*|new flight|alternative flight|next flight)\b", re.I)),
    (OutcomeType.COMPENSATION, re.compile(r"\bcompensation\b", re.I)),
    (OutcomeType.CASH_REFUND, re.compile(r"\brefund\w*\b", re.I)),
]
_DESCRIBE = {
    OutcomeType.VOUCHER: "a voucher",
    OutcomeType.CALLBACK: "a callback",
    OutcomeType.REBOOKING: "a rebooking",
    OutcomeType.COMPENSATION: "compensation on its own",
    OutcomeType.CASH_REFUND: "a refund on its own",
    OutcomeType.REFUND_AND_COMPENSATION: "a refund and compensation",
    OutcomeType.NOTHING: "that",
}


def fallback(offer: OutcomeType) -> str:
    return f"To be clear, the passenger has not agreed to {_DESCRIBE[offer]}."


def _named(sentence: str) -> list[tuple[int, OutcomeType]]:
    hits = sorted((m.start(), o) for o, pat in _OUTCOME_WORDS for m in pat.finditer(sentence))
    money = (OutcomeType.CASH_REFUND, OutcomeType.COMPENSATION)
    if set(money) <= {o for _, o in hits}:  # "a refund plus compensation"
        first = min(s for s, o in hits if o in money)
        hits = sorted(
            [*((s, o) for s, o in hits if o not in money), (first, OutcomeType.REFUND_AND_COMPENSATION)]
        )
    return hits


def offers_in(rep_text: str) -> set[OutcomeType]:
    """What a plain "yes" would agree to: the outcomes in the rep's last offer this turn
    ("We usually give a voucher first. But I can process a refund, is that okay?" -> refund)."""
    offered: set[OutcomeType] = set()
    for sentence in re.split(r"(?<=[.?!])\s+", rep_text):
        if not (_OFFER_CUE.search(sentence) or sentence.rstrip().endswith("?")):
            continue
        named = {o for start, o in _named(sentence) if not _DENIAL.search(sentence[:start])}
        if named:
            offered = named
    return offered


def accepts(agent_sentence: str, offer: OutcomeType, said_before: str = "") -> bool:
    """The sentence agrees to ``offer``: an acceptance that names it, or that names nothing while the
    agent has not asked for anything else earlier in the same reply ("We'd like the refund. Could you
    process that?" asks for the refund, not the voucher on the table)."""
    if _RESIST.search(agent_sentence):
        return False
    named = {o for _, o in _named(agent_sentence)}
    if named and offer not in named:
        return False
    if not named and any(o != offer for _, o in _named(said_before)):
        return False
    return bool(_ACCEPT_START.search(agent_sentence) or _ACCEPT_VERB.search(agent_sentence))


def unauthorised(offers: set[OutcomeType], mandate: Mandate, approved: set[OutcomeType]) -> set[OutcomeType]:
    return {o for o in offers if o not in approved and not mandate.permits(o)}
