"""Flags on what the other side of the call is asking for.

Rules are fast and run on every rep utterance; they drive refusals and handoffs.
"""

from __future__ import annotations

import re
from enum import StrEnum


class Flag(StrEnum):
    OTP = "otp"
    PAYMENT = "payment"
    IDENTITY = "identity"
    PHONE = "phone"
    DOB = "dob"
    PROMPT_INJECTION = "prompt_injection"
    HUMAN_CHECK = "human_check"
    COMMITMENT = "commitment"


_PATTERNS: dict[Flag, re.Pattern[str]] = {
    Flag.OTP: re.compile(
        r"\b(otp|one[- ]?time (pass(word|code)?|code|pin)|verification code|security code|"
        r"(code|pin) (that |which )?(we|i|was)?\s*(just )?(sent|texted|messaged)|"
        r"\d[- ]digit code|code on (your|the) (phone|mobile))\b",
        re.I,
    ),
    Flag.PAYMENT: re.compile(
        r"\b(card number|credit card|debit card|cvv|cvc|expiry|expiration date|"
        r"(make|take|process) (a |the )?payment|pay (a |the )?(fee|charge|difference)|"
        r"bank (account|details)|upi|sort code|iban)\b",
        re.I,
    ),
    Flag.IDENTITY: re.compile(
        r"\b(passport( number)?|(?<!boarding )pass (number|no\.?)|id (number|card)|travel document|"
        r"aadhaar|national id|emirates id|security question|"
        r"mother'?s maiden|verify (your|the passenger'?s|their) identity|"
        r"last (four|4) (digits )?of (the |your )?card|billing address|password)\b",
        re.I,
    ),
    Flag.PHONE: re.compile(r"\b(phone|mobile|contact|cell) (number|no\.?)\b", re.I),
    Flag.DOB: re.compile(r"\b(date of birth|dob|birth ?date|born on)\b", re.I),
    Flag.PROMPT_INJECTION: re.compile(
        r"(ignore (all |any |your |the )?(previous |prior |above )?(instructions|rules|prompt)|"
        r"system prompt|developer mode|you are now|new instructions|repeat (your|the) instructions|"
        r"disregard (your|the)|act as (an? )?(unrestricted|different)|jailbreak|"
        r"reveal (your|the) (prompt|instructions|data))",
        re.I,
    ),
    Flag.HUMAN_CHECK: re.compile(
        r"\b(are|is) (you|this|it) (a |an )?(robot|bot|human|real person|machine|ai|computer|recording)\b|"
        r"\bam i (talking|speaking) (to|with) (a |an )?(real )?(robot|bot|human|person|machine|ai)\b",
        re.I,
    ),
    Flag.COMMITMENT: re.compile(
        r"\b(full and final|waive (your|the|any) (right|claim)s?|agree to (the |these )?terms|"
        r"sign (the |a )?(form|waiver|release)|no further claims)\b",
        re.I,
    ),
}

# Flags that must be handled by the passenger in person; waiving a claim is the passenger's decision too.
HANDOFF_FLAGS = frozenset({Flag.OTP, Flag.PAYMENT, Flag.IDENTITY, Flag.COMMITMENT})


def scan(text: str) -> set[Flag]:
    return {flag for flag, pat in _PATTERNS.items() if pat.search(text)}


def requires_handoff(flags: set[Flag], shareable_keys: list[str]) -> bool:
    if flags & HANDOFF_FLAGS:
        return True
    if Flag.PHONE in flags and "phone" not in shareable_keys:
        return True
    return Flag.DOB in flags and "date_of_birth" not in shareable_keys
