"""Deterministic check on every agent utterance (and DTMF digit string) before TTS.

Blocks: Luhn-valid card numbers, unapproved long digit runs (OTP/phone/account-shaped),
passport-shaped IDs, unapproved e-mail addresses, and any canary/secret value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from vocalis.guards.normalize import compact, digit_runs, spoken_digits_to_numerals

SAFE_FALLBACK = (
    "I'm sorry, I'm not able to share that detail. The passenger can provide it directly."
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PASSPORT = re.compile(r"\b[A-Z][0-9]{7}\b")


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


@dataclass
class GuardVerdict:
    allowed: bool
    text: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class OutputGuard:
    """Configured per call with the values that are fine to say."""

    secrets: dict[str, str] = field(default_factory=dict)
    allowed_numbers: set[str] = field(default_factory=set)
    allowed_emails: set[str] = field(default_factory=set)
    max_free_digits: int = 3  # short numbers (menu choices, hours, small counts) are fine
    enabled: bool = True

    def allow_text(self, text: str) -> None:
        """Whitelist every number and e-mail that appears in approved T0 text."""
        self.allowed_numbers.update(digit_runs(text))
        self.allowed_emails.update(e.lower() for e in _EMAIL.findall(text))

    def check(self, text: str) -> GuardVerdict:
        if not self.enabled:
            return GuardVerdict(True, text)
        reasons: list[str] = []
        norm = spoken_digits_to_numerals(text)
        flat = compact(norm)

        for key, value in self.secrets.items():
            v = compact(value)
            if len(v) >= 3 and v in flat:
                reasons.append(f"secret:{key}")

        for run in re.findall(r"\d+", norm):
            if run in self.allowed_numbers or len(run) <= self.max_free_digits:
                continue
            if 13 <= len(run) <= 19 and luhn_ok(run):
                reasons.append("card_number")
            elif not any(run in a or a in run for a in self.allowed_numbers if len(a) >= 4):
                reasons.append(f"unapproved_number:{len(run)}d")

        if _PASSPORT.search(text):
            reasons.append("passport_like")
        for email in _EMAIL.findall(text):
            if email.lower() not in self.allowed_emails:
                reasons.append("unapproved_email")

        if reasons:
            return GuardVerdict(False, SAFE_FALLBACK, sorted(set(reasons)))
        return GuardVerdict(True, text)

    def check_dtmf(self, digits: str) -> GuardVerdict:
        """Keypresses: menu choices are short; long strings must be approved numbers."""
        if not self.enabled or len(digits.strip("#*")) <= 2:
            return GuardVerdict(True, digits)
        core = digits.strip("#*")
        if core in self.allowed_numbers:
            return GuardVerdict(True, digits)
        return GuardVerdict(False, "", ["dtmf_unapproved_number"])
