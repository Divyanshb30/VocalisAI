"""Late-binding secrets. See docs/adr/0004-late-binding-secrets.md.

The LLM never sees T1 or T2 values. It may write a placeholder such as ⟦phone⟧; the
renderer substitutes the real value only when the passenger allowed that key for this
case. Anything else is blocked, and the caller is told a handoff is needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from vocalis.core.models import DataTier, VaultEntry

# Accept the canonical ⟦key⟧ plus the ASCII fallbacks models sometimes produce.
PLACEHOLDER = re.compile(r"⟦\s*([a-z_]+)\s*⟧|\[\[\s*([a-z_]+)\s*\]\]|\{\{\s*([a-z_]+)\s*\}\}")


@dataclass
class RenderResult:
    text: str
    shared: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)

    @property
    def needs_handoff(self) -> bool:
        return bool(self.blocked)


class Vault:
    def __init__(self, entries: list[VaultEntry] | None = None) -> None:
        self._entries = {e.key: e for e in entries or []}

    def add(self, entry: VaultEntry) -> None:
        self._entries[entry.key] = entry

    def get(self, key: str) -> VaultEntry | None:
        return self._entries.get(key)

    def shareable_keys(self) -> list[str]:
        """T1 keys the passenger allowed; these are the only placeholders the model may use."""
        return sorted(
            k for k, e in self._entries.items() if e.tier is DataTier.T1 and e.allowed
        )

    def withheld_keys(self) -> list[str]:
        return sorted(
            k
            for k, e in self._entries.items()
            if e.tier is DataTier.T2 or (e.tier is DataTier.T1 and not e.allowed)
        )

    def secret_values(self) -> dict[str, str]:
        """Every value that must never be spoken unless explicitly shared (T1 and T2)."""
        return {
            k: e.value.get_secret_value()
            for k, e in self._entries.items()
            if e.tier in (DataTier.T1, DataTier.T2)
        }

    def allowed_values(self) -> set[str]:
        return {
            e.value.get_secret_value()
            for e in self._entries.values()
            if e.tier is DataTier.T1 and e.allowed
        }

    def render(self, text: str) -> RenderResult:
        result = RenderResult(text=text)

        def sub(match: re.Match[str]) -> str:
            key = next(g for g in match.groups() if g)
            entry = self._entries.get(key)
            if entry is not None and entry.tier is DataTier.T1 and entry.allowed:
                result.shared.append(key)
                return entry.value.get_secret_value()
            result.blocked.append(key)
            return "that detail"

        result.text = PLACEHOLDER.sub(sub, text)
        return result

    def prompt_hint(self) -> str:
        """What the model is told about data it cannot see."""
        shareable = self.shareable_keys()
        withheld = self.withheld_keys()
        parts = []
        if shareable:
            joined = ", ".join(f"⟦{k}⟧" for k in shareable)
            parts.append(
                f"If the representative needs one of these, write the placeholder exactly and it "
                f"will be filled in: {joined}. You do not know these values."
            )
        if withheld:
            parts.append(
                "You are NOT authorised to share: "
                + ", ".join(withheld)
                + ". If asked for any of them, or for an OTP, card details, passport or payment, "
                "call request_handoff so the passenger can join the call."
            )
        return " ".join(parts)
