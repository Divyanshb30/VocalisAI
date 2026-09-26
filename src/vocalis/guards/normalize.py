"""Normalise spoken text so guards catch numbers however they are said.

"four four one seven", "double seven", "oh nine" and "4-4-1-7" all become digit runs.
"""

from __future__ import annotations

import re

_WORD_DIGITS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_MULT = {"double": 2, "triple": 3}
_TOKEN = re.compile(r"[a-z]+|\d+|[^\sa-z\d]")


def spoken_digits_to_numerals(text: str) -> str:
    """Rewrite spelled-out digit sequences as numerals; leaves other words alone."""
    tokens = _TOKEN.findall(text.lower())
    out: list[str] = []
    mult = 1
    for tok in tokens:
        if tok in _MULT:
            mult = _MULT[tok]
            continue
        if tok in _WORD_DIGITS and (tok not in ("o", "oh") or (out and out[-1].isdigit())):
            out.append(_WORD_DIGITS[tok] * mult)
        elif tok.isdigit():
            out.append(tok * mult if mult > 1 and len(tok) == 1 else tok)
        elif tok in "-.,/ " and out and out[-1].isdigit():
            continue  # separators inside a number: "4-4-1-7", "98 76"
        else:
            out.append(" " + tok + " ")
        mult = 1
    joined = "".join(out)
    return re.sub(r"\s+", " ", joined).strip()


def digit_runs(text: str, min_len: int = 1) -> list[str]:
    """All maximal digit runs after normalisation."""
    return [m for m in re.findall(r"\d+", spoken_digits_to_numerals(text)) if len(m) >= min_len]


def compact(text: str) -> str:
    """Lowercase alphanumerics only, for substring matching of codes like X7K2QB."""
    return re.sub(r"[^a-z0-9]", "", text.lower())
