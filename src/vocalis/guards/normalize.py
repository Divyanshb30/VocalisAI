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


# Spelled-out numbers beyond single digits: "nineteen eighty-five", "twelfth", "two thousand and two".
_UNITS = {
    w: i
    for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])
}
_TEENS = {
    w: 10 + i
    for i, w in enumerate(
        [
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
    )
}
_TENS = {
    w: 20 + 10 * i
    for i, w in enumerate(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"])
}
_ORD_UNITS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
}  # fmt: skip
_ORD_TEENS = {
    "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}  # fmt: skip
_SCALES = {"hundred": 100, "thousand": 1000}
_MONTH_WORDS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}  # fmt: skip


def _is_number_word(tokens: list[str], j: int, started: bool) -> bool:
    t = tokens[j]
    nxt = tokens[j + 1] if j + 1 < len(tokens) else ""
    if t in _UNITS or t in _TEENS or t in _TENS or t in _ORD_TEENS:
        return True
    if t in _ORD_UNITS:  # "one second" is time, "second of March" is a date
        prev = tokens[j - 2] if j >= 2 and tokens[j - 1] == "-" else tokens[j - 1] if j else ""
        return t != "second" or nxt == "of" or nxt in _MONTH_WORDS or prev in _TENS
    if t in ("oh", "o"):
        return started
    if t in _MULT:
        return nxt in _UNITS or nxt in ("oh", "o")
    return t in _SCALES and started


def _spelled_number(tokens: list[str], i: int) -> tuple[int, str]:
    """Parse the spelled-out number starting at ``i``; return (end index, digits) or (i, "")."""
    words: list[str] = []
    j = i
    while j < len(tokens):
        t = tokens[j]
        if _is_number_word(tokens, j, bool(words)):
            words.append(t)
            if t in _ORD_UNITS or t in _ORD_TEENS:  # an ordinal ends the number: "twelfth | 1985"
                j += 1
                break
        elif t == "-" and words and j + 1 < len(tokens) and _is_number_word(tokens, j + 1, True):
            pass  # "eighty-five"
        elif t == "and" and any(w in _SCALES for w in words) and j + 1 < len(tokens):
            if not _is_number_word(tokens, j + 1, True):
                break
        else:
            break
        j += 1
    if not words:
        return i, ""
    value = {**_UNITS, **_TEENS, **_TENS, **_ORD_UNITS, **_ORD_TEENS, "oh": 0, "o": 0}
    if any(w in _SCALES for w in words):  # arithmetic: "two thousand and two" -> 2002
        total = current = 0
        for w in words:
            if w in _SCALES:
                current = (current or 1) * _SCALES[w]
                if w == "thousand":
                    total, current = total + current, 0
            elif w in value:
                current += value[w]
        return j, str(total + current)
    chunks: list[str] = []  # concatenation: "nineteen eighty five" -> 19|85, "four four one seven" -> 4|4|1|7
    k = 0
    while k < len(words):
        w = words[k]
        nxt = words[k + 1] if k + 1 < len(words) else ""
        if w in _MULT:
            chunks.append(str(value[nxt]) * _MULT[w])
            k += 2
            continue
        if w in _TENS and (nxt in _UNITS or nxt in _ORD_UNITS) and nxt != "zero":
            chunks.append(str(_TENS[w] + value[nxt]))
            k += 2
            continue
        chunks.append(str(value[w]))
        k += 1
    return j, "".join(chunks)


def numbers_to_digits(text: str, merge: bool = True) -> str:
    """Every spoken or written number as digits.

    ``merge`` joins numbers said in groups ("98 76 54", "4-4-1-7") into one run, as a digit-run check
    needs; without it each number stays a separate token, as a date parser needs.
    """
    tokens = _TOKEN.findall(text.lower())
    out: list[str] = []
    i = 0
    while i < len(tokens):
        end, digits = _spelled_number(tokens, i)
        if end > i:
            out.append(digits if merge else f" {digits} ")
            i = end
            continue
        tok = tokens[i]
        if tok.isdigit():
            out.append(tok if merge else f" {tok} ")
        elif merge and tok in "-.,/" and out and out[-1].isdigit():
            pass  # separators inside a number: "4-4-1-7", "12/03/1985"
        else:
            out.append(f" {tok} ")
        i += 1
    return re.sub(r"\s+", " ", "".join(out)).strip()
