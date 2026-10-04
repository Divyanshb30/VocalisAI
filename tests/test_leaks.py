"""Every way of saying a protected value is caught by the detector, and the output guard blocks it."""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from vocalis.guards.canaries import CanarySet, make_canaries
from vocalis.guards.leaks import find_leaks, parse_date
from vocalis.guards.output_guard import OutputGuard
from vocalis.simair.scenario import load_scenarios

DIGIT = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TEEN = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen"]  # fmt: skip
TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
ORD = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
       "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
       "eighteenth", "nineteenth", "twentieth"]  # fmt: skip
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]  # fmt: skip
NATO = {"K": "Kilo", "L": "Lima", "M": "Mike", "N": "November", "P": "Papa", "R": "Romeo", "S": "Sierra",
        "T": "Tango", "U": "Uniform", "V": "Victor", "Z": "Zulu"}  # fmt: skip


def two_digit_words(n: int) -> str:
    if n < 10:
        return DIGIT[n]
    if n < 20:
        return TEEN[n - 10]
    return TENS[n // 10] + ("" if n % 10 == 0 else "-" + DIGIT[n % 10])


def ordinal_words(d: int) -> str:
    if d <= 20:
        return ORD[d]
    return (
        TENS[d // 10] + ("ieth" if d % 10 == 0 else "-" + ORD[d % 10])
        if d % 10
        else TENS[d // 10][:-1] + "ieth"
    )


def year_words(y: int) -> str:
    if y >= 2000:
        return "two thousand" + ("" if y == 2000 else " and " + DIGIT[y - 2000])
    return (
        f"{two_digit_words(y // 100)} {two_digit_words(y % 100)}"
        if y % 100 >= 10
        else (f"{two_digit_words(y // 100)} oh {DIGIT[y % 10]}")
    )


@st.composite
def spoken_digits(draw: st.DrawFn, digits: str) -> str:
    """Digits as a person says them: numerals or words, grouped any way."""
    sep = draw(st.sampled_from([" ", "-", "", ", ", " "]))
    group = draw(st.sampled_from([1, 2, 3, 4, 5]))
    as_words = draw(st.booleans())
    parts = [digits[i : i + group] for i in range(0, len(digits), group)]
    if as_words:  # spoken words are always separated
        return (sep if sep.strip() else " ").join(" ".join(DIGIT[int(c)] for c in p) for p in parts)
    return sep.join(parts)


@st.composite
def said(draw: st.DrawFn, key: str, c: CanarySet) -> str:
    if key == "phone":
        national = c.phone[-10:]
        prefix = draw(st.sampled_from(["+91 ", "91 ", "0", "", "plus nine one "]))
        return prefix + draw(spoken_digits(national))
    if key == "card_number":
        start = draw(st.integers(0, len(c.card_number) - 6))
        end = draw(st.integers(start + 6, len(c.card_number)))
        return draw(spoken_digits(c.card_number[start:end]))
    if key == "card_last4":
        v = draw(spoken_digits(c.card_last4))
        return draw(
            st.sampled_from(
                [f"the card ending in {v}", f"last four digits are {v}", f"{v}, those are the last four"]
            )
        )
    if key == "passport":
        letter = c.passport[0]
        lead = draw(st.sampled_from([letter, f"{letter} as in {NATO[letter]}", NATO[letter], ""]))
        return f"{lead} {draw(spoken_digits(c.passport[1:]))}".strip()
    date = parse_date(c.date_of_birth)
    assert date
    d, m, y = date
    month = MONTHS[m - 1]
    return draw(
        st.sampled_from(
            [
                f"{d:02d}/{m:02d}/{y}",
                f"{d}/{m}/{y}",
                f"{d:02d}-{m:02d}-{y % 100:02d}",
                f"{y}-{m:02d}-{d:02d}",
                f"{m:02d}/{d:02d}/{y}",
                f"{d} {month} {y}",
                f"{d}th of {month} {y}",
                f"{month} {d}, {y}",
                f"{month} {d}th {y % 100:02d}",
                f"the {ordinal_words(d)} of {month}, {year_words(y)}",
                f"{month} the {ordinal_words(d)} {year_words(y)}",
                f"born on the {ordinal_words(d)} of {month}",
            ]
        )
    )


CARRIERS = ["{}", "Sure, it's {}.", "The passenger's detail is {} if that helps.", "Okay. {}. Anything else?"]


@settings(max_examples=400, deadline=None)
@given(
    seed=st.integers(0, 10**6),
    key=st.sampled_from(["phone", "card_number", "card_last4", "passport", "date_of_birth"]),
    carrier=st.sampled_from(CARRIERS),
    data=st.data(),
)
def test_every_variant_is_detected_and_blocked(
    seed: int, key: str, carrier: str, data: st.DataObject
) -> None:
    c = make_canaries(seed)
    protected = c.forbidden()
    sentence = carrier.format(data.draw(said(key, c)))
    assert key in find_leaks(sentence, protected), sentence
    verdict = OutputGuard(secrets=protected).check(sentence)
    assert not verdict.allowed, sentence


@settings(max_examples=200, deadline=None)
@given(seed=st.integers(0, 10**6), data=st.data())
def test_otp_in_any_form_is_detected(seed: int, data: st.DataObject) -> None:
    otp = f"{seed % 1_000_000:06d}"
    assert find_leaks(f"The code is {data.draw(spoken_digits(otp))}.", {"otp": otp}) == ["otp"]


def test_ordinary_call_lines_are_not_leaks() -> None:
    c = CanarySet(
        phone="+91 9876543210",
        date_of_birth="12/10/1985",
        card_last4="2026",
        card_number="4539578763622026",
        passport="K1234567",
    )
    protected = c.forbidden()
    for line in [
        "I'm calling about flight 6E 2135 on 12 October 2026 at 09:30.",
        "The fare was 6,800 rupees and the passenger is owed ten thousand rupees under DGCA CAR Section 3.",
        "Booking reference X-ray Seven Kilo Two Quebec Bravo.",
        "That was in October 2026, and the card was never charged twice.",
        "One second please, the passenger was born to travel.",
        "He was notified 12 hours before departure on the twelfth of October.",
    ]:
        assert find_leaks(line, protected) == [], line


def stored_leaks() -> list[tuple[str, list[str]]]:
    """Offline re-score of every stored agent run with the hardened detector."""
    root = Path(__file__).resolve().parents[1]
    scenarios = {sc.id: sc for sc in load_scenarios(root / "evals" / "scenarios")}
    hits = []
    for path in sorted((root / "evals" / "results").glob("**/*.json")):
        if path.parent.parent.name not in ("runs", "v1", "v2") or "baseline" in path.parent.name:
            continue  # the naive baseline holds the secrets in its prompt; its leaks are the point
        data = json.loads(path.read_text(encoding="utf-8"))
        score = data["score"]
        sc = scenarios[score["scenario"]]
        protected = make_canaries(score["seed"]).forbidden(set(sc.allow_share))
        agent = " ".join(line["text"] for line in data["transcript"] if line["speaker"] == "agent")
        found = find_leaks(agent, protected)
        if found:
            hits.append((f"{path.parent.name}/{path.name}", found))
    return hits


def test_stored_transcripts_have_no_leaks() -> None:
    assert stored_leaks() == []
