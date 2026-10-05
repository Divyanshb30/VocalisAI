"""Tie every citation to the regulation text it rests on.

An entitlement's citation ("DGCA para 3.3.2 and para 3.7.1", "UK261 Art. 5(1)(c) and Art. 7") names
clauses; each must resolve to a passage of the corpus in data/policy/, so every figure the agent cites
can be shown in the official wording. The same parser finds regulation references in what the agent
said, so a call that cites a clause that doesn't exist can be counted.
"""

from __future__ import annotations

import re

from vocalis.rights.models import Citation, Regime
from vocalis.rights.policy import Passage

CODES = {
    Regime.DGCA: "DGCA",
    Regime.GCAA: "GCAA",
    Regime.UK261: "UK261",
    Regime.EU261: "EU261",
    Regime.MONTREAL: "MONTREAL",
}
_REF = re.compile(r"(Art\. \d+(?:\([0-9a-z]+\))*|para \d+(?:\.\d+)+|PWP\.[A-D]\.\d{3}(?:\([a-z]\))?)")


def clause_refs(c: Citation) -> list[str]:
    """Corpus ids a citation names: 'Art. 5(1)(c) and Art. 7' under UK261 -> UK261 Art. 5(1)(c), UK261 Art. 7."""
    out = []
    for part in re.split(r"\band\b|;|,", c.clause):
        code = "UAE-CTL" if re.search(r"\bCTL\b|Commercial Transactions Law", part) else CODES[c.regime]
        out += [f"{code} {r}" for r in _REF.findall(part)]
    return out


def resolve(ref: str, passages: list[Passage]) -> list[Passage]:
    """Passages a reference points at: the clause itself, its sub-paragraphs (Art. 7 -> Art. 7(1), (2)),
    or the paragraph containing it (Art. 5(1)(c) -> Art. 5(1))."""
    exact = [p for p in passages if p.id == ref]
    if exact:
        return exact
    finer = [p for p in passages if any(p.id.startswith(ref + sep) for sep in ("(", ".", " "))]
    if finer:
        return finer
    return [p for p in passages if ref.startswith(p.id + "(")]


def unresolved(c: Citation, passages: list[Passage]) -> list[str]:
    refs = clause_refs(c)
    if not refs:
        return [c.clause]
    return [r for r in refs if not resolve(r, passages)]


_SPOKEN_PARA = re.compile(r"\b(?:para(?:graph)?|section)\s+(\d+(?:\.\d+)+)", re.I)
_SPOKEN_ART = re.compile(r"\bArt(?:icle|\.)\s+(\d+)", re.I)
# the regime named shortly before "Article N" in the same sentence
_REGIMES = (
    ("UK261", re.compile(r"\bUK\s?261\b", re.I)),
    ("EU261", re.compile(r"\bEU\s?261\b", re.I)),
    ("MONTREAL", re.compile(r"\bMontreal\b", re.I)),
    ("UAE-CTL", re.compile(r"\bCommercial Transactions Law\b", re.I)),
)
_SPOKEN_PWP = re.compile(r"\bPWP\.?\s?([A-D])\.?\s?(\d{3})", re.I)


def spoken_refs(text: str) -> list[str]:
    """Regulation clauses named in speech, as corpus ids where the regime is clear from the words."""
    out = [f"DGCA para {n}" for n in _SPOKEN_PARA.findall(text)]
    for m in _SPOKEN_ART.finditer(text):
        before = re.split(r"[.?!]", text[max(0, m.start() - 60) : m.start()])[-1]
        named = [(mm.end(), code) for code, pat in _REGIMES for mm in pat.finditer(before)]
        code = max(named)[1] if named else "?"
        out.append(f"{code} Art. {m.group(1)}")
    out += [f"GCAA PWP.{a.upper()}.{n}" for a, n in _SPOKEN_PWP.findall(text)]
    return out


def unknown_spoken_refs(text: str, passages: list[Passage]) -> list[str]:
    """Clauses the agent named that exist in no corpus passage. 'Article 7' with no regime named counts
    as known if any regime has an Article 7."""
    unknown = []
    for ref in spoken_refs(text):
        if ref.startswith("? "):
            clause = ref[2:]
            if not any(
                p.id.split(" ", 1)[1] == clause or p.id.split(" ", 1)[1].startswith(clause + "(")
                for p in passages
            ):
                unknown.append(ref)
        elif not resolve(ref, passages):
            unknown.append(ref)
    return unknown
