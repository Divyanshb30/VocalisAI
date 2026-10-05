"""Build data/policy/{dgca,gcaa,uae_ctl}.txt from the official PDFs, one passage per numbered clause.

The PDFs are downloaded to data/raw/policy/ (git-ignored) and converted to text with pypdf:

    uv run --no-project --with pypdf python scripts/build_policy_corpus.py

Text is kept verbatim apart from page furniture (headers, footers, page numbers) and words the PDF
text layer splits in two ("bey ond"), which are listed below so every change is visible.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

RAW = Path("data/raw/policy")
OUT = Path("data/policy")

SOURCES = {
    "dgca": {
        "pdf": "dgca-car-s3m4-rev4.pdf",
        "url": "https://www.dgca.gov.in/digigov-portal/Upload?flag=iframeAttachView&attachId=we1PSlOuQhYdHcwKKrm7ew%3D%3D",
        "source": "DGCA Civil Aviation Requirements, Section 3, Series M, Part IV, Rev. 4 dated 25 Jan 2023 "
        "(effective 15 Feb 2023): facilities to be provided to passengers by airlines due to denied "
        "boarding, cancellation of flights and delays in flights",
        "jurisdiction": "IN",
    },
    "gcaa": {
        "pdf": "car-pwp-issue02.pdf",
        "url": "https://www.gcaa.gov.ae/en/epublication/EPublications/Civil%20Aviation%20Regulations%20(CARs)/"
        "CAR%20III%20-%20GENERAL%20REGULATIONS/CAR-PWP%20-%20PASSENGER%20WELFARE%20PROGRAM%20-%20ISSUE%2002.pdf",
        "source": "UAE GCAA Civil Aviation Regulations CAR-PWP, Passenger Welfare Program, Issue 02 "
        "(issued 25 Dec 2024, applicable 1 Jan 2025)",
        "jurisdiction": "AE",
    },
    "uae_ctl": {
        "pdf": "uae-ctl-2022.pdf",
        "url": "https://uaelegislation.gov.ae/en/legislations/1610",
        "source": "UAE Federal Decree-Law No. 50 of 2022 promulgating the Commercial Transactions Law "
        "(official English translation), carriage of persons and air carriage",
        "jurisdiction": "AE",
    },
}

# Words the PDF text layer splits in two, and the few the GCAA text spells with stray spaces.
SPLIT_WORDS = {
    "bey ond": "beyond", "cancell ation": "cancellation", "d elay": "delay", "dir ectly": "directly",
    "pre sented": "presented", "acco unt": "account", "a ffected": "affected", "t he": "the",
    "alternate fli ghts": "alternate flights", "con cerned": "concerned", "de lays": "delays",
    "check -in": "check-in", "one -way": "one-way", "pas sengers": "passengers", "w hat": "what",
    "t hat": "that", "i n accordance": "in accordance", "ret urn": "return", "re -routing": "re-routing",
    "th at": "that", "th e": "the", "colour -coded": "colour-coded", "obligati ons": "obligations",
    "fac ilitate": "facilitate", "equiv alent": "equivalent", "indem nity": "indemnity", "direc t": "direct",
    "go ods": "goods", "vo id": "void", "intent  to": "intent to", "impac t": "impact",
}  # fmt: skip


def _pdf_text(name: str) -> str:
    from pypdf import PdfReader

    return "\n".join(p.extract_text() or "" for p in PdfReader(RAW / name).pages)


def _clean(text: str) -> str:
    text = " ".join(text.split())
    for bad, good in SPLIT_WORDS.items():
        text = re.sub(rf"\b{re.escape(bad)}\b", good, text)
    return text


def _sections(body: str, heading: str, keep: Callable[[str], bool]) -> list[tuple[str, str, str]]:
    """(id, title, text) for each heading match whose id ``keep`` accepts."""
    out = []
    parts = re.split(heading, body)
    for i in range(1, len(parts) - 2, 3):
        sid, title, text = parts[i], parts[i + 1], parts[i + 2]
        if keep(sid):
            text = re.sub(r"\s*SUBPART [A-D]:.*$", "", _clean(text))  # the next subpart's heading
            out.append((sid, _clean(title), text))
    return out


def dgca() -> list[tuple[str, str, str]]:
    lines = []
    for line in _pdf_text(SOURCES["dgca"]["pdf"]).splitlines():
        s = line.strip()
        if re.fullmatch(
            r"(\d+|Rev\. 4, dated 25th Jan, 2023|CIVIL AVIATION REQUIREMENTS|SERIES .M. PART IV|SECTION 3|"
            r"6TH AUGUST, 2010)",
            s,
        ):
            continue  # page furniture
        lines.append(line)
    body = "\n".join(lines)
    body = body[body.index("1.4 The operating airline") : body.index("(Arun Kumar)")]
    # numbered paragraphs: "3.3.2  Passengers who ..."; headings like "3.3 Cancellation of Flight" have no text
    paras = re.split(r"(?m)^\s*(\d\.\d+(?:\.\d+)?)\s+(?=[A-Z])", body)  # "Para\n1.5 which": a reference
    out = []
    titles = {}
    for sid, text in zip(paras[1::2], paras[2::2], strict=True):
        text = _clean(text)
        if sid.count(".") == 1 and len(text) < 60:  # a heading: "3.3 Cancellation of Flight"
            titles[sid] = text
            continue
        parent = sid.rsplit(".", 1)[0] if sid.count(".") == 2 else sid
        if sid.startswith(("2.", "3.1.", "3.10.")) and sid not in ("2.6", "2.8", "2.9", "3.10.1"):
            continue  # booking information, other definitions, grievance admin: not about entitlements
        out.append(
            (f"para {sid}", titles.get(parent, "Definitions" if sid.startswith("2.") else "General"), text)
        )
    return out


def gcaa() -> list[tuple[str, str, str]]:
    text = _pdf_text(SOURCES["gcaa"]["pdf"])
    text = re.sub(
        r"(?m)^\s*PASSENGER WELFARE PROGRAM\s*$|^\s*CAR-PWP-ISSUE 02\s+Page \d+ of 18\s*$", "", text
    )
    body = text[re.search(r"PWP\.A\.002 Aim\s*\n\s*This regulation", text).start() :]  # type: ignore[union-attr]
    # a heading names a section and a capitalised title; "PWP.B.004 as applicable;" is a reference
    heading = (
        r"(?m)^\s*((?:GM\d? to |GM\d |AMC\d? )?PWP\.[A-D]\.? ?\d{3}(?: \([a-z]\))?(?: \([a-z0-9]\))*)"
        r"\s+([A-Z][^\n;.]*)\n"
    )
    want = (
        "PWP.A.003",
        "PWP.A.005",
        "PWP.B.002",
        "PWP.B.003",
        "PWP.B.004",
        "PWP.B.005",
        "PWP.B.006",
        "PWP.D.001",
    )
    secs = _sections(body, heading, lambda sid: any(w in sid.replace("PWP.B 0", "PWP.B.0") for w in want))
    out = []
    for sid, title, txt in secs:
        sid = sid.replace("PWP.B 0", "PWP.B.0")
        if sid == "PWP.A.005":  # terminology: keep the two definitions entitlements depend on
            for term in ("Cancellation", "Delay", "Denied boarding"):
                m = re.search(rf"{term} – (.*?)(?= [A-Z][a-z]+(?: [A-Za-z]+)? – |$)", txt)
                if m:
                    out.append((f"{sid} {term}", f"Definition: {term}", m.group(1)))
            continue
        sub = (
            re.split(r"(?:(?<=\. )|(?<=; )|(?<=: )|^)\(([a-c])\) ", txt)
            if sid.startswith("PWP.B.00")
            else [txt]
        )
        if len(sub) > 1 and sid in ("PWP.B.003", "PWP.B.004", "PWP.B.006"):
            for letter, part in zip(sub[1::2], sub[2::2], strict=True):
                out.append((f"{sid}({letter})", title, part.strip()))
        else:
            out.append((sid, title, txt))
    return out


def uae_ctl() -> list[tuple[str, str, str]]:
    text = _pdf_text(SOURCES["uae_ctl"]["pdf"])
    text = re.sub(
        r"(?m)^\s*Federal Decree by Law No\. \(50\) of 2022, Promulgating the Commercial Transactions Law \d+\s*$",
        "",
        text,
    )
    titles = {322: "Carriage of persons: force majeure", 333: "Carriage of persons: delay and injury",
              354: "Air carriage: application", 357: "Air carriage: delay"}  # fmt: skip
    out = []
    for num, title in titles.items():
        m = re.search(rf"Article \({num}\)\s*\n(.*?)(?=\n\s*Article \(\d+\))", text, re.S)
        assert m, num
        out.append((f"Art. {num}", title, _clean(m.group(1))))
    return out


def write(name: str, sections: list[tuple[str, str, str]]) -> None:
    meta = SOURCES[name]
    lines = [
        f"# source: {meta['source']}",
        f"# url: {meta['url']}",
        f"# jurisdiction: {meta['jurisdiction']}",
        "# retrieved: 2026-10-05 (text layer of the official PDF; page furniture removed)",
    ]
    for sid, title, text in sections:
        lines += [f"§ {sid} | {title}", text]
    (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{name}: {len(sections)} passages")


if __name__ == "__main__":
    write("dgca", dgca())
    write("gcaa", gcaa())
    write("uae_ctl", uae_ctl())
