"""Document-extraction benchmark: field-level accuracy, vision-only vs vision + barcode.

uv run python -m evals.docbench.run --data evals/docbench/data --model gemini-flash-latest
uv run python -m evals.docbench.run --data evals/docbench/captured --out evals/results/docbench_captured.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from datetime import date
from pathlib import Path

from vocalis.docintel.bcbp import read_image
from vocalis.docintel.extract import extract_with_gemini, reconcile, redact

FIELDS = [
    "passenger_first_name",
    "passenger_last_name",
    "booking_reference",
    "airline_code",
    "flight_number",
    "origin_iata",
    "destination_iata",
    "departure_date",
    "departure_time",
    "disruption",
    "notice_date",
    "fare_total",
    "currency",
]


def norm(v: object) -> str:
    s = str(v or "").upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s.lstrip("0") if s.isdigit() else s


async def run(data: Path, model: str, limit: int | None) -> dict:
    labels = json.loads((data / "labels.json").read_text(encoding="utf-8"))
    items = list(labels.items())[:limit]
    per_doc = []
    for name, truth in items:
        path = data / name
        t0 = time.perf_counter()
        try:
            doc = redact(await extract_with_gemini(path.read_bytes(), "image/jpeg", model))
        except Exception as exc:
            per_doc.append({"doc": name, "error": str(exc)[:200]})
            continue
        latency = time.perf_counter() - t0
        rec = reconcile(doc, read_image(path), reference=date(2026, 10, 1))
        row = {
            "doc": name,
            "type": truth.get("kind", truth["doc_type"]),
            "severity": truth.get("severity"),
            "latency_s": round(latency, 2),
            "conflicts": len(rec.conflicts),
        }
        for variant, fields in (("vision", doc.model_dump()), ("reconciled", rec.fields.model_dump())):
            checked = [f for f in FIELDS if f in truth]
            correct = [f for f in checked if norm(fields.get(f)) == norm(truth[f])]
            row[f"{variant}_correct"] = len(correct)
            row[f"{variant}_total"] = len(checked)
            row[f"{variant}_wrong"] = sorted(set(checked) - set(correct))
            row[f"{variant}_read"] = {
                f: fields.get(f) for f in row[f"{variant}_wrong"]
            }  # for auditing misses
        row["barcode_decoded"] = rec.barcode is not None
        per_doc.append(row)
        await asyncio.sleep(4)  # stay inside free-tier requests-per-minute
    ok = [r for r in per_doc if "error" not in r]

    def acc(variant: str, rows: list[dict] | None = None) -> float:
        rows = ok if rows is None else rows
        c = sum(r[f"{variant}_correct"] for r in rows)
        t = sum(r[f"{variant}_total"] for r in rows)
        return round(c / t, 4) if t else 0.0

    def breakdown(key: str) -> dict[str, dict[str, float | int]]:
        groups: dict[str, list[dict]] = {}
        for r in ok:
            if r.get(key):
                groups.setdefault(r[key], []).append(r)
        return {
            k: {"documents": len(g), "vision": acc("vision", g), "with_barcode": acc("reconciled", g)}
            for k, g in sorted(groups.items())
        }

    summary = {
        "model": model,
        "documents": len(items),
        "errors": len(per_doc) - len(ok),
        "field_accuracy_vision": acc("vision"),
        "field_accuracy_with_barcode": acc("reconciled"),
        "barcode_conflicts_caught": sum(r["conflicts"] for r in ok),
        "mean_latency_s": round(sum(r["latency_s"] for r in ok) / len(ok), 2) if ok else None,
        "by_severity": breakdown("severity"),
        "by_type": breakdown("type"),
    }
    return {"summary": summary, "per_doc": per_doc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("evals/docbench/data"))
    ap.add_argument("--model", default="gemini-flash-latest")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, default=Path("evals/results/docbench.json"))
    a = ap.parse_args()
    result = asyncio.run(run(a.data, a.model, a.limit))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
