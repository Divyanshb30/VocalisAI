"""Regulation retrieval benchmark: BM25 vs dense vs hybrid (RRF) on a hand-written gold set.

uv run python -m evals.retrieval   # -> evals/results/retrieval.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from vocalis.rights.policy import EMBED_MODEL, RRF_K, PolicyIndex

GOLD = Path("evals/retrieval_gold.yaml")
OUT = Path("evals/results/retrieval.json")
MODES = ("bm25", "dense", "hybrid")


def evaluate(index: PolicyIndex, gold: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [p.id for p in index.passages]
    missing = {r for g in gold for r in g["relevant"]} - set(ids)
    if missing:
        raise ValueError(f"gold set cites passages not in the corpus: {sorted(missing)}")
    results: dict[str, Any] = {}
    for mode in MODES:
        r1 = r5 = rr = 0.0
        misses = []
        for g in gold:
            ranked = [ids[i] for i, _ in index.rank(g["q"], mode)]  # type: ignore[arg-type]
            first = next((n for n, pid in enumerate(ranked, 1) if pid in g["relevant"]), None)
            r1 += first == 1
            r5 += first is not None and first <= 5
            rr += 1 / first if first and first <= 10 else 0.0
            if not first or first > 5:
                misses.append({"q": g["q"], "first_relevant_rank": first, "top": ranked[:3]})
        n = len(gold)
        results[mode] = {
            "recall_at_1": round(r1 / n, 4),
            "recall_at_5": round(r5 / n, 4),
            "mrr_at_10": round(rr / n, 4),
            "misses_at_5": misses,
        }
    return {
        "questions": len(gold),
        "passages": len(ids),
        "sources": sorted({p.source for p in index.passages}),
        "dense_model": EMBED_MODEL,
        "rrf_k": RRF_K,
        "results": results,
    }


def main() -> None:
    gold = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    out = evaluate(PolicyIndex(), gold)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for mode, r in out["results"].items():
        print(f"{mode:7} R@1 {r['recall_at_1']:.0%}  R@5 {r['recall_at_5']:.0%}  MRR@10 {r['mrr_at_10']:.2f}")


if __name__ == "__main__":
    main()
