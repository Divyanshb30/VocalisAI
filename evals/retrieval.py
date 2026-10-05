"""Regulation retrieval benchmark: BM25 vs dense vs hybrid (RRF) on a hand-written gold set.

uv run python -m evals.retrieval   # -> evals/results/retrieval.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from vocalis.rights.policy import DENSE_WEIGHT, EMBED_MODEL, RRF_K, PolicyIndex, scope_for

GOLD = Path("evals/retrieval_gold.yaml")
PARAPHRASE = Path("evals/retrieval_paraphrase.yaml")  # the same questions, reworded by an LLM
REGION = {"DGCA": "IN", "GCAA": "AE", "UAE-CTL": "AE", "MONTREAL": "INTL", "EU261": "EU/UK", "UK261": "EU/UK"}
OUT = Path("evals/results/retrieval.json")
MODES = ("bm25", "dense", "hybrid", "hybrid_scoped")
# hybrid_scoped searches only the rules that apply to the case, as the agent does: a question's region
# stands for the case's jurisdiction (Montreal questions are international, so they search everything)
REGIMES = {
    "IN": ["DGCA"],
    "AE": ["GCAA"],
    "EU/UK": ["UK261", "EU261"],
    "INTL": ["DGCA", "GCAA", "UK261", "EU261"],
}


def ranking(index: PolicyIndex, ids: list[str], g: dict[str, Any], mode: str) -> list[str]:
    if mode != "hybrid_scoped":
        return [ids[i] for i, _ in index.rank(g["q"], mode)]  # type: ignore[arg-type]
    scope = scope_for(REGIMES[REGION[g["relevant"][0].split(" ", 1)[0]]])
    return [ids[i] for i, _ in index.rank(g["q"], "hybrid") if index.passages[i].jurisdiction in scope]


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
            ranked = ranking(index, ids, g, mode)
            first = next((n for n, pid in enumerate(ranked, 1) if pid in g["relevant"]), None)
            r1 += first == 1
            r5 += first is not None and first <= 5
            rr += 1 / first if first and first <= 10 else 0.0
            if not first or first > 5:
                misses.append({"q": g["q"], "first_relevant_rank": first, "top": ranked[:3]})
        n = len(gold)
        by_region: dict[str, list[int]] = {}
        for g in gold:
            region = REGION[g["relevant"][0].split(" ", 1)[0]]
            ranked = ranking(index, ids, g, mode)
            hit = any(pid in g["relevant"] for pid in ranked[:5])
            by_region.setdefault(region, []).append(hit)
        results[mode] = {
            "recall_at_5_by_region": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_region.items())},
            "questions_by_region": {k: len(v) for k, v in sorted(by_region.items())},
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
        "dense_weight": DENSE_WEIGHT,
        "results": results,
    }


def main() -> None:
    gold = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    index = PolicyIndex()
    out = evaluate(index, gold)
    if PARAPHRASE.exists():
        para = yaml.safe_load(PARAPHRASE.read_text(encoding="utf-8"))
        out["paraphrase"] = {
            "questions": len(para),
            "results": {
                m: {k: v for k, v in r.items() if k != "misses_at_5"}
                for m, r in evaluate(index, para)["results"].items()
            },
        }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for mode, r in out["results"].items():
        print(f"{mode:7} R@1 {r['recall_at_1']:.0%}  R@5 {r['recall_at_5']:.0%}  MRR@10 {r['mrr_at_10']:.2f}")


if __name__ == "__main__":
    main()
