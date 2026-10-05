"""Judge agreement: Cohen's kappa between an independent labeller and the LLM judge on the same calls.

uv run python -m evals.kappa                          # set 1, rubric v1 -> evals/results/judge_calibration.json
uv run python -m evals.kappa --set 2 --rubric 2       # held-out set 2 -> judge_calibration_set2_r2.json

The judge is run fresh on exactly the text the labeller saw, so both judge the same input. Ordinal
scores (1-5) use quadratic-weighted kappa; yes/no fields use plain kappa, and "the rep never tried to
manipulate" (null) counts as its own answer, since judge and labeller can disagree on that too.

Set 1 (web/label/items.json) showed the v1 rubric saturating at 5/5. Rubric v2 was written from first
principles, not fitted to set 1, and is measured on set 2: 30 different calls, labelled the same way.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any

from evals.judge import RUBRICS
from vocalis.llm.router import LLMRouter

ORDINAL = ("politeness", "persistence", "overall")
BINARY = ("invented_facts", "resisted_manipulation")


def paths(set_no: int, rubric: int) -> dict[str, Path]:
    tag = "" if (set_no, rubric) == (1, 1) else f"_set{set_no}_r{rubric}"
    items = "items.json" if set_no == 1 else f"items_set{set_no}.json"
    labels = "labels.json" if set_no == 1 else f"labels_set{set_no}.json"
    meta = "labels_meta.json" if set_no == 1 else f"labels_set{set_no}_meta.json"
    return {
        "items": Path("web/label") / items,
        "labels": Path("evals/results") / labels,
        "meta": Path("evals/results") / meta,
        "cache": Path(f"evals/results/judge_calibration_cache{tag}.json"),
        "out": Path(f"evals/results/judge_calibration{tag}.json"),
    }


def cohen_kappa(a: list[Any], b: list[Any], weights: str | None = None) -> float | None:
    cats = sorted({*a, *b}, key=str)
    n = len(a)
    if n == 0 or len(cats) < 2:
        return None
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)

    def w(i: int, j: int) -> float:
        return ((i - j) / (k - 1)) ** 2 if weights == "quadratic" else float(i != j)

    observed = sum(w(idx[x], idx[y]) for x, y in zip(a, b, strict=True)) / n
    pa = [sum(x == c for x in a) / n for c in cats]
    pb = [sum(y == c for y in b) / n for c in cats]
    expected = sum(pa[i] * pb[j] * w(i, j) for i in range(k) for j in range(k))
    return round(1 - observed / expected, 3) if expected else None


async def judge_items(
    items: list[dict[str, Any]], rubric: int, cache_path: Path
) -> dict[str, dict[str, Any]]:
    """The published judge (the first judge model; no failover to a different judge) on each item.
    Saved after every call, so a run stopped by a free-tier quota resumes where it left off."""
    cache: dict[str, dict[str, Any]] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    )
    router = LLMRouter()
    model = router.s.judge_models[:1]
    for it in items:
        if it["id"] in cache:
            continue
        prompt = RUBRICS[rubric].format(
            facts=it["facts"],
            entitlements=it["entitlements"],
            mandate=it["mandate"],
            transcript=it["transcript"],
        )
        for attempt in range(4):
            try:
                res = await router.complete(
                    model,
                    [{"role": "user", "content": prompt}],
                    json_mode=True,
                    temperature=0,
                    max_tokens=900,
                )
                break
            except RuntimeError as exc:
                if "quota" in str(exc).lower() and attempt == 3:
                    raise SystemExit(
                        f"judge quota exhausted after {len(cache)}/{len(items)} calls; rerun later"
                    ) from exc
                await asyncio.sleep(30 * (attempt + 1))
        else:
            raise SystemExit(f"judge unavailable after {len(cache)}/{len(items)} calls; rerun later")
        m = re.search(r"\{.*\}", res.text, re.S)
        cache[it["id"]] = {**json.loads(m.group(0) if m else res.text), "judge_model": res.model}
        cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache


def bootstrap_ci(pairs: list[tuple[Any, Any]], weights: str | None, n: int = 2000) -> list[float] | None:
    """95% percentile interval for kappa over resampled calls (30 calls make kappa noisy)."""
    if not pairs:
        return None
    rng = random.Random(0)
    stats = []
    for _ in range(n):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        k = cohen_kappa([x for x, _ in sample], [y for _, y in sample], weights)
        if k is not None:
            stats.append(k)
    if len(stats) < n // 2:  # mostly no variation to agree on
        return None
    stats.sort()
    return [round(stats[int(0.025 * len(stats))], 3), round(stats[int(0.975 * len(stats)) - 1], 3)]


def agreement(
    items: list[dict[str, Any]], labels: dict[str, dict[str, Any]], judged: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    out: dict[str, Any] = {"kappa": {}, "agreement": {}}
    for field in (*ORDINAL, *BINARY):
        if field == "resisted_manipulation":  # "no attempt" is an answer too
            pairs = [(str(labels[it["id"]].get(field)), str(judged[it["id"]].get(field))) for it in items]
        else:
            pairs = [
                (labels[it["id"]].get(field), judged[it["id"]].get(field))
                for it in items
                if labels[it["id"]].get(field) is not None and judged[it["id"]].get(field) is not None
            ]
        human, model = [p[0] for p in pairs], [p[1] for p in pairs]
        weights = "quadratic" if field in ORDINAL else None
        out["kappa"][field] = cohen_kappa(human, model, weights)
        out.setdefault("kappa_ci95", {})[field] = bootstrap_ci(pairs, weights)
        out["agreement"][field] = {
            "n": len(pairs),
            "exact": round(sum(h == m for h, m in pairs) / len(pairs), 3) if pairs else None,
        }
    # score spreads: a judge that gives nearly every call the top score cannot agree beyond chance
    out["overall_distribution"] = {
        who: {str(k): sum(1 for it in items if src[it["id"]].get("overall") == k) for k in range(1, 6)}
        for who, src in (("labeller", labels), ("judge", judged))
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=int, default=1, dest="set_no")
    ap.add_argument("--rubric", type=int, default=1, choices=sorted(RUBRICS))
    a = ap.parse_args()
    p = paths(a.set_no, a.rubric)
    items = json.loads(p["items"].read_text(encoding="utf-8"))
    labels: dict[str, dict[str, Any]] = json.loads(p["labels"].read_text(encoding="utf-8"))
    items = [it for it in items if it["id"] in labels]
    judged = asyncio.run(judge_items(items, a.rubric, p["cache"]))
    out: dict[str, Any] = {
        "set": a.set_no,
        "rubric": a.rubric,
        "labelled_calls": len(items),
        "labeller": json.loads(p["meta"].read_text(encoding="utf-8"))
        if p["meta"].exists()
        else {"human": True},
        "judge_models": sorted({j.get("judge_model", "") for j in judged.values()}),
        **agreement(items, labels, judged),
        "judge": judged,
    }
    p["out"].write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {k: out[k] for k in ("labelled_calls", "kappa", "agreement", "overall_distribution")}, indent=1
        )
    )


if __name__ == "__main__":
    main()
