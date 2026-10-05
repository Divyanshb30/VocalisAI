"""Judge calibration: Cohen's kappa between human labels and the LLM judge on the same calls.

uv run python -m evals.kappa   # reads evals/results/labels.json (downloaded from web/label.html)
                               # -> evals/results/judge_calibration.json

The judge is run fresh on exactly the text the labeller saw (web/label/items.json), so both judge the
same input. Ordinal scores (1-5) use quadratic-weighted kappa; yes/no fields use plain kappa.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from evals.judge import RUBRIC
from vocalis.llm.router import LLMRouter

ITEMS = Path("web/label/items.json")
LABELS = Path("evals/results/labels.json")
OUT = Path("evals/results/judge_calibration.json")
ORDINAL = ("politeness", "persistence", "overall")
BINARY = ("invented_facts", "resisted_manipulation")


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


CACHE = Path("evals/results/judge_calibration_cache.json")


async def judge_items(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The published judge (the first judge model; no failover to a different judge) on each item.
    Saved after every call, so a run stopped by a free-tier quota resumes where it left off."""
    cache: dict[str, dict[str, Any]] = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    router = LLMRouter()
    model = router.s.judge_models[:1]
    for it in items:
        if it["id"] in cache:
            continue
        prompt = RUBRIC.format(
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
                    max_tokens=500,
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
        CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache


def main() -> None:
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    labels: dict[str, dict[str, Any]] = json.loads(LABELS.read_text(encoding="utf-8"))
    items = [it for it in items if it["id"] in labels]
    judged = asyncio.run(judge_items(items))
    meta = Path("evals/results/labels_meta.json")
    out: dict[str, Any] = {
        "labelled_calls": len(items),
        "labeller": json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {"human": True},
        "judge_models": sorted({j.get("judge_model", "") for j in judged.values()}),
        "kappa": {},
        "agreement": {},
    }
    for field in (*ORDINAL, *BINARY):
        pairs = [
            (labels[it["id"]].get(field), judged[it["id"]].get(field))
            for it in items
            if labels[it["id"]].get(field) is not None and judged[it["id"]].get(field) is not None
        ]
        human, model = [p[0] for p in pairs], [p[1] for p in pairs]
        out["kappa"][field] = cohen_kappa(human, model, "quadratic" if field in ORDINAL else None)
        out["agreement"][field] = {
            "n": len(pairs),
            "exact": round(sum(h == m for h, m in pairs) / len(pairs), 3) if pairs else None,
        }
    # score spreads: a judge that gives nearly every call the top score cannot agree beyond chance
    out["overall_distribution"] = {
        who: {str(k): sum(1 for it in items if src[it["id"]].get("overall") == k) for k in range(1, 6)}
        for who, src in (("labeller", labels), ("judge", judged))
    }
    out["judge"] = judged
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("labelled_calls", "kappa", "agreement")}, indent=1))


if __name__ == "__main__":
    main()
