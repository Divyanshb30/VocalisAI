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


async def judge_items(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = (
        json.loads(OUT.read_text(encoding="utf-8")).get("judge", {}) if OUT.exists() else {}
    )
    router = LLMRouter()
    for it in items:
        if it["id"] in cache:
            continue
        prompt = RUBRIC.format(
            facts=it["facts"],
            entitlements=it["entitlements"],
            mandate=it["mandate"],
            transcript=it["transcript"],
        )
        res = await router.complete(
            router.s.judge_models,
            [{"role": "user", "content": prompt}],
            json_mode=True,
            temperature=0,
            max_tokens=500,
        )
        m = re.search(r"\{.*\}", res.text, re.S)
        cache[it["id"]] = {**json.loads(m.group(0) if m else res.text), "judge_model": res.model}
    return cache


def main() -> None:
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    labels: dict[str, dict[str, Any]] = json.loads(LABELS.read_text(encoding="utf-8"))
    items = [it for it in items if it["id"] in labels]
    judged = asyncio.run(judge_items(items))
    out: dict[str, Any] = {"labelled_calls": len(items), "kappa": {}, "agreement": {}}
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
    out["judge"] = judged
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("labelled_calls", "kappa", "agreement")}, indent=1))


if __name__ == "__main__":
    main()
