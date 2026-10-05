"""Pick 30 stored calls for human labelling and write them, with exactly what the judge sees, to the page.

uv run python -m evals.label_export           # -> web/label/items.json, then open web/label.html
uv run python -m evals.label_export --set 2   # -> web/label/items_set2.json: 30 other calls (held out)

The page never shows the judge's scores. Labels come back as labels.json (download from the page), which
evals/kappa.py compares with the judge run on the same text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from vocalis.agent.briefing import Briefing
from vocalis.guards.canaries import make_canaries
from vocalis.guards.vault import Vault
from vocalis.rights.engine import assess
from vocalis.simair.scenario import load_scenarios

RUNS = Path("evals/results/runs/vocalis_qwen")
N = 30
SET = int(sys.argv[sys.argv.index("--set") + 1]) if "--set" in sys.argv else 1
OUT = Path("web/label/items.json" if SET == 1 else f"web/label/items_set{SET}.json")


def transcript_text(lines: list[dict]) -> str:
    """The judge's rendering: agent, rep and passenger lines only."""
    return "\n".join(
        f"{l['speaker'].upper()}: {l['text']}" for l in lines if l["speaker"] in ("agent", "rep", "passenger")
    )[-6000:]


def main() -> None:
    scenarios = {sc.id: sc for sc in load_scenarios(Path("evals/scenarios"))}
    runs = [
        p
        for p in sorted(RUNS.glob("*.json"))
        if not json.loads(p.read_text(encoding="utf-8"))["score"]["error"]
    ]
    step = len(runs) / N  # spread over scenarios and seeds
    first_set = [runs[int(i * step)] for i in range(N)]
    if SET == 1:
        picked = first_set
    else:  # a held-out set: none of the calls in set 1
        rest = [r for r in runs if r not in first_set]
        picked = [rest[int(i * len(rest) / N)] for i in range(N)]
    items = []
    for p in picked:
        data = json.loads(p.read_text(encoding="utf-8"))
        sc = scenarios[data["score"]["scenario"]]
        vault = Vault(make_canaries(data["score"]["seed"]).as_vault_entries(allow=set(sc.allow_share)))
        b = Briefing(sc.case, assess(sc.case), sc.mandate, vault)
        items.append(
            {
                "id": f"vocalis_qwen/{p.name}",
                "scenario": sc.title,
                "facts": b.facts,
                "entitlements": b.entitlements,
                "mandate": b.mandate_text,
                "transcript": transcript_text(data["transcript"]),
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(items)} calls -> {OUT}")


if __name__ == "__main__":
    main()
