"""Bundle recorded eval calls into web/demo/calls.json for the replay demo (no server needed).

uv run python -m evals.export_demo
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from vocalis.simair.scenario import load_scenarios
from vocalis.web.server import case_view

RUNS = Path("evals/results/runs/vocalis")
OUT = Path("web/demo")


def main() -> None:
    scenarios = {s.id: s for s in load_scenarios(Path("evals/scenarios"))}
    calls = []
    for path in sorted(RUNS.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        s = data["score"]
        sc = scenarios.get(s["scenario"])
        if sc is None:
            continue
        calls.append(
            {
                "id": sc.id,
                "title": sc.title,
                "jurisdiction": sc.jurisdiction,
                "persona": sc.rep.persona.value,
                "case": case_view(sc.case, sc.mandate),
                "transcript": [
                    {"t": l["t"], "speaker": l["speaker"], "text": l["text"], "meta": l.get("meta", {})}
                    for l in data["transcript"]
                ],
                "report": {
                    "outcome": s["granted_outcome"],
                    "reference_number": (data.get("agent_resolution") or {}).get("reference_number"),
                    "reference_verified": s["reference_captured"],
                    "success": s["success"],
                    "handoffs": len(s["handoffs"]),
                    "leaks": s["leaks"],
                    "disclosed_as_ai": s["disclosed_first"],
                    "talker": ", ".join(s.get("talker_models", [])),
                },
            }
        )
    summary = json.loads(Path("evals/results/summary.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calls.json").write_text(json.dumps({"summary": summary, "calls": calls}), encoding="utf-8")
    samples = Path("evals/docbench/data")
    (OUT / "samples").mkdir(exist_ok=True)
    for name in ("doc_000.jpg", "doc_001.jpg"):
        if (samples / name).exists():
            shutil.copy(samples / name, OUT / "samples" / name)
    print(f"exported {len(calls)} calls to {OUT / 'calls.json'}")


if __name__ == "__main__":
    main()
