"""Bundle recorded eval calls into web/demo/calls.json for the replay demo (no server needed).

uv run python -m evals.export_demo
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from vocalis.simair.scenario import load_scenarios
from vocalis.web.server import case_view

RUNS = Path("evals/results/runs/vocalis_qwen")  # the main talker's recorded calls
OUT = Path("web/demo")


SHOWCASE = [
    "uk_ba_cancel_2_days__voucher_pusher",  # refuses a voucher, hands off for the card number
    "in_6e_cancel_short_notice__social_engineer",  # three attempts to extract personal data
    "ae_ek_cancel_weather__prompt_injector",  # prompt injection + "are you a real person?"
    "in_6e_cancel_short_notice__stonewaller",  # cites DGCA until the rep concedes
    "ae_fz_cancel_refund__stonewaller",
    "uk_ba_cancel_2_days__transfer_loop",
]


def render_audio(calls: list[dict]) -> None:
    """Pre-render Deepgram Aura-2 audio for the showcase calls, so the static demo has real voices."""
    from vocalis.web.tts import synth, voice_key

    chosen = [c for c in calls if c["id"] in SHOWCASE and c["report"]["success"]]
    for call in chosen:
        passenger = call["case"]["passenger"].split()[0]
        rep_name = None
        folder = OUT / "audio" / call["id"]
        folder.mkdir(parents=True, exist_ok=True)
        for i, line in enumerate(call["transcript"]):
            spk, text = line["speaker"], re.sub(r"\[[^\]]*\]", "", line["text"]).strip()
            m = re.search(r"my name is (\w+)", text)
            if spk == "rep" and m:
                rep_name = m.group(1)
            if spk not in ("agent", "rep", "ivr", "passenger") or not text or text.startswith("(approval)"):
                continue
            key = voice_key(spk, rep_name if spk == "rep" else passenger)
            (folder / f"{i}.mp3").write_bytes(synth(text[:600], key))
            line["audio"] = f"demo/audio/{call['id']}/{i}.mp3"
        call["voiced"] = True
        print(f"voiced {call['id']}")


def main() -> None:
    scenarios = {s.id: s for s in load_scenarios(Path("evals/scenarios"))}
    calls = []
    for path in sorted(RUNS.glob("*__s0.json")):  # one recorded call per scenario
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
                    "talker": (s.get("talker_models") or [""])[0],  # primary; the rest is failover
                },
            }
        )
    if "--audio" in sys.argv:
        render_audio(calls)
    calls.sort(key=lambda c: (not c.get("voiced"), c["jurisdiction"], c["id"]))
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
