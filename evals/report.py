"""Aggregate run results into summary.json, docs/evals.md and the README metrics table.

uv run python -m evals.report
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

RUNS = Path("evals/results/runs")
SUMMARY = Path("evals/results/summary.json")
DOCBENCH = Path("evals/results/docbench.json")
EVALS_MD = Path("docs/evals.md")
README = Path("README.md")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}


def pct(v: float | None) -> str:
    return "—" if v is None else f"{100 * v:.0f}%"


def fmt_rate(r: dict[str, Any]) -> str:
    if not r["n"]:
        return "—"
    return f"{pct(r['rate'])} ({r['k']}/{r['n']}, 95% CI {pct(r['ci95'][0])}–{pct(r['ci95'][1])})"


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    idx = min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))
    return xs[idx]


def load(config: str) -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text(encoding="utf-8"))["score"] for p in sorted((RUNS / config).glob("*.json"))
    ]


def _legacy_neutral(r: dict[str, Any]) -> int:
    """Runs scored before injection-turn handoffs were neutral: subtract them here."""
    if "handoff_neutral" in r:
        return 0
    return sum(1 for h in r["handoffs"] if h.get("event") == "inject")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in rows if not r.get("error")]
    n = len(done)
    tp = sum(r["handoff_tp"] for r in done)
    fp = sum(max(0, r["handoff_fp"] - _legacy_neutral(r)) for r in done)
    fn = sum(r["handoff_fn"] for r in done)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    lat = [x for r in done for x in r["latencies_s"]]
    leaked = sum(r["leaked"] for r in done)
    judged = [r["judge"] for r in done if isinstance(r.get("judge"), dict) and "overall" in r["judge"]]
    by = lambda key: {  # noqa: E731
        k: rate(sum(r["success"] for r in g), len(g)) for k, g in _group(done, key).items()
    }
    return {
        "runs": len(rows),
        "completed": n,
        "errors": len(rows) - n,
        "task_success": rate(sum(r["success"] for r in done), n),
        "target_achieved": rate(sum(r["target_achieved"] for r in done), n),
        "reference_captured": rate(sum(r["reference_captured"] for r in done), n),
        "leak_runs": rate(leaked, n),
        # exact one-sided 95% upper bound for 0 events in n trials (~3/n, the rule of three)
        "leak_upper_bound_rule_of_three": round(1 - 0.05 ** (1 / n), 4) if n and leaked == 0 else None,
        "mandate_violations": sum(r["mandate_violation"] for r in done),
        "handoff": {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1},
        "disclosure_first_utterance": rate(sum(r["disclosed_first"] for r in done), n),
        "honest_when_asked_if_human": rate(
            sum(r["human_checks_honest"] for r in done if r["human_checks"]),
            sum(1 for r in done if r["human_checks"]),
        ),
        "ivr_reached_agent": rate(sum(r["ivr_reached_queue"] for r in done), n),
        "llm_calls_during_hold": sum(r["hold_llm_calls"] for r in done),
        "reply_latency_s": {
            "p50": percentile(lat, 0.5),
            "p95": percentile(lat, 0.95),
            "n": len(lat),
            "note": "text mode: rep text in -> first guarded sentence out (no STT/TTS)",
        },
        "judge": {
            "n": len(judged),
            "overall_mean": round(statistics.mean(j["overall"] for j in judged), 2) if judged else None,
            "invented_facts_runs": sum(bool(j.get("invented_facts")) for j in judged),
        },
        "success_by_jurisdiction": by("jurisdiction"),
        "success_by_persona": by("persona"),
        "talker_models": sorted({m for r in done for m in r.get("talker_models", [])}),
        "talker_tokens_per_call": round(statistics.mean(r.get("talker_tokens", 0) for r in done))
        if done
        else None,
        "llm_ttfb_s": {
            "p50": percentile([x for r in done for x in r.get("llm_ttfb_s", [])], 0.5),
            "p95": percentile([x for r in done for x in r.get("llm_ttfb_s", [])], 0.95),
            "n": sum(len(r.get("llm_ttfb_s", [])) for r in done),
        },
        "actual_cost_usd": 0.0,
    }


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    g: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return dict(sorted(g.items()))


def metrics_table(s: dict[str, Any]) -> str:
    """Only rows that have a real measured value; nothing is shown for metrics not yet run."""
    v = s.get("vocalis")
    b = s.get("baseline")
    doc = s.get("docbench")
    rows: list[tuple[str, str, str]] = []
    if v and v["completed"]:
        h = v["handoff"]
        lat = v["reply_latency_s"]
        leak = fmt_rate(v["leak_runs"])
        if v["leak_upper_bound_rule_of_three"] is not None:
            leak += f"; 95% upper bound {pct(v['leak_upper_bound_rule_of_three'])}"
        rows.append(
            (
                "Task success",
                "outcome inside the mandate **and** correct reference captured",
                fmt_rate(v["task_success"]),
            )
        )
        rows.append(
            ("Sensitive-data leak rate", "runs where any unauthorised value or canary was spoken", leak)
        )
        m = s.get("vocalis_on_baseline_scenarios")
        if b and b["completed"] and m:
            rows.append(
                (
                    "Leak rate vs naive baseline",
                    "same attack scenarios: VocalisAI vs secrets-in-prompt with guards off",
                    f"{fmt_rate(m['leak_runs'])} vs {fmt_rate(b['leak_runs'])} "
                    f"(talkers: {', '.join(m['talker_models'])} vs {', '.join(b['talker_models'])})",
                )
            )
        if h["f1"] is not None:
            rows.append(
                (
                    "Handoff accuracy",
                    "precision / recall / F1 on events that need the passenger",
                    f"P {pct(h['precision'])} · R {pct(h['recall'])} · F1 {pct(h['f1'])}",
                )
            )
        disc = fmt_rate(v["disclosure_first_utterance"])
        if v["honest_when_asked_if_human"]["n"]:
            disc += f"; honest when asked: {fmt_rate(v['honest_when_asked_if_human'])}"
        rows.append(("AI disclosure", "discloses in the first utterance", disc))
        rows.append(
            ("IVR navigation", "reached a human through the phone menu", fmt_rate(v["ivr_reached_agent"]))
        )
        if lat["p50"] is not None:
            rows.append(
                (
                    "Reply latency (text mode)",
                    "rep turn in → first guarded sentence out, p50 / p95",
                    f"{lat['p50']:.2f}s / {lat['p95']:.2f}s (n={lat['n']})",
                )
            )
        t = v.get("llm_ttfb_s") or {}
        if t.get("p50") is not None:
            rows.append(
                (
                    "Talker first-token latency",
                    "LLM time to first token per turn, p50 / p95",
                    f"{t['p50']:.2f}s / {t['p95']:.2f}s (n={t['n']})",
                )
            )
        if v.get("talker_tokens_per_call"):
            rows.append(
                (
                    "Cost per call",
                    "talker tokens per call; dollars actually paid",
                    f"{v['talker_tokens_per_call']:,} tokens; $0.00 (free tiers + local rep)",
                )
            )
    if doc:
        ds = doc["summary"]
        if ds.get("documents"):
            rows.append(
                (
                    "Document extraction",
                    "field accuracy, vision only → with barcode cross-check",
                    f"{pct(ds['field_accuracy_vision'])} → {pct(ds['field_accuracy_with_barcode'])} "
                    f"({ds['documents'] - ds['errors']} synthetic docs, {ds['model']})",
                )
            )
    if not rows:
        return ""
    out = ["| Metric | Definition | Result |", "|---|---|---|"]
    out += [f"| {a} | {b_} | {c} |" for a, b_, c in rows]
    if v and v["completed"]:
        out.append("")
        out.append(
            f"_{v['completed']} completed simulated calls; talker: {', '.join(v['talker_models']) or 'n/a'}._"
        )
    return "\n".join(out)


def main() -> None:
    summary: dict[str, Any] = {}
    for config in ("vocalis", "baseline"):
        rows = load(config)
        if rows:
            summary[config] = aggregate(rows)
    base_ids = {r["scenario"] for r in load("baseline")}
    if base_ids:
        # Prefer runs made with the same talker as the baseline (vocalis_matched), if they exist.
        matched = [r for r in load("vocalis_matched") if r["scenario"] in base_ids] or [
            r for r in load("vocalis") if r["scenario"] in base_ids
        ]
        if matched:
            summary["vocalis_on_baseline_scenarios"] = aggregate(matched)
    if DOCBENCH.exists():
        summary["docbench"] = json.loads(DOCBENCH.read_text(encoding="utf-8"))
        summary["docbench"].pop("per_doc", None)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    table = metrics_table(summary)
    if table:
        EVALS_MD.write_text(
            "# Evaluation results\n\nGenerated by `uv run python -m evals.report` from `evals/results/runs/`.\n\n"
            + table
            + "\n\n## Full summary\n\n```json\n"
            + json.dumps(summary, indent=1)
            + "\n```\n",
            encoding="utf-8",
        )
    readme = README.read_text(encoding="utf-8")
    new = re.sub(
        r"(<!-- metrics:start -->)(.*?)(<!-- metrics:end -->)",
        lambda m: f"{m.group(1)}\n{table}\n{m.group(3)}" if table else f"{m.group(1)}\n{m.group(3)}",
        readme,
        flags=re.S,
    )
    README.write_text(new, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(table)


if __name__ == "__main__":
    main()
