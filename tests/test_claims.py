"""The README may only claim what is built and measured; docs/claims.md maps each claim to evidence."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pytest

from evals.report import SUMMARY, aggregate, load, metrics_table, voice_aggregate

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
LEDGER = (ROOT / "docs" / "claims.md").read_text(encoding="utf-8")

# Sprint items that are designed but not built. Each pattern leaves this list in the change that ships it.
NOT_BUILT = {
    "CallBridge": r"call ?bridge",
    "Silero VAD / smart-turn": r"silero|smart[- ]turn",
    "async planner": r"planner",
    "calibrated judge": r"calibrat",
    "CI safety gate": r"ci (safety )?gate|gates? ci",
    "Postgres / pgvector": r"postgres|pgvector",
    "airline-policy retrieval": r"airline'?s?[- ](own )?polic",
    "take-over in the web page": r"take[- ]?over",
    "Twilio / real calls": r"twilio|pstn",
    "calendar / mail MCP": r"calendar|\bmail\b",
    "Opik on the voice pipeline": r"opik",
    "Sentry / CloudWatch": r"sentry|cloudwatch",
    "Whisper fine-tune": r"whisper (lora|fine|second)|fine-tun",
    "call-state classifier model": r"classifier",
}


def _summary() -> dict[str, Any]:
    return json.loads((ROOT / SUMMARY).read_text(encoding="utf-8"))


def _ledger_rows() -> list[list[str]]:
    """Table rows of the 'What it does' section: claim | code | tests | evidence."""
    section = LEDGER.split("## What it does", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and not set(cells[0]) <= {"-"} and cells[0] != "Claim":
            rows.append(cells)
    return rows


def _ticked(cell: str) -> list[str]:
    return re.findall(r"`([^`]+)`", cell)


def test_readme_metrics_are_generated_from_summary() -> None:
    block = re.search(r"<!-- metrics:start -->\n(.*?)\n<!-- metrics:end -->", README, re.S)
    assert block, "README lost its generated metrics block"
    assert block.group(1) == metrics_table(_summary()), "README metrics drifted: run `python -m evals.report`"


@pytest.mark.parametrize("role", ["text", "voice", "baseline"])
def test_summary_matches_the_stored_runs(role: str) -> None:
    summary = _summary()
    config = summary["primary"][role]
    rows = load(config)
    assert rows, f"no stored runs for {config}"
    fresh = aggregate(rows)
    if any(r.get("voice_turns") for r in rows):
        fresh["voice"] = voice_aggregate(rows)
    assert json.loads(json.dumps(fresh)) == summary[config], f"summary.json is stale for {config}"


def test_ledger_has_rows() -> None:
    assert len(_ledger_rows()) >= 15


@pytest.mark.parametrize("row", _ledger_rows(), ids=lambda r: r[0][:50])
def test_ledger_code_paths_exist(row: list[str]) -> None:
    paths = [p for p in _ticked(row[1]) if "/" in p]
    assert paths, f"claim has no code: {row[0]}"
    missing = [p for p in paths if not (ROOT / p).exists()]
    assert not missing, f"{row[0]}: missing {missing}"


def _test_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name.startswith("test_")
    }


@pytest.mark.parametrize("row", _ledger_rows(), ids=lambda r: r[0][:50])
def test_ledger_tests_exist(row: list[str]) -> None:
    for ref in _ticked(row[2]):
        file, _, name = ref.partition("::")
        path = ROOT / file
        assert path.exists(), f"{row[0]}: no test file {file}"
        assert name in _test_names(path), f"{row[0]}: no test {ref}"


def _resolve(summary: dict[str, Any], key: str) -> Any:
    primary = summary["primary"]
    for role in ("text", "voice", "baseline"):
        key = key.replace(f"<{role}>", primary[role])
    # config names may contain '/', so match the longest top-level key first
    top = max((k for k in summary if key == k or key.startswith(k + ".")), key=len, default=None)
    assert top is not None, f"no summary entry for {key}"
    node: Any = summary[top]
    for part in key[len(top) + 1 :].split(".") if key != top else []:
        assert isinstance(node, dict) and part in node, f"no summary key {key}"
        node = node[part]
    return node


@pytest.mark.parametrize("row", _ledger_rows(), ids=lambda r: r[0][:50])
def test_ledger_metric_keys_exist(row: list[str]) -> None:
    summary = _summary()
    for key in _ticked(row[3]):
        assert _resolve(summary, key) is not None, f"{row[0]}: {key} is empty"


def test_readme_components_exist() -> None:
    section = README.split("### Components", 1)[1].split("\n### ", 1)[0]
    paths = [p for line in section.splitlines() for p in _ticked(line) if "/" in p]
    assert len(paths) >= 10
    missing = [p for p in paths for part in p.split(", ") if not (ROOT / part).exists()]
    assert not missing, missing


@pytest.mark.parametrize("feature", NOT_BUILT)
def test_readme_does_not_claim_unbuilt(feature: str) -> None:
    hits = re.findall(rf".{{0,40}}(?:{NOT_BUILT[feature]}).{{0,40}}", README, re.I)
    assert not hits, f"README mentions {feature}, which is not built: {hits}"


def test_ledger_lists_every_unbuilt_item() -> None:
    section = LEDGER.split("## Not built", 1)[1].split("\n## ", 1)[0].lower()
    for feature, pattern in NOT_BUILT.items():
        assert re.search(pattern, section, re.I), f"docs/claims.md does not list {feature} as not built"
