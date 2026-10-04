from __future__ import annotations

from pathlib import Path

from evals.report import check_provenance, provenance
from vocalis.core.provenance import code_hash, snapshot
from vocalis.simair.call import same_model, talker_mismatch

QWEN = "cerebras/qwen-3.8-27b"


def test_served_model_must_be_the_pinned_talker() -> None:
    assert same_model(QWEN, "qwen-3.8-27b")
    assert not same_model(QWEN, "gpt-oss-120b")
    assert talker_mismatch([QWEN], [QWEN], {"qwen-3.8-27b": 12}) is None
    # failover to another model, or a config that silently built a different talker, fails the run
    assert talker_mismatch([QWEN], [QWEN], {"qwen-3.8-27b": 3, "openai/gpt-oss-120b": 1})
    assert talker_mismatch([QWEN], ["cerebras/gpt-oss-120b"], {"gpt-oss-120b": 5})
    assert talker_mismatch(None, ["cerebras/gpt-oss-120b"], {}) is None  # unpinned (web demo)


def test_code_hash_ignores_line_endings_and_tracks_call_code(tmp_path: Path) -> None:
    f = tmp_path / "src/vocalis/agent/x.py"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"a = 1\n")
    h = code_hash(tmp_path)
    f.write_bytes(b"a = 1\r\n")
    assert code_hash(tmp_path) == h
    f.write_bytes(b"a = 2\n")
    assert code_hash(tmp_path) != h
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/x.md").write_text("docs do not change the call")
    f.write_bytes(b"a = 1\n")
    assert code_hash(tmp_path) == h


def test_snapshot_has_commit_and_hash() -> None:
    s = snapshot()
    assert len(s["code_hash"]) == 12 and s["dirty"] in (True, False, None)


def _row(code: str, served: dict[str, int] | None = None) -> dict:
    return {
        "scenario": "x",
        "seed": 0,
        "error": None,
        "provenance": {
            "code_hash": code,
            "git_sha": "abc",
            "talker_pinned": [QWEN],
            "talker_served": served or {"qwen-3.8-27b": 4},
        },
    }


def test_report_refuses_mixed_code_or_wrong_talker() -> None:
    assert check_provenance("c", [_row("h1"), _row("h1")]) == []
    assert check_provenance("c", [_row("h1"), _row("h2")])
    assert check_provenance("c", [_row("h1", {"gpt-oss-120b": 2})])
    legacy = {"scenario": "y", "seed": 0, "error": None, "provenance": None}
    assert provenance([_row("h1"), legacy])["unrecorded_runs"] == 1
