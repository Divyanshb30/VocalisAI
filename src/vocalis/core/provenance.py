"""Where a run came from: the commit, whether the code that ran was committed, and a hash of it.

The hash covers only the code that shapes a call (agent, guards, simulator, scenarios, ...), so a
docs-only commit does not split one config's runs into two code versions.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CALL_CODE = (
    "src/vocalis/agent",
    "src/vocalis/core",
    "src/vocalis/guards",
    "src/vocalis/llm",
    "src/vocalis/rights",
    "src/vocalis/simair",
    "src/vocalis/telephony",
    "evals/run.py",
    "evals/scenarios",
    "data/policy",
)
_SUFFIXES = {".py", ".yaml", ".yml", ".txt"}


def code_hash(root: Path = ROOT) -> str:
    h = hashlib.sha256()
    for entry in CALL_CODE:
        p = root / entry
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix in _SUFFIXES)
        for f in files:
            if "__pycache__" in f.parts:
                continue
            h.update(f.relative_to(root).as_posix().encode())
            h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def _git(*args: str, root: Path = ROOT) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def snapshot(root: Path = ROOT) -> dict[str, Any]:
    """Commit, uncommitted changes to the call code, and the code hash, taken once per eval process."""
    status = _git("status", "--porcelain", "--", *CALL_CODE, root=root)
    return {
        "git_sha": _git("rev-parse", "--short=12", "HEAD", root=root),
        "dirty": bool(status) if status is not None else None,
        "code_hash": code_hash(root),
    }
