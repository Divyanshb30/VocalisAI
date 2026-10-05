"""The captured-document benchmark: fabricated documents, each at three severities, reproducible."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from evals.docbench.capture import SEVERITIES, generate


def test_each_document_is_captured_at_every_severity_reproducibly(tmp_path: Path) -> None:
    generate(4, tmp_path / "a", seed=3)
    generate(4, tmp_path / "b", seed=3)
    labels = json.loads((tmp_path / "a" / "labels.json").read_text(encoding="utf-8"))
    assert len(labels) == 4 * len(SEVERITIES)
    kinds = {v["kind"] for v in labels.values()}
    assert kinds == {"boarding_pass", "email_cancellation", "email_delay", "eticket"}
    for i in range(4):
        truths = [
            {k: v for k, v in labels[f"doc_{i:03d}_{s}.jpg"].items() if k != "severity"} for s in SEVERITIES
        ]
        assert truths[0] == truths[1] == truths[2]  # same document, only the photo differs
    for name in labels:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    light = Image.open(tmp_path / "a" / "doc_001_light.jpg")
    heavy = Image.open(tmp_path / "a" / "doc_001_heavy.jpg")
    assert heavy.width * heavy.height < light.width * light.height  # farther away: fewer pixels on the text


def test_fabricated_documents_use_example_domains_only(tmp_path: Path) -> None:
    generate(8, tmp_path, seed=5)
    labels = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    emails = {v.get("email") for v in labels.values() if v.get("email")}
    assert emails and all(e.endswith("@example.com") for e in emails)
