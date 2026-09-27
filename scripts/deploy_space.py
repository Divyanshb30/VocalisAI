"""Deploy the live backend to a free Hugging Face Space (Docker) and point the public demo at it.

    1. Put a Hugging Face *write* token in .env as HF_TOKEN=...
    2. uv run python scripts/deploy_space.py
    3. git add web/config.js && git commit -m "point demo at the live backend" && git push

Secrets (your API keys) are read from .env and stored as Space secrets; they never enter git.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SECRETS = [
    "CEREBRAS_API_KEY",
    "GROQ_API_KEY",
    "GOOGLE_API_KEY",
    "DEEPGRAM_API_KEY",
    "OPIK_API_KEY",
    "OPIK_WORKSPACE",
]
SPACE_README = """---
title: VocalisAI
emoji: 📞
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: AI agent that phones the airline for you
---

Live backend for [VocalisAI](https://github.com/Divyanshb30/VocalisAI).
"""


def main() -> None:
    env = {**dotenv_values(ROOT / ".env"), **os.environ}
    token = env.get("HF_TOKEN")
    if not token:
        sys.exit("Add HF_TOKEN=<a Hugging Face write token> to .env first (huggingface.co/settings/tokens).")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/vocalisai"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    for key in SECRETS:
        if env.get(key):
            api.add_space_secret(repo_id, key, env[key])
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=ROOT,
        allow_patterns=[
            "Dockerfile",
            "pyproject.toml",
            "uv.lock",
            "src/**",
            "web/**",
            "evals/scenarios/**",
            "evals/results/summary.json",
        ],
        ignore_patterns=["**/__pycache__/**", "web/demo/audio/**"],
        commit_message="deploy",
    )
    api.upload_file(
        path_or_fileobj=SPACE_README.encode(), path_in_repo="README.md", repo_id=repo_id, repo_type="space"
    )
    url = f"https://{re.sub(r'[^a-z0-9-]', '-', user.lower())}-vocalisai.hf.space"
    (ROOT / "web" / "config.js").write_text(
        f'window.VOCALIS_API = "{url}";  // hosted backend (Hugging Face Space). Override with ?api=...\n',
        encoding="utf-8",
    )
    print(f"Space: https://huggingface.co/spaces/{repo_id}  (first build takes ~5-10 min)")
    print(f"API:   {url}")
    print('Now: git add web/config.js && git commit -m "point demo at the live backend" && git push')


if __name__ == "__main__":
    main()
