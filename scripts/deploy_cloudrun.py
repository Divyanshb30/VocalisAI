"""Deploy the live backend to Google Cloud Run (free tier, scales to zero) and point the public demo at it.

    uv run python scripts/deploy_cloudrun.py [--region us-central1] [--service vocalisai]

Needs the gcloud CLI logged in with a project selected. API keys are read from .env and passed
through a temporary env-vars file that is deleted afterwards; they never enter git or the
command line.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
KEYS = [
    "CEREBRAS_API_KEY",
    "GROQ_API_KEY",
    "GOOGLE_API_KEY",
    "DEEPGRAM_API_KEY",
    "OPIK_API_KEY",
    "OPIK_WORKSPACE",
]
RUNTIME = {
    # live demo: Groq first (30 req/min) — Cerebras' free 5 req/min is too low for live turns
    "VOCALIS_TALKER_MODELS": "groq/openai/gpt-oss-120b,cerebras/gpt-oss-120b,gemini/gemini-flash-lite-latest",
    # no local Ollama in the cloud: the simulated airline rep runs on free-tier models
    "VOCALIS_REP_MODELS": "groq/openai/gpt-oss-20b,gemini/gemini-flash-lite-latest",
    "VOCALIS_LIVE_PER_IP": "3",
    "VOCALIS_LIVE_PER_DAY": "40",
}


def gcloud() -> str:
    exe = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if not exe:
        raise SystemExit("gcloud CLI not found")
    return exe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--service", default="vocalisai")
    a = ap.parse_args()

    env = dotenv_values(ROOT / ".env")
    values = {k: env[k] for k in KEYS if env.get(k)} | RUNTIME
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(values, f)
        env_file = f.name
    try:
        subprocess.run(
            [
                gcloud(),
                "run",
                "deploy",
                a.service,
                "--source",
                str(ROOT),
                "--region",
                a.region,
                "--allow-unauthenticated",
                "--memory",
                "2Gi",
                "--cpu",
                "1",
                "--timeout",
                "3600",
                "--min-instances",
                "0",
                "--max-instances",
                "2",
                "--concurrency",
                "20",
                "--env-vars-file",
                env_file,
                "--quiet",
            ],
            check=True,
        )
    finally:
        Path(env_file).unlink(missing_ok=True)

    url = subprocess.run(
        [
            gcloud(),
            "run",
            "services",
            "describe",
            a.service,
            "--region",
            a.region,
            "--format",
            "value(status.url)",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (ROOT / "web" / "config.js").write_text(
        f'window.VOCALIS_API = "{url}";  // hosted backend (Cloud Run). Override with ?api=...\n',
        encoding="utf-8",
    )
    print(f"Backend: {url}")
    print('Then: git add web/config.js && git commit -m "point demo at the live backend" && git push')


if __name__ == "__main__":
    main()
