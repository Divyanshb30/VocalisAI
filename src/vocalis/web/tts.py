"""Deepgram Aura-2 voices for the web app, cached on disk so a line is only paid for once."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

from vocalis.core.settings import get_settings

CACHE = Path(".cache/tts")

# role -> Aura-2 voice (accent / tone picked per speaker)
VOICES = {
    "agent": "aura-2-draco-en",  # British, warm: the VocalisAI agent
    "ivr": "aura-2-electra-en",  # professional: the phone menu
    "rep_f": "aura-2-asteria-en",
    "rep_m": "aura-2-mars-en",
    "pass_f": "aura-2-luna-en",
    "pass_m": "aura-2-orion-en",
}

FEMALE_NAMES = {"asha", "emily", "leila", "priya", "sophie", "aisha", "neha", "sara", "meera"}


def voice_key(speaker: str, name: str | None = None) -> str:
    female = (name or "").strip().lower() in FEMALE_NAMES
    if speaker == "rep":
        return "rep_f" if female else "rep_m"
    if speaker == "passenger":
        return "pass_f" if female else "pass_m"
    return "ivr" if speaker == "ivr" else "agent"


def available() -> bool:
    return bool(get_settings().deepgram_api_key)


def synth(text: str, key: str) -> bytes:
    """MP3 for ``text`` in the voice for ``key``; cached by content."""
    model = VOICES.get(key, VOICES["agent"])
    digest = hashlib.sha1(f"{model}|{text}".encode()).hexdigest()
    path = CACHE / f"{digest}.mp3"
    if path.exists():
        return path.read_bytes()
    resp = httpx.post(
        "https://api.deepgram.com/v1/speak",
        params={"model": model, "encoding": "mp3"},
        headers={"Authorization": f"Token {get_settings().deepgram_api_key}"},
        json={"text": text},
        timeout=30,
    )
    resp.raise_for_status()
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return resp.content
