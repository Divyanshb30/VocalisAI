"""Quota-aware router over free-tier LLM providers. See docs/adr/0009-free-tier-llm-router.md.

Each role (talker, planner, judge, rep) has an ordered list of models. Calls go to the
first model whose provider has a key and remaining daily budget; on error or rate limit
the next model is tried. Token usage per model per day is persisted so a long eval batch
does not blow through a provider's daily cap.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import litellm
from loguru import logger

from vocalis.core.settings import Settings, get_settings

litellm.suppress_debug_info = True

# Conservative daily token budgets for free tiers (leave headroom below the published caps).
DAILY_TOKEN_BUDGET = {
    "cerebras": 900_000,
    "groq": 180_000,
    "gemini": 2_000_000,
    "ollama_chat": 10**12,
    "ollama": 10**12,
}
_QUOTA_FILE = Path(".cache/llm_quota.json")


def provider_of(model: str) -> str:
    return model.split("/", 1)[0]


def _has_key(model: str, s: Settings) -> bool:
    p = provider_of(model)
    return {
        "cerebras": bool(s.cerebras_api_key),
        "groq": bool(s.groq_api_key),
        "gemini": bool(s.google_api_key),
        "ollama_chat": True,
        "ollama": True,
    }.get(p, True)


def _litellm_kwargs(model: str, s: Settings) -> dict[str, Any]:
    p = provider_of(model)
    if p == "cerebras":
        return {"api_key": s.cerebras_api_key}
    if p == "groq":
        return {"api_key": s.groq_api_key}
    if p == "gemini":
        return {"api_key": s.google_api_key}
    if p in ("ollama", "ollama_chat"):
        # think=False: qwen3 otherwise spends the whole token budget thinking and returns no content
        return {"api_base": s.ollama_base_url, "think": False}
    return {}


class QuotaLedger:
    def __init__(self, path: Path = _QUOTA_FILE) -> None:
        self.path = path
        self._data: dict[str, dict[str, int]] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except json.JSONDecodeError:
                self._data = {}

    def _today(self) -> dict[str, int]:
        return self._data.setdefault(date.today().isoformat(), {})

    def used(self, model: str) -> int:
        return self._today().get(provider_of(model), 0)

    def remaining(self, model: str) -> int:
        return DAILY_TOKEN_BUDGET.get(provider_of(model), 10**9) - self.used(model)

    def add(self, model: str, tokens: int) -> None:
        today = self._today()
        today[provider_of(model)] = today.get(provider_of(model), 0) + tokens
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=1))


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    raw: Any = None


class LLMRouter:
    def __init__(self, settings: Settings | None = None, ledger: QuotaLedger | None = None) -> None:
        self.s = settings or get_settings()
        self.ledger = ledger or QuotaLedger()

    def candidates(self, models: Sequence[str], need_tokens: int = 4000) -> list[str]:
        return [m for m in models if _has_key(m, self.s) and self.ledger.remaining(m) > need_tokens]

    async def complete(
        self,
        models: Sequence[str],
        messages: list[dict[str, Any]],
        *,
        json_mode: bool = False,
        temperature: float = 0.4,
        max_tokens: int = 800,
        tags: dict[str, str] | None = None,
    ) -> LLMResult:
        errors: list[str] = []
        for model in self.candidates(models):
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=60,
                num_retries=1,
                **_litellm_kwargs(model, self.s),
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if tags:
                kwargs["metadata"] = tags
            t0 = time.perf_counter()
            try:
                resp = await litellm.acompletion(**kwargs)
            except Exception as exc:  # rate limit, provider down, bad model id
                errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:160]}")
                logger.warning(f"LLM fallback: {errors[-1]}")
                continue
            usage = getattr(resp, "usage", None)
            tin = int(getattr(usage, "prompt_tokens", 0) or 0)
            tout = int(getattr(usage, "completion_tokens", 0) or 0)
            self.ledger.add(model, tin + tout)
            text = resp.choices[0].message.content or ""
            return LLMResult(text, model, tin, tout, time.perf_counter() - t0, resp)
        raise RuntimeError("All LLM candidates failed: " + " | ".join(errors or ["no model has a key/quota"]))


def enable_opik_tracing() -> bool:
    """Send LiteLLM calls to Opik when a key is configured."""
    s = get_settings()
    if not s.opik_api_key:
        return False
    os.environ.setdefault("OPIK_API_KEY", s.opik_api_key)
    if s.opik_workspace:
        os.environ.setdefault("OPIK_WORKSPACE", s.opik_workspace)
    os.environ.setdefault("OPIK_PROJECT_NAME", s.opik_project_name)
    if "opik" not in litellm.callbacks:
        litellm.callbacks = [*litellm.callbacks, "opik"]
    return True
