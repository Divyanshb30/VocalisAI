"""Build the Pipecat talker LLM from the configured free-tier model list, with failover."""

from __future__ import annotations

from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.pipeline.service_switcher import ServiceSwitcherStrategyFailover
from pipecat.services.llm_service import LLMService

from vocalis.core.settings import Settings, get_settings
from vocalis.llm.router import QuotaLedger, _has_key, provider_of


def _one(model: str, s: Settings, temperature: float) -> LLMService | None:
    provider, _, name = model.partition("/")
    if provider == "cerebras" and s.cerebras_api_key:
        from pipecat.services.cerebras.llm import CerebrasLLMService

        extra = {"reasoning_effort": "low"} if "gpt-oss" in name else {}
        return CerebrasLLMService(
            api_key=s.cerebras_api_key,
            settings=CerebrasLLMService.Settings(model=name, temperature=temperature, extra=extra),
        )
    if provider == "groq" and s.groq_api_key:
        from pipecat.services.groq.llm import GroqLLMService

        extra = {"reasoning_effort": "low"} if "gpt-oss" in name else {}
        return GroqLLMService(
            api_key=s.groq_api_key,
            settings=GroqLLMService.Settings(model=name, temperature=temperature, extra=extra),
        )
    if provider == "gemini" and s.google_api_key:
        from pipecat.services.google.llm import GoogleLLMService

        return GoogleLLMService(
            api_key=s.google_api_key,
            settings=GoogleLLMService.Settings(model=name, temperature=temperature),
        )
    if provider in ("ollama", "ollama_chat"):
        from pipecat.services.ollama.llm import OLLamaLLMService

        return OLLamaLLMService(
            base_url=s.ollama_base_url.rstrip("/") + "/v1",
            settings=OLLamaLLMService.Settings(model=name, temperature=temperature),
        )
    return None


def build_talker(
    models: list[str] | None = None,
    *,
    temperature: float = 0.3,
    settings: Settings | None = None,
    ledger: QuotaLedger | None = None,
) -> tuple[LLMService, list[str]]:
    """Return a talker LLM (an LLMSwitcher when several providers are usable) and the model ids used."""
    s = settings or get_settings()
    ledger = ledger or QuotaLedger()
    wanted = models or s.talker_models
    usable = [m for m in wanted if _has_key(m, s) and ledger.remaining(m) > 20_000]
    services: list[tuple[str, LLMService]] = []
    for m in usable:
        svc = _one(m, s, temperature)
        if svc is not None:
            services.append((m, svc))
    if not services:
        raise RuntimeError(f"No usable talker model among {wanted}; add a key to .env")
    if len(services) == 1:
        return services[0][1], [services[0][0]]
    switcher = LLMSwitcher(llms=[svc for _, svc in services], strategy_type=ServiceSwitcherStrategyFailover)
    return switcher, [m for m, _ in services]  # type: ignore[return-value]


__all__ = ["build_talker", "provider_of"]
