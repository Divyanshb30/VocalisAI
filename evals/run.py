"""Run SimAir scenarios and write per-run results.

uv run python -m evals.run --config vocalis --seeds 0 1 2
uv run python -m evals.run --config baseline --only in_6e --seeds 0
uv run python -m evals.run --smoke           # CI safety subset
uv run python -m evals.run --config vocalis_voice --seeds 0 --concurrency 1   # audio loopback (Deepgram)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from evals.judge import judge
from vocalis.agent.briefing import Briefing
from vocalis.core.provenance import snapshot
from vocalis.llm.router import LLMRouter, enable_opik_tracing
from vocalis.rights.engine import assess
from vocalis.simair.call import CallSimulation, RunConfig
from vocalis.simair.scenario import Scenario, load_scenarios
from vocalis.simair.scoring import SCORING_VERSION, score

RESULTS = Path("evals/results/runs")
SMOKE = {
    "in_6e_cancel_short_notice__social_engineer",
    "in_6e_cancel_short_notice__prompt_injector",
    "uk_ba_cancel_2_days__cooperative",
    "ae_fz_cancel_refund__social_engineer",
    "uk_vs_delay_long_haul__social_engineer",
    "ae_ek_cancel_weather__prompt_injector",
}

QWEN = "cerebras/qwen-3.8-27b"  # the main talker
GPT_OSS = ["cerebras/gpt-oss-120b", "groq/openai/gpt-oss-120b", "gemini/gemini-flash-lite-latest"]
CONFIGS: dict[str, dict[str, Any]] = {
    "vocalis": dict(guards=True, baseline_prompt_secrets=False, talker_models=GPT_OSS),
    "baseline": dict(guards=False, baseline_prompt_secrets=True, talker_models=GPT_OSS),
    # Same guards as "vocalis", kept separate so a same-talker comparison with "baseline" is possible
    "vocalis_matched": dict(guards=True, baseline_prompt_secrets=False, talker_models=GPT_OSS),
    # Same agent, but the far end is heard as phone audio through streaming STT and replies are timed
    # to the first TTS audio byte (see vocalis/telephony/voicelink.py)
    "vocalis_voice": dict(guards=True, baseline_prompt_secrets=False, talker_models=GPT_OSS),
    # Same agent with a different talker model, for a model comparison (Cerebras qwen-3.8-27b, 450 req/min)
    "vocalis_qwen": dict(guards=True, baseline_prompt_secrets=False, talker_models=[QWEN]),
    "vocalis_qwen_voice": dict(guards=True, baseline_prompt_secrets=False, talker_models=[QWEN]),
    # naive baseline with the main talker: secrets in the prompt, guards off
    "baseline_qwen": dict(guards=False, baseline_prompt_secrets=True, talker_models=[QWEN]),
}


PROVENANCE: dict[str, Any] = {}  # taken once per process, before any call runs


async def run_one(
    sc: Scenario, config: str, seed: int, router: LLMRouter, with_judge: bool, voice: Any = None
) -> dict:
    cfg = RunConfig(seed=seed, label=config, pace=True, voice=voice, strict_talker=True, **CONFIGS[config])
    sim = CallSimulation(sc, cfg, router)
    t0 = time.perf_counter()
    result = await sim.run()
    scored = score(result, sc)
    if with_judge and not result.error:
        briefing = Briefing(sc.case, assess(sc.case), sc.mandate, sim.vault)
        scored["judge"] = await judge(result, briefing, router, router.s.judge_models)
    out = RESULTS / config / f"{sc.id}__s{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "score": scored,
        "transcript": [asdict(l) for l in result.transcript],
        "granted": {
            "outcome": result.granted.outcome.value if result.granted and result.granted.outcome else None,
            "amount": str(result.granted.amount) if result.granted and result.granted.amount else None,
        },
        "reference_issued": result.reference_issued,
        "agent_resolution": result.agent_resolution,
        "offers": result.offers,
        "guard_blocks": result.guard_blocks,
        "provenance": {
            **PROVENANCE,
            "scoring_version": SCORING_VERSION,
            "talker_pinned": cfg.talker_models,
            "talker_built": result.talker_models,
            "talker_served": result.talker_served_models,
            "rep_served": result.rep_served_models,
            "judge_models": router.s.judge_models if with_judge else [],
            **({"stt": voice.stt_model, "tts": voice.agent_voice} if voice is not None else {}),
        },
    }
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    logger.info(
        f"[{config}] {sc.id} s{seed}: success={scored['success']} leaks={scored['leaks']} "
        f"handoff tp/fp/fn={scored['handoff_tp']}/{scored['handoff_fp']}/{scored['handoff_fn']} "
        f"err={scored['error']} ({time.perf_counter() - t0:.0f}s)"
    )
    return scored


async def cerebras_hour_used(router: LLMRouter, talker: str) -> int | None:
    """Requests Cerebras counts against this model in the last hour (a 1-token call, read from its headers)."""
    model = talker.split("/", 1)[1]
    try:
        async with httpx.AsyncClient(timeout=30) as http:
            r = await http.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {router.s.cerebras_api_key}"},
                json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            )
        used = int(r.headers["x-ratelimit-limit-requests-hour"]) - int(
            r.headers["x-ratelimit-remaining-requests-hour"]
        )
    except Exception as exc:
        logger.warning(f"could not read the hourly request budget: {exc}")
        return None
    logger.info(f"cerebras: {used} requests used in the last hour")
    return used


async def watch_hourly_budget(router: LLMRouter, talker: str) -> None:
    """Pace against a model's rolling requests/hour cap (gpt-oss-120b: 150), from what Cerebras has counted."""
    if talker not in CallSimulation.HOURLY or not router.s.cerebras_api_key:
        return
    CallSimulation.hour_probe = lambda: cerebras_hour_used(router, talker)
    used = await cerebras_hour_used(router, talker)
    if used:
        CallSimulation.hour_log[talker] = [time.monotonic()] * used


async def main_async(a: argparse.Namespace) -> None:
    PROVENANCE.update(snapshot())
    if PROVENANCE["dirty"] and not a.allow_dirty:
        raise SystemExit(
            "call code has uncommitted changes; commit first (or pass --allow-dirty for a trial run)"
        )
    logger.info(f"provenance: {PROVENANCE}")
    enable_opik_tracing()
    router = LLMRouter()
    try:  # load the local rep model into memory before timing anything
        await router.complete(
            router.s.rep_models,
            [{"role": "user", "content": "Reply with {} only."}],
            json_mode=True,
            max_tokens=10,
        )
    except Exception as exc:
        logger.warning(f"rep warm-up failed: {exc}")
    talker = (CONFIGS[a.config].get("talker_models") or router.s.talker_models or [""])[0]
    await watch_hourly_budget(router, talker)
    scenarios = load_scenarios(Path("evals/scenarios"))
    if a.smoke:
        scenarios = [s for s in scenarios if s.id in SMOKE]
    if a.only:
        scenarios = [s for s in scenarios if any(o in s.id for o in a.only)]
    if a.limit:
        scenarios = scenarios[: a.limit]
    voice = None
    if a.config.endswith("_voice"):
        from vocalis.telephony.voicelink import VoiceLink

        voice = VoiceLink(router.s.deepgram_api_key or "")
        await voice.warm_up()
    sem = asyncio.Semaphore(a.concurrency)
    results: list[dict] = []

    async def guarded(sc: Scenario, seed: int) -> None:
        out = RESULTS / a.config / f"{sc.id}__s{seed}.json"
        if a.resume and out.exists():
            return
        async with sem:
            try:
                results.append(await run_one(sc, a.config, seed, router, not a.no_judge, voice))
            except Exception as exc:
                logger.exception(f"{sc.id} s{seed} crashed: {exc}")

    await asyncio.gather(*(guarded(sc, seed) for seed in a.seeds for sc in scenarios))
    if voice is not None:
        await voice.aclose()
    ok = [r for r in results if not r["error"]]
    if results:
        print(
            f"\n{a.config}: {len(results)} runs, {len(ok)} completed, "
            f"success {sum(r['success'] for r in ok)}/{len(ok)}, "
            f"leaked runs {sum(r['leaked'] for r in ok)}/{len(ok)}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=sorted(CONFIGS), default="vocalis")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip runs that already have results")
    ap.add_argument(
        "--allow-dirty", action="store_true", help="run with uncommitted call code (not for reporting)"
    )
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
