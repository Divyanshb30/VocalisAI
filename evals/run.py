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
from vocalis.llm.router import LLMRouter, enable_opik_tracing
from vocalis.rights.engine import assess
from vocalis.simair.call import CallSimulation, RunConfig
from vocalis.simair.scenario import Scenario, load_scenarios
from vocalis.simair.scoring import score

RESULTS = Path("evals/results/runs")
SMOKE = {
    "in_6e_cancel_short_notice__social_engineer",
    "in_6e_cancel_short_notice__prompt_injector",
    "uk_ba_cancel_2_days__cooperative",
    "ae_fz_cancel_refund__social_engineer",
    "uk_vs_delay_long_haul__social_engineer",
    "ae_ek_cancel_weather__prompt_injector",
}

CONFIGS = {
    "vocalis": dict(guards=True, baseline_prompt_secrets=False),
    "baseline": dict(guards=False, baseline_prompt_secrets=True),
    # Same guards as "vocalis", kept separate so a same-talker comparison with "baseline" is possible
    "vocalis_matched": dict(guards=True, baseline_prompt_secrets=False),
    # Same agent, but the far end is heard as phone audio through streaming STT and replies are timed
    # to the first TTS audio byte (see vocalis/telephony/voicelink.py)
    "vocalis_voice": dict(guards=True, baseline_prompt_secrets=False),
}


async def run_one(
    sc: Scenario, config: str, seed: int, router: LLMRouter, with_judge: bool, voice: Any = None
) -> dict:
    cfg = RunConfig(seed=seed, label=config, pace=True, voice=voice, **CONFIGS[config])
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
    }
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    logger.info(
        f"[{config}] {sc.id} s{seed}: success={scored['success']} leaks={scored['leaks']} "
        f"handoff tp/fp/fn={scored['handoff_tp']}/{scored['handoff_fp']}/{scored['handoff_fn']} "
        f"err={scored['error']} ({time.perf_counter() - t0:.0f}s)"
    )
    return scored


def seed_hourly_window(router: LLMRouter) -> None:
    """Start the shared hourly request window from what the provider says is already used this hour."""
    talker = router.s.talker_models[0] if router.s.talker_models else ""
    if not talker.startswith("cerebras/") or not router.s.cerebras_api_key:
        return
    try:
        r = httpx.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {router.s.cerebras_api_key}"},
            json={
                "model": talker.split("/", 1)[1],
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=30,
        )
        used = int(r.headers["x-ratelimit-limit-requests-hour"]) - int(
            r.headers["x-ratelimit-remaining-requests-hour"]
        )
    except Exception as exc:
        logger.warning(f"could not read the hourly request budget: {exc}")
        return
    CallSimulation.hour_log["cerebras"] = [time.monotonic()] * used
    logger.info(f"cerebras: {used} requests already used this hour")


async def main_async(a: argparse.Namespace) -> None:
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
    seed_hourly_window(router)
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
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
