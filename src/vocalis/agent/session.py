"""AgentSession: the Pipecat pipeline + Flows state machine for one call.

In text mode (evals) the pipeline has no STT/TTS; the harness injects what the other side
says and reads back what the agent says. The same Flows nodes, tools and guards run in
audio mode with STT/TTS added around them.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from loguru import logger
from pipecat.flows import (
    NO_RESPONSE,
    ContextStrategy,
    ContextStrategyConfig,
    FlowManager,
    FlowsFunctionSchema,
    NodeConfig,
)
from pipecat.frames.frames import EndFrame, LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.llm_service import LLMService
from pipecat.workers.runner import WorkerRunner

from vocalis.agent.briefing import Briefing, nato
from vocalis.agent.processors import GuardEvent, LineSink, OutputGuardProcessor
from vocalis.agent.prompts import CONFIRM_TASK, IVR_TASK, NEGOTIATE_TASK, role_message
from vocalis.core.models import OutcomeType

OUTCOME_VALUES = [o.value for o in OutcomeType]

_REFERENCE = re.compile(
    r"reference(?: number)?(?: is|:)?\s+((?:[A-Za-z0-9][\s-]?){5,8})(?![A-Za-z0-9])", re.I
)


def reference_in(text: str) -> str | None:
    """A reference number as spoken by the rep ("R 8 Q 4 Z T" or "R8Q4ZT"), if there is one."""
    m = _REFERENCE.search(text or "")
    if not m:
        return None
    ref = re.sub(r"[^A-Za-z0-9]", "", m.group(1)).upper()
    return ref if 5 <= len(ref) <= 8 else None


@dataclass
class SessionHooks:
    """How the call environment reacts to the agent's tool calls."""

    on_dtmf: Callable[[str], Awaitable[None]]
    on_handoff: Callable[[str], Awaitable[str]]
    approve_offer: Callable[[OutcomeType, Decimal | None], Awaitable[bool]]


@dataclass
class AgentTurn:
    spoken: list[str]
    latency_s: float | None  # injection -> first guarded sentence ready

    @property
    def text(self) -> str:
        return " ".join(s.strip() for s in self.spoken if s.strip())


@dataclass
class SessionRecord:
    dtmf: list[str] = field(default_factory=list)
    handoff_requests: list[str] = field(default_factory=list)
    offers: list[dict[str, Any]] = field(default_factory=list)
    resolution: dict[str, Any] | None = None
    ended: bool = False
    guard_blocks: list[GuardEvent] = field(default_factory=list)
    vault_blocks: list[list[str]] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)
    reference_corrections: list[dict[str, str]] = field(default_factory=list)


class AgentSession:
    def __init__(
        self,
        briefing: Briefing,
        llm: LLMService,
        hooks: SessionHooks,
        *,
        guards_enabled: bool = True,
        baseline_prompt_secrets: dict[str, str] | None = None,
    ) -> None:
        self.b = briefing
        self.llm = llm
        self.hooks = hooks
        self.record = SessionRecord()
        self._busy = 0
        self._baseline_secrets = baseline_prompt_secrets
        self.guard = briefing.output_guard(enabled=guards_enabled)
        self.guard_proc = OutputGuardProcessor(
            briefing.vault,
            self.guard,
            on_block=self.record.guard_blocks.append,
            on_vault_block=self.record.vault_blocks.append,
        )
        self.sink = LineSink()
        self.context = LLMContext()
        self.aggregators = LLMContextAggregatorPair(self.context)
        self.pipeline = Pipeline(
            [self.aggregators.user(), self.llm, self.guard_proc, self.sink, self.aggregators.assistant()]
        )
        self.worker = PipelineWorker(
            self.pipeline,
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
            idle_timeout_secs=None,
            enable_rtvi=False,
        )
        self.flow = FlowManager(
            worker=self.worker,
            llm=self.llm,
            context_aggregator=self.aggregators,
            global_functions=[self._fn_handoff(), self._fn_end_call()],
        )
        self._runner_task: asyncio.Task[None] | None = None
        self._last_heard = ""

    # ------------------------------------------------------------------ prompts
    def _role(self) -> str:
        role = role_message(self.b)
        if self._baseline_secrets:
            # Naive baseline for the ablation: secrets in the prompt, protected only by instructions.
            listed = ", ".join(f"{k}: {v}" for k, v in self._baseline_secrets.items())
            role += (
                "\nPassenger profile (sensitive, share only if truly required and the passenger "
                f"agreed): {listed}"
            )
        return role

    # ---------------------------------------------------------------- functions
    def _tracked(self, fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        async def wrapper(args: dict[str, Any], flow_manager: FlowManager) -> Any:
            self._busy += 1
            try:
                return await fn(args, flow_manager)
            finally:
                self._busy -= 1

        return wrapper

    def _fn_handoff(self) -> FlowsFunctionSchema:
        async def handler(args: dict[str, Any], _fm: FlowManager) -> Any:
            reason = str(args.get("reason", ""))
            self.record.handoff_requests.append(reason)
            outcome = await self.hooks.on_handoff(reason)
            return {"status": "passenger_completed_step_directly", "note": outcome}

        return FlowsFunctionSchema(
            name="request_handoff",
            description=(
                "Bring the passenger onto the call to handle something only they may do: an OTP, "
                "card or payment details, passport or identity checks, or anything you are not "
                "authorised to share or agree to."
            ),
            properties={"reason": {"type": "string", "description": "What the passenger needs to do"}},
            required=["reason"],
            handler=self._tracked(handler),
        )

    def _fn_end_call(self) -> FlowsFunctionSchema:
        async def handler(args: dict[str, Any], _fm: FlowManager) -> Any:
            self.record.ended = True
            return ({"status": "call_ended"}, NO_RESPONSE)

        return FlowsFunctionSchema(
            name="end_call",
            description="Hang up after saying goodbye, once the call is complete.",
            properties={"summary": {"type": "string"}},
            required=["summary"],
            handler=self._tracked(handler),
        )

    def _fn_press_keys(self) -> FlowsFunctionSchema:
        async def handler(args: dict[str, Any], _fm: FlowManager) -> Any:
            digits = "".join(ch for ch in str(args.get("digits", "")) if ch in "0123456789*#")
            verdict = self.guard.check_dtmf(digits)
            if not verdict.allowed:
                self.record.guard_blocks.append(GuardEvent(digits, verdict.reasons))
                return ({"status": "blocked", "reason": "unapproved number"}, NO_RESPONSE)
            self.record.dtmf.append(digits)
            await self.hooks.on_dtmf(digits)
            return ({"status": "pressed", "digits": digits}, NO_RESPONSE)

        return FlowsFunctionSchema(
            name="press_keys",
            description="Press keys on the phone keypad to answer an automated menu.",
            properties={"digits": {"type": "string", "description": "Keys to press, e.g. '2' or '0'"}},
            required=["digits"],
            handler=self._tracked(handler),
        )

    def _fn_evaluate_offer(self) -> FlowsFunctionSchema:
        async def handler(args: dict[str, Any], _fm: FlowManager) -> Any:
            try:
                outcome = OutcomeType(str(args.get("outcome")))
            except ValueError:
                return {
                    "decision": "clarify",
                    "guidance": "Ask the representative to state the offer clearly.",
                }
            amount = args.get("amount")
            amt = Decimal(str(amount)) if amount not in (None, "") else None
            m = self.b.mandate
            entry: dict[str, Any] = {"outcome": outcome.value, "amount": str(amt) if amt else None}
            if outcome in m.forbidden:
                entry["decision"] = "decline"
                guidance = "Politely decline; the passenger does not accept this. Restate the entitlement and ask for the target outcome."
            elif m.permits(outcome) and (m.min_amount is None or amt is None or amt >= m.min_amount):
                entry["decision"] = "accept"
                guidance = "Accept it, and ask for the reference number."
            else:
                approved = await self.hooks.approve_offer(outcome, amt)
                entry["decision"] = (
                    "accept_after_passenger_approval" if approved else "decline_after_passenger_review"
                )
                guidance = (
                    "The passenger approved this. Accept it and ask for the reference number."
                    if approved
                    else "The passenger declined this. Politely decline and restate the target outcome."
                )
            self.record.offers.append(entry)
            return {"decision": entry["decision"], "guidance": guidance}

        return FlowsFunctionSchema(
            name="evaluate_offer",
            description="Check an offer from the representative against the passenger's mandate before agreeing.",
            properties={
                "outcome": {"type": "string", "enum": OUTCOME_VALUES},
                "amount": {"type": "number", "description": "Amount offered, if any"},
            },
            required=["outcome"],
            handler=self._tracked(handler),
        )

    def _fn_record_resolution(self) -> FlowsFunctionSchema:
        async def handler(args: dict[str, Any], _fm: FlowManager) -> Any:
            ref = "".join(ch for ch in str(args.get("reference_number", "")).upper() if ch.isalnum())
            # Don't trust the LLM to copy exact strings: parse the reference from what the rep said.
            heard = reference_in(self._last_heard)
            if heard and heard != ref:
                self.record.reference_corrections.append({"llm": ref, "parsed": heard})
                ref = heard
            self.record.resolution = {
                "outcome": args.get("outcome"),
                "amount": args.get("amount"),
                "currency": args.get("currency"),
                "reference_number": ref,
            }
            self.record.nodes.append("confirm")
            return ({"status": "recorded"}, self._confirm_node(ref))

        return FlowsFunctionSchema(
            name="record_resolution",
            description="Record the agreed resolution once the representative confirms it and gives a reference number.",
            properties={
                "outcome": {"type": "string", "enum": OUTCOME_VALUES},
                "amount": {"type": "number"},
                "currency": {"type": "string"},
                "reference_number": {
                    "type": "string",
                    "description": "Exactly as the representative said it, letters and digits only",
                },
            },
            required=["outcome", "reference_number"],
            handler=self._tracked(handler),
        )

    # -------------------------------------------------------------------- nodes
    def _goal(self) -> str:
        return "refunds, cancellations and compensation for an existing booking"

    def ivr_node(self) -> NodeConfig:
        return NodeConfig(
            name="ivr",
            role_message=self._role(),
            task_messages=[{"role": "developer", "content": IVR_TASK.format(goal=self._goal())}],
            functions=[self._fn_press_keys()],
            respond_immediately=False,
        )

    def negotiate_node(self) -> NodeConfig:
        return NodeConfig(
            name="negotiate",
            role_message=self._role(),
            task_messages=[{"role": "developer", "content": NEGOTIATE_TASK}],
            functions=[self._fn_evaluate_offer(), self._fn_record_resolution()],
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.RESET),
            respond_immediately=False,
        )

    def _confirm_node(self, ref: str) -> NodeConfig:
        return NodeConfig(
            name="confirm",
            task_messages=[{"role": "developer", "content": CONFIRM_TASK.format(reference_nato=nato(ref))}],
            functions=[],
            respond_immediately=True,
        )

    # ------------------------------------------------------------------ control
    async def start(self) -> None:
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self.worker)
        self._runner_task = asyncio.create_task(runner.run())
        await asyncio.sleep(0.2)
        await self.flow.initialize(self.ivr_node())
        self.record.nodes.append("ivr")

    async def stop(self) -> None:
        try:
            await self.worker.queue_frame(EndFrame())
            if self._runner_task:
                await asyncio.wait_for(self._runner_task, timeout=10)
        except Exception as exc:  # pragma: no cover - shutdown best effort
            logger.debug(f"session stop: {exc}")
            await self.worker.cancel()

    def _busy_now(self) -> bool:
        return self._busy > 0

    async def hear(self, text: str, *, timeout: float = 90.0) -> AgentTurn:
        """The other side said ``text``; return what the agent says back."""
        self._last_heard = text
        self.sink.reset_turn()
        t0 = time.monotonic()
        await self.worker.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": text}], run_llm=True)
        )
        await self.sink.wait_turn(self._busy_now, timeout=timeout)
        latency = self.sink.first_text_at - t0 if self.sink.first_text_at else None
        return AgentTurn(list(self.sink.spoken), latency)

    async def note(self, text: str, role: str = "user") -> None:
        """Add something to the context without running the LLM."""
        await self.worker.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": role, "content": text}], run_llm=False)
        )

    async def enter_conversation(self, disclosure: str) -> str:
        """Human picked up: switch to the negotiation node and speak the scripted disclosure."""
        await self.flow.set_node_from_config(self.negotiate_node())
        self.record.nodes.append("negotiate")
        verdict = self.guard.check(disclosure)
        spoken = verdict.text
        await self.note(spoken, role="assistant")
        return spoken

    @property
    def current_node(self) -> str | None:
        return self.record.nodes[-1] if self.record.nodes else None
