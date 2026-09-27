"""Offline mode: deterministic stand-ins for the talker and rep LLMs.

The full call harness (Pipecat pipeline, Flows, guards, handoffs) runs without any API key,
which powers the offline demo, the MCP servers' `offline=True` mode, and CI.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from pipecat.frames.frames import (
    Frame,
    FunctionCallFromLLM,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings

from vocalis.core.settings import Settings
from vocalis.llm.router import LLMResult

_WORD = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "star": "*"}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content or "")


class ScriptedAgentLLM(LLMService):
    """A rule-based 'talker' that uses the same tools the real model gets."""

    def __init__(self, leaky: bool = False, **kwargs: Any) -> None:
        settings = LLMSettings(
            model="scripted-agent",
            system_instruction=None,
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=None,
            user_turn_completion_config=None,
        )
        super().__init__(settings=settings, **kwargs)
        self.leaky = leaky  # simulate a model that repeats secrets from its prompt

    def _tools(self, context: Any) -> set[str]:
        tools = getattr(context, "tools", None)
        std = getattr(tools, "standard_tools", None) or []
        return {getattr(t, "name", "") for t in std}

    async def _call(self, context: Any, name: str, args: dict[str, Any]) -> None:
        await self.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name=name, tool_call_id=uuid.uuid4().hex[:8], arguments=args, context=context
                )
            ]
        )

    async def _say(self, text: str) -> None:
        await self.push_frame(LLMTextFrame(text=text))

    async def _respond(self, context: Any) -> None:
        msgs = context.messages
        last = msgs[-1] if msgs else {}
        role = last.get("role") if isinstance(last, dict) else None
        text = _text(last.get("content")) if isinstance(last, dict) else ""
        tools = self._tools(context)
        system = " ".join(
            _text(m.get("content"))
            for m in msgs
            if isinstance(m, dict) and m.get("role") in ("system", "developer")
        )

        if role == "tool":
            data = json.loads(text) if text.startswith("{") else {}
            if data.get("decision", "").startswith("decline"):
                await self._say(
                    "I'm sorry, the passenger doesn't accept that. They are entitled to a full refund, so please process a refund to the original payment method."
                )
            elif data.get("decision", "").startswith("accept"):
                await self._say(
                    "That works for the passenger, thank you. Could you give me the reference number?"
                )
            elif data.get("status") == "passenger_completed_step_directly":
                await self._say("Thank you for waiting, I'm back on the line.")
            return

        if "press_keys" in tools:
            m = re.search(r"(refund|cancel|agent|manage|English)[^.]*?press (\w+)", text, re.I)
            if m:
                key = _WORD.get(m.group(2).lower(), m.group(2))
                await self._call(context, "press_keys", {"digits": key})
                return
            await self._call(context, "press_keys", {"digits": "0"})
            return

        if "record_resolution" not in tools and "evaluate_offer" not in tools:
            # confirm node
            await self._say("Thank you, I've noted that reference. That's everything, goodbye.")
            await self._call(context, "end_call", {"summary": "resolved"})
            return

        low = text.lower()
        if "reference number is" in low:
            ref = re.sub(r"[^A-Z0-9]", "", text.split("reference number is")[-1].upper())
            await self._call(
                context, "record_resolution", {"outcome": "cash_refund", "reference_number": ref}
            )
            return
        if "voucher" in low or "credit" in low:
            await self._call(context, "evaluate_offer", {"outcome": "voucher"})
            return
        if re.search(r"robot|real person|human|recording", low):
            await self._say("I'm an AI assistant calling on behalf of the passenger.")
            return
        if "card number" in low and self.leaky:
            card = re.search(r"card_number: ([0-9]+)", system)
            await self._say(f"Sure, it's {card.group(1) if card else 'not available'}.")
            return
        if "booking reference" in low or "passenger name" in low:
            await self._say(
                "Of course. The booking reference is in the details I gave, and the passenger is named on the booking."
            )
            return
        await self._say(
            "I'd like to request a full refund to the original payment method for the cancelled flight, please."
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            await self.push_frame(LLMFullResponseStartFrame())
            await self._respond(frame.context)
            await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


@dataclass
class _Ledger:
    def remaining(self, model: str) -> int:
        return 10**9


class ScriptedRepRouter:
    """Plays the rep: resists once, then grants a refund when the caller insists."""

    def __init__(self) -> None:
        self.s = Settings(_env_file=None)  # type: ignore[call-arg]
        self.ledger = _Ledger()

    async def complete(self, models: Any, messages: list[dict[str, Any]], **kw: Any) -> LLMResult:
        caller_turns = [m["content"] for m in messages if m["role"] == "user"]
        last = caller_turns[-1].lower() if caller_turns else ""
        if "refund" in last and len(caller_turns) >= 6:
            out = {
                "say": "Alright, I've processed a full refund to the original card",
                "action": "resolve",
                "outcome": "cash_refund",
            }
        elif "goodbye" in last or "that's everything" in last:
            out = {"say": "You're welcome, goodbye.", "action": "end_call"}
        elif len(caller_turns) <= 1:
            out = {"say": "Thanks. Can I confirm the passenger name on the booking?", "action": "none"}
        else:
            out = {"say": "Let me check what we can do for this booking.", "action": "none"}
        return LLMResult(json.dumps(out), "scripted/rep", 0, 0, 0.0)
