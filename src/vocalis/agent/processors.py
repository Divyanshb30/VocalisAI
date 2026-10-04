"""Pipeline processors: the sentence-level output guard and the text-mode line sink."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TextFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from vocalis.guards.output_guard import SAFE_FALLBACK, OutputGuard
from vocalis.guards.vault import Vault

_SENTENCE_END = re.compile(r"([.!?])(\s+|$)")


@dataclass
class GuardEvent:
    original: str
    reasons: list[str]
    at: float = field(default_factory=time.time)


class OutputGuardProcessor(FrameProcessor):
    """Buffers LLM tokens into sentences, renders vault placeholders, and guards each sentence.

    Two frames leave per sentence:
    - an ``LLMTextFrame`` (skip_tts) with the placeholder/fallback text, for the LLM context;
    - a ``TextFrame`` (not appended to context) with the rendered text, for TTS / the line.
    So a permitted T1 value is spoken but never written into the model's context.
    """

    def __init__(
        self,
        vault: Vault,
        guard: OutputGuard,
        on_block: Callable[[GuardEvent], None] | None = None,
        on_vault_block: Callable[[list[str]], None] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.vault = vault
        self.guard = guard
        self._buf = ""
        self._on_block = on_block
        self._on_vault_block = on_vault_block
        self.blocks: list[GuardEvent] = []

    async def _emit(self, sentence: str) -> None:
        sentence = sentence.strip()
        if not sentence:
            return
        rendered = self.vault.render(sentence)
        context_text = sentence
        spoken = rendered.text
        if rendered.needs_handoff:
            spoken = context_text = SAFE_FALLBACK
            if self._on_vault_block:
                self._on_vault_block(rendered.blocked)
        verdict = self.guard.check(spoken)
        if not verdict.allowed:
            ev = GuardEvent(original=spoken, reasons=verdict.reasons)
            self.blocks.append(ev)
            if self._on_block:
                self._on_block(ev)
            spoken = context_text = verdict.text

        ctx = LLMTextFrame(text=context_text + " ")
        ctx.skip_tts = True
        await self.push_frame(ctx)
        out = TextFrame(text=spoken)
        out.append_to_context = False
        await self.push_frame(out)

    async def _drain(self, final: bool) -> None:
        while True:
            m = _SENTENCE_END.search(self._buf)
            if not m:
                break
            end = m.end()
            sentence, self._buf = self._buf[:end], self._buf[end:]
            await self._emit(sentence)
        if final and self._buf.strip():
            sentence, self._buf = self._buf, ""
            await self._emit(sentence)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
            self._buf += frame.text
            await self._drain(final=False)
            return
        if isinstance(frame, LLMFullResponseEndFrame):
            await self._drain(final=True)
        await self.push_frame(frame, direction)


class LineSink(FrameProcessor):
    """Text-mode "phone line": collects what the agent says and signals when a turn ends."""

    def __init__(self, quiet_s: float = 0.6, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.quiet_s = quiet_s
        self.spoken: list[str] = []
        self.responses = 0
        self.first_text_at: float | None = None
        self._last_end = 0.0
        self._active = False
        self._changed = asyncio.Event()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.total_requests = 0
        self.llm_ttfb_s: list[float] = []
        self.llm_services: set[str] = (
            set()
        )  # which provider actually answered (failover is invisible otherwise)

    def reset_turn(self) -> None:
        self.spoken = []
        self.responses = 0
        self.first_text_at = None
        self._active = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, MetricsFrame):
            for d in frame.data:
                if isinstance(d, LLMUsageMetricsData):
                    self.prompt_tokens += d.value.prompt_tokens or 0
                    # cached input doesn't count toward provider rate limits (Groq reports it gross)
                    self.cached_tokens += d.value.cache_read_input_tokens or 0
                    self.completion_tokens += d.value.completion_tokens or 0
                elif isinstance(d, TTFBMetricsData) and "LLM" in (d.processor or "") and d.value > 0:
                    self.llm_ttfb_s.append(d.value)
                    self.llm_services.add((d.processor or "").split("#")[0])
        if isinstance(frame, LLMFullResponseStartFrame):
            self._active = True
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._active = False
            self.responses += 1
            self.total_requests += 1
            self._last_end = time.monotonic()
        elif (
            isinstance(frame, TextFrame)
            and not isinstance(frame, LLMTextFrame)
            and frame.append_to_context is False
        ):
            if self.first_text_at is None:
                self.first_text_at = time.monotonic()
            self.spoken.append(frame.text)
        self._changed.set()
        await self.push_frame(frame, direction)

    async def wait_turn(self, busy: Callable[[], bool], timeout: float = 90.0) -> None:
        """Return once the LLM has answered and nothing has happened for ``quiet_s``."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._changed.clear()
            quiet = (
                self.responses > 0
                and not self._active
                and not busy()
                and time.monotonic() - self._last_end >= self.quiet_s
            )
            if quiet:
                return
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=0.1)
            except TimeoutError:
                pass
        raise TimeoutError("agent did not finish its turn")
