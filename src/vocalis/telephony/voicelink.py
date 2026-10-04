"""VoiceLink: the audio leg of a simulated call (L2 audio loopback).

The far end's words (IVR prompts, the rep) are synthesised with Deepgram Aura-2, pushed
through PhoneLineSim (8 kHz, μ-law, band-pass, noise) and streamed in real time, 20 ms at a
time, to Deepgram's streaming STT, the way a phone call delivers them. The agent hears the
transcript, not the original text. The agent's first sentence goes to streaming TTS and the
arrival of its first audio byte is timed, so each turn gives a wall-clock voice-to-voice
latency:

    end of the rep's speech -> STT endpoint -> LLM + guard -> first agent audio byte

Keypresses travel as in-band DTMF tones over the same simulated line and are decoded with
Goertzel, as a real IVR would.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import re
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import httpx
import numpy as np
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from vocalis.telephony import dtmf
from vocalis.telephony.phoneline import PhoneLineSim, float_to_pcm16, pcm16_to_float

SPEAK_URL = "https://api.deepgram.com/v1/speak"
LISTEN_URL = "wss://api.deepgram.com/v1/listen"
SPEAK_WS_URL = "wss://api.deepgram.com/v1/speak"
CACHE = Path(".cache/voicelink")
FRAME = 160  # 20 ms at 8 kHz
VOICED_RMS = 0.02  # ~-34 dBFS; line noise sits at -45 dBFS


@dataclass
class Heard:
    truth: str
    text: str  # what STT returned
    speech_s: float  # length of the far end's speech
    speech_end: float  # monotonic time the last voiced 20 ms frame was delivered
    final_at: float  # monotonic time the final transcript arrived (after VAD hangover + Finalize)
    timed_out: bool = False

    @property
    def endpoint_s(self) -> float:
        return self.final_at - self.speech_end


class VoiceLink:
    def __init__(
        self,
        api_key: str,
        *,
        stt_model: str = "nova-3",
        hangover_ms: int = 300,
        agent_voice: str = "aura-2-draco-en",
        line: PhoneLineSim | None = None,
    ) -> None:
        self.key = api_key
        self.stt_model = stt_model
        self.hangover_ms = hangover_ms
        self.agent_voice = agent_voice
        self.line = line or PhoneLineSim(seed=7)
        self.http = httpx.AsyncClient(timeout=30, headers={"Authorization": f"Token {api_key}"})
        self.detector = dtmf.DTMFDetector()
        self._tts_ws: ClientConnection | None = None
        self._tts_lock = asyncio.Lock()
        self._background: set[asyncio.Task[None]] = set()
        self.tts_reconnects = 0

    async def aclose(self) -> None:
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)
        if self._tts_ws is not None:
            await self._tts_ws.close()
        await self.http.aclose()

    # ----------------------------------------------------------------- far end
    async def far_end_audio(self, text: str, voice: str) -> np.ndarray:
        """``text`` spoken in ``voice``, as it sounds after the phone line (float32, 8 kHz)."""
        text = spelled_for_tts(text)
        digest = hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()
        path = CACHE / f"{digest}.wav"
        if path.exists():
            wav = path.read_bytes()
        else:
            # A WAV container, not raw PCM: the raw REST response is cut off mid-word at the end.
            r = await self.http.post(
                SPEAK_URL,
                params={"model": voice, "encoding": "linear16", "sample_rate": 16000, "container": "wav"},
                json={"text": text},
            )
            r.raise_for_status()
            wav = r.content
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_bytes(wav)
        with wave.open(io.BytesIO(wav)) as w:
            raw = w.readframes(w.getnframes())
        return self.line.process(pcm16_to_float(raw), 16000)

    async def listen(self, text: str, voice: str, *, max_wait_s: float = 3.0) -> Heard:
        """Play ``text`` down the line in real time and return what streaming STT made of it.

        End of turn is decided locally, as Pipecat does: once ``hangover_ms`` of silence follow the
        last voiced frame (energy VAD), the client sends ``Finalize`` and waits for the final
        transcript. Pauses between the rep's sentences are not treated as turn ends.
        """
        audio = await self.far_end_audio(text, voice)
        async with self._tts_lock:  # a live call keeps the TTS socket open; reopen it now, not mid-reply
            await self._tts()
        n = math.ceil(len(audio) / FRAME)
        audio = np.pad(audio, (0, n * FRAME - len(audio)))
        rms = np.sqrt(np.mean(audio.reshape(n, FRAME) ** 2, axis=1))
        voiced = np.flatnonzero(rms > VOICED_RMS)
        last_voiced = int(voiced[-1]) if len(voiced) else n - 1
        noise = self.line.process(np.zeros(FRAME * 50, dtype=np.float32), 8000)
        tail = max(0, last_voiced + 1 + self.hangover_ms // 20 + 150 - n)  # room for Finalize to return
        pcm = float_to_pcm16(np.concatenate([audio, np.resize(noise, tail * FRAME)]))
        params = {
            "model": self.stt_model,
            "encoding": "linear16",
            "sample_rate": 8000,
            "channels": 1,
            "interim_results": "true",
            "smart_format": "true",
            "punctuate": "true",
        }
        finals: list[str] = []
        final_at: float | None = None
        finalized = asyncio.Event()

        ws = await connect(
            f"{LISTEN_URL}?{urlencode(params)}",
            additional_headers={"Authorization": f"Token {self.key}"},
            max_size=None,
        )

        async def reader() -> None:
            nonlocal final_at
            async for msg in ws:
                d = json.loads(msg)
                if d.get("type") != "Results":
                    continue
                alt = d["channel"]["alternatives"][0]["transcript"]
                if d.get("is_final") and alt:
                    finals.append(alt)
                if d.get("from_finalize"):
                    final_at = final_at or time.monotonic()
                    finalized.set()

        marks: dict[str, float] = {}

        async def sender() -> None:
            t0 = time.monotonic()
            # A frame is delivered once its 20 ms of audio has happened, like a live call.
            for i in range(len(pcm) // (2 * FRAME)):
                await asyncio.sleep(max(0.0, t0 + (i + 1) * 0.02 - time.monotonic()))
                await ws.send(pcm[i * 2 * FRAME : (i + 1) * 2 * FRAME])
                if i == last_voiced:
                    marks["speech_end"] = time.monotonic()
                if i == last_voiced + self.hangover_ms // 20:
                    await ws.send(json.dumps({"type": "Finalize"}))
                    marks["finalize"] = time.monotonic()
            await asyncio.sleep(max_wait_s)

        read_task = asyncio.create_task(reader())
        send_task = asyncio.create_task(sender())
        # Hand the transcript over the moment it is final; the socket closes in the background.
        await asyncio.wait(
            {send_task, asyncio.create_task(finalized.wait())}, return_when=asyncio.FIRST_COMPLETED
        )
        timed_out = not finalized.is_set()
        if timed_out:
            final_at = time.monotonic()
        send_task.cancel()
        closer = asyncio.create_task(self._close_stt(ws, read_task))
        self._background.add(closer)
        closer.add_done_callback(self._background.discard)
        assert final_at is not None
        return Heard(
            truth=text,
            text=" ".join(finals).strip(),
            speech_s=(last_voiced + 1) * FRAME / 8000,
            speech_end=marks.get("speech_end", final_at),
            final_at=final_at,
            timed_out=timed_out,
        )

    @staticmethod
    async def _close_stt(ws: ClientConnection, read_task: asyncio.Task[None]) -> None:
        try:
            await ws.send(json.dumps({"type": "CloseStream"}))
            await asyncio.wait_for(read_task, timeout=3)
        except Exception:
            read_task.cancel()
        finally:
            await ws.close()

    # ----------------------------------------------------------------- agent
    async def warm_up(self) -> None:
        """Open the TTS websocket once; a live call keeps it open for its whole length."""
        await self.first_audio_byte("Hello.")

    async def _tts(self) -> ClientConnection:
        if self._tts_ws is None or self._tts_ws.state is not State.OPEN:
            url = f"{SPEAK_WS_URL}?{urlencode({'model': self.agent_voice, 'encoding': 'linear16', 'sample_rate': 24000})}"
            self._tts_ws = await connect(
                url, additional_headers={"Authorization": f"Token {self.key}"}, max_size=None
            )
        return self._tts_ws

    async def first_audio_byte(self, text: str) -> float:
        """Speak ``text`` in the agent's voice over the TTS websocket; monotonic time of the first audio byte.

        Requested at Aura-2's native 24 kHz and downsampled for the line locally (asking Deepgram for
        8 kHz μ-law costs ~1 s of extra first-byte latency).
        """
        async with self._tts_lock:
            for attempt in range(2):
                try:
                    ws = await self._tts()
                    await ws.send(json.dumps({"type": "Speak", "text": text}))
                    await ws.send(json.dumps({"type": "Flush"}))
                    first: float | None = None
                    while True:
                        msg = await ws.recv()
                        if isinstance(msg, bytes):
                            first = first or time.monotonic()
                        elif json.loads(msg).get("type") == "Flushed":
                            return first or time.monotonic()
                except ConnectionClosed:
                    self._tts_ws = None
                    self.tts_reconnects += 1
                    if attempt:
                        raise
        raise RuntimeError("unreachable")

    def dtmf_over_line(self, digits: str) -> str:
        """Send keypresses as tones down the simulated line; return what the far end decodes."""
        return self.detector.decode(self.line.process(dtmf.encode(digits), 8000))


_SPELLED = re.compile(r"\b[A-Za-z0-9](?: [A-Za-z0-9]\b){3,}")


def spelled_for_tts(text: str) -> str:
    """Read spelled codes one character at a time, as a person would.

    Aura-2 normalises "8 Q H 7" as "8 quetzals H 7" (the currency); commas make it say each character.
    """
    return _SPELLED.sub(lambda m: ", ".join(m.group(0).split()), text)


# --------------------------------------------------------------------- scoring
_CURRENCY = {"£": "pounds", "€": "euros", "₹": "rupees", "$": "dollars"}


_CONTRACTIONS = [
    (r"n't\b", " not"),
    (r"'ve\b", " have"),
    (r"'re\b", " are"),
    (r"'m\b", " am"),
    (r"'ll\b", " will"),
    (r"'d\b", " would"),
    (r"'s\b", ""),
]
_TITLES = {"mr": "mister", "mrs": "missus", "ms": "miss", "dr": "doctor"}
# British -> American spelling, folded the same way on both sides
_SPELLING = [
    (r"isation$", "ization"),
    (r"is(e|ed|es|ing)$", r"iz\1"),
    (r"ll(ed|ing)$", r"l\1"),
    (r"our$", "or"),
]
_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _num_words(n: int) -> list[str]:
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        return [_TENS[n // 10]] + ([_ONES[n % 10]] if n % 10 else [])
    for size, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")):
        if n >= size:
            rest = n % size
            return _num_words(n // size) + [name] + (_num_words(rest) if rest else [])
    return [str(n)]


def normalise_words(text: str) -> list[str]:
    """Whisper-style normalisation for WER, applied identically to reference and hypothesis: case,
    punctuation, currency symbols, contractions, titles, British/American spelling, numbers as words,
    and codes split into letters and digits ("AI504", "R8Q4ZT" and "R 8 Q 4 Z T" all agree)."""
    t = text.lower().replace("’", "'")
    for sym, word in _CURRENCY.items():
        t = re.sub(rf"{re.escape(sym)}\s*([\d,.]+)", rf"\1 {word}", t)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    t = t.replace("%", " percent")
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", t)
    for pat, rep in _CONTRACTIONS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    out: list[str] = []
    for tok in t.split():
        mixed = bool(re.search(r"\d", tok)) and bool(re.search(r"[a-z]", tok))
        parts = list(tok) if mixed else [tok]  # codes are read one character at a time
        for p in parts:
            if p.isdigit():
                out += [_ONES[int(d)] for d in p] if mixed or p.startswith("0") else _num_words(int(p))
                continue
            p = _TITLES.get(p, p)
            for pat, rep in _SPELLING:
                p = re.sub(pat, rep, p)
            out.append(p)
    # "five hundred and twenty" == "five hundred twenty"
    numeric = {*_ONES, *_TENS[2:], "hundred", "thousand", "million", "billion"}
    return [
        w
        for i, w in enumerate(out)
        if not (w == "and" and 0 < i < len(out) - 1 and out[i - 1] in numeric and out[i + 1] in numeric)
    ]


def word_errors(truth: str, heard: str) -> tuple[int, int]:
    """(edit distance in words, words in truth) after normalisation."""
    a, b = normalise_words(truth), normalise_words(heard)
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, wb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb))
        prev = cur
    return prev[-1], len(a)
