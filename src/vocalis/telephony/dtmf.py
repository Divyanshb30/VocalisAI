"""In-band DTMF: generate keypad tones as audio and detect them with the Goertzel algorithm.

This is how keypresses travel over a phone line when the media path cannot signal them
out of band (Twilio bidirectional media streams cannot send DTMF). The simulated IVR
decodes the agent's tones exactly the way a real IVR would.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROWS = (697, 770, 852, 941)
COLS = (1209, 1336, 1477, 1633)
KEYPAD = ("123A", "456B", "789C", "*0#D")
KEY_FREQS: dict[str, tuple[int, int]] = {
    key: (ROWS[r], COLS[c]) for r, row in enumerate(KEYPAD) for c, key in enumerate(row)
}


def tone(key: str, sample_rate: int = 8000, duration_ms: int = 120, amplitude: float = 0.35) -> np.ndarray:
    low, high = KEY_FREQS[key.upper()]
    t = np.arange(int(sample_rate * duration_ms / 1000)) / sample_rate
    wave = amplitude * (np.sin(2 * np.pi * low * t) + np.sin(2 * np.pi * high * t)) / 2
    return wave.astype(np.float32)


def encode(digits: str, sample_rate: int = 8000, tone_ms: int = 120, gap_ms: int = 80) -> np.ndarray:
    """A digit string as float32 audio in [-1, 1]."""
    gap = np.zeros(int(sample_rate * gap_ms / 1000), dtype=np.float32)
    parts: list[np.ndarray] = [gap]
    for key in digits:
        parts += [tone(key, sample_rate, tone_ms), gap]
    return np.concatenate(parts)


def _goertzel_power(block: np.ndarray, freq: float, sample_rate: int) -> float:
    n = len(block)
    k = round(n * freq / sample_rate)
    w = 2 * np.pi * k / n
    coeff = 2 * np.cos(w)
    s_prev = s_prev2 = 0.0
    for x in block:
        s = x + coeff * s_prev - s_prev2
        s_prev2, s_prev = s_prev, s
    return float(s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2)


@dataclass
class DTMFDetector:
    """Frame-by-frame detector; a key is emitted once per press (debounced)."""

    sample_rate: int = 8000
    block_ms: int = 40
    min_energy: float = 1e-4
    twist_ratio: float = 6.0

    def detect_block(self, block: np.ndarray) -> str | None:
        energy = float(np.mean(block**2))
        if energy < self.min_energy:
            return None
        row_p = [_goertzel_power(block, f, self.sample_rate) for f in ROWS]
        col_p = [_goertzel_power(block, f, self.sample_rate) for f in COLS]
        r, c = int(np.argmax(row_p)), int(np.argmax(col_p))
        # The winning tone in each group must clearly dominate the others.
        if row_p[r] < self.twist_ratio * (sorted(row_p)[-2] + 1e-9):
            return None
        if col_p[c] < self.twist_ratio * (sorted(col_p)[-2] + 1e-9):
            return None
        return KEYPAD[r][c]

    def decode(self, audio: np.ndarray) -> str:
        n = int(self.sample_rate * self.block_ms / 1000)
        out: list[str] = []
        prev: str | None = None
        for i in range(0, len(audio) - n + 1, n):
            key = self.detect_block(audio[i : i + n])
            if key and key != prev:
                out.append(key)
            prev = key
        return "".join(out)
