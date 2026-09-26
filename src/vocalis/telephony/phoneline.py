"""PhoneLineSim: make clean audio sound like it went through the phone network.

Narrowband (8 kHz), G.711 μ-law companding, 300–3400 Hz band-pass, line noise and
occasional packet loss. Every "remote" audio path in simulation passes through this, so
STT, turn detection and DTMF are exercised under realistic conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MU = 255.0


def mulaw_encode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(MU * np.abs(x)) / np.log1p(MU)
    return np.round((y + 1) / 2 * 255).astype(np.uint8)


def mulaw_decode(q: np.ndarray) -> np.ndarray:
    y = q.astype(np.float32) / 255 * 2 - 1
    return (np.sign(y) * ((1 + MU) ** np.abs(y) - 1) / MU).astype(np.float32)


def resample(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampler with a moving-average anti-alias pre-filter."""
    if src_rate == dst_rate:
        return x.astype(np.float32)
    if dst_rate < src_rate:
        taps = max(1, round(src_rate / dst_rate))
        x = np.convolve(x, np.ones(taps) / taps, mode="same")
    n_out = int(len(x) * dst_rate / src_rate)
    src_idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(src_idx, np.arange(len(x)), x).astype(np.float32)


def _one_pole_highpass(x: np.ndarray, rate: int, cutoff: float) -> np.ndarray:
    rc = 1.0 / (2 * np.pi * cutoff)
    alpha = rc / (rc + 1.0 / rate)
    y = np.empty_like(x)
    prev_x = prev_y = 0.0
    for i, xi in enumerate(x):
        prev_y = alpha * (prev_y + xi - prev_x)
        prev_x = xi
        y[i] = prev_y
    return y


def _one_pole_lowpass(x: np.ndarray, rate: int, cutoff: float) -> np.ndarray:
    dt = 1.0 / rate
    alpha = dt / (1.0 / (2 * np.pi * cutoff) + dt)
    y = np.empty_like(x)
    acc = 0.0
    for i, xi in enumerate(x):
        acc += alpha * (xi - acc)
        y[i] = acc
    return y


@dataclass
class PhoneLineSim:
    noise_db: float = -45.0
    packet_loss: float = 0.0  # fraction of 20 ms frames dropped
    bandpass: bool = True
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """float32 audio at any rate in -> float32 phone-quality audio at 8 kHz out."""
        x = resample(audio.astype(np.float32), sample_rate, 8000)
        if self.bandpass:
            x = _one_pole_highpass(x, 8000, 300.0)
            x = _one_pole_lowpass(x, 8000, 3400.0)
        if self.noise_db > -120:
            x = x + self._rng.normal(0, 10 ** (self.noise_db / 20), len(x)).astype(np.float32)
        if self.packet_loss > 0:
            frame = 160  # 20 ms at 8 kHz
            for start in range(0, len(x), frame):
                if self._rng.random() < self.packet_loss:
                    x[start : start + frame] = 0.0
        return mulaw_decode(mulaw_encode(x))


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
