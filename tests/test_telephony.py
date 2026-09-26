import numpy as np

from vocalis.telephony.dtmf import DTMFDetector, encode
from vocalis.telephony.phoneline import PhoneLineSim, mulaw_decode, mulaw_encode, resample


def test_dtmf_roundtrip_clean():
    for digits in ["2", "1234567890", "*#", "0913"]:
        assert DTMFDetector().decode(encode(digits)) == digits


def test_dtmf_survives_phone_line():
    line = PhoneLineSim(noise_db=-40, seed=1)
    audio = encode("40172#", sample_rate=16000)
    assert DTMFDetector().decode(line.process(audio, 16000)) == "40172#"


def test_dtmf_ignores_speech_like_noise():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.1, 8000).astype(np.float32)
    assert DTMFDetector().decode(noise) == ""


def test_mulaw_roundtrip_error_small():
    x = np.sin(np.linspace(0, 40 * np.pi, 8000)).astype(np.float32) * 0.5
    err = np.abs(mulaw_decode(mulaw_encode(x)) - x).max()
    assert err < 0.03


def test_resample_lengths():
    x = np.zeros(16000, dtype=np.float32)
    assert len(resample(x, 16000, 8000)) == 8000
    assert len(resample(x, 8000, 24000)) == 48000
