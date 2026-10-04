"""Audio-leg helpers that need no network: spelled codes for TTS, WER normalisation, references heard."""

from vocalis.agent.session import reference_in
from vocalis.telephony.phoneline import PhoneLineSim
from vocalis.telephony.voicelink import VoiceLink, normalise_words, spelled_for_tts, word_errors


def test_spelled_codes_read_one_character_at_a_time() -> None:
    assert (
        spelled_for_tts("Your reference number is R 8 Q 4 Z T.")
        == "Your reference number is R, 8, Q, 4, Z, T."
    )
    # ordinary text and compact codes are left alone
    assert spelled_for_tts("Booking W9RD3L on BA 1446, a B grade.") == "Booking W9RD3L on BA 1446, a B grade."


def test_wer_normalisation() -> None:
    assert normalise_words("A refund of £520, ref R 8 Q 4.") == [
        "a",
        "refund",
        "of",
        "520",
        "pounds",
        "ref",
        "r8q4",
    ]
    assert word_errors("Press 2 for refunds.", "press two for refunds") == (1, 4)
    assert word_errors(
        "For refunds, cancellations and compensation, press 3.",
        "For refunds, cancellations, and compensation, press 3.",
    ) == (0, 7)


def test_reference_heard_in_stt_styles() -> None:
    assert reference_in("Your reference number is r j z eight e a.") == "RJZ8EA"
    assert reference_in("Your reference number is. R8Q4ZT.") == "R8Q4ZT"
    assert reference_in("your reference is Romeo Juliet Zulu eight Echo Alpha") == "RJZ8EA"
    assert reference_in("Your reference number is R 8 Q 4 Z T. I have also noted it.") == "R8Q4ZT"
    assert reference_in("Your reference number is eight q h seven eight. S.") == "8QH78S"
    assert reference_in("Your reference number is eight s q n eight. Eight.") == "8SQN88"
    assert reference_in("Your reference number is 8QH78S. A refund is on its way.") == "8QH78S"
    # garbled codes are not "recovered" from the surrounding words
    assert reference_in("Your reference number is eight Quetzels h seven eight.") is None


def test_keypresses_survive_the_line() -> None:
    link = VoiceLink("test-key", line=PhoneLineSim(seed=3))
    assert link.dtmf_over_line("2") == "2"
    assert link.dtmf_over_line("1029#") == "1029#"
