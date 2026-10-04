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
    assert normalise_words("A refund of £520 for Mr Shah.") == [
        "a",
        "refund",
        "of",
        "five",
        "hundred",
        "twenty",
        "pounds",
        "for",
        "mister",
        "shah",
    ]
    same = [
        ("Your flight was cancelled.", "your flight was canceled"),
        ("I've processed it.", "I have processed it"),
        ("Your reference number is R 8 Q 4 Z T.", "Your reference number is R8Q4ZT."),
        ("more than 3 hours", "more than three hours"),
        ("five hundred and twenty", "520"),
        ("Press 2 for refunds.", "press two for refunds"),
    ]
    for truth, heard in same:
        assert word_errors(truth, heard)[0] == 0, (truth, heard)
    assert word_errors(
        "For security I need the passport number.", "For security I need the pass number."
    ) == (1, 7)


def test_reference_heard_in_stt_styles() -> None:
    assert reference_in("Your reference number is r j z eight e a.") == "RJZ8EA"
    assert reference_in("Your reference number is. R8Q4ZT.") == "R8Q4ZT"
    assert reference_in("your reference is Romeo Juliet Zulu eight Echo Alpha") == "RJZ8EA"
    assert reference_in("Your reference number is R 8 Q 4 Z T. I have also noted it.") == "R8Q4ZT"
    assert reference_in("Your reference number is eight q h seven eight. S.") == "8QH78S"
    assert reference_in("Your reference number is eight s q n eight. Eight.") == "8SQN88"
    assert reference_in("Your reference number is 8QH78S. A refund is on its way.") == "8QH78S"
    assert reference_in("Your reference number, is h nine seven l k. P.") == "H97LKP"
    # garbled codes are not "recovered" from the surrounding words
    assert reference_in("Your reference number is eight Quetzels h seven eight.") is None


def test_keypresses_survive_the_line() -> None:
    link = VoiceLink("test-key", line=PhoneLineSim(seed=3))
    assert link.dtmf_over_line("2") == "2"
    assert link.dtmf_over_line("1029#") == "1029#"
