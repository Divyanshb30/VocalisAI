from __future__ import annotations

from vocalis.core.models import Mandate, OutcomeType
from vocalis.guards import input_guard
from vocalis.guards.commitment import accepts, offers_in
from vocalis.guards.output_guard import OutputGuard

V, R, C = OutcomeType.VOUCHER, OutcomeType.CASH_REFUND, OutcomeType.COMPENSATION


def test_offers_are_read_from_the_reps_last_offer() -> None:
    assert offers_in(
        "The best I can do today is a travel voucher for the full amount. Shall I issue that?"
    ) == {V}
    assert offers_in("Would you like a travel credit or a full refund?") == {V, R}
    assert offers_in("We usually offer a voucher first. But I will process a full refund, is that okay?") == {
        R
    }
    assert offers_in("I'm afraid we can't offer a voucher on this fare.") == set()
    assert offers_in("Your flight was cancelled due to weather.") == set()


def test_acceptance_of_the_offer_on_the_table() -> None:
    assert accepts("Yes, please.", V)
    assert accepts("Okay, go ahead and issue the voucher.", V)
    assert accepts("Could you please arrange that?", V)
    assert not accepts("No thank you, the passenger is entitled to a cash refund.", V)
    assert not accepts("Could you please process the full refund instead?", V)
    # "that" refers to what the agent asked for earlier in the same reply
    assert not accepts("Could you please process that?", V, said_before="We'd like the full refund.")
    assert not accepts("Yes, the passenger confirms they are declining the voucher.", V)


def test_guard_blocks_agreeing_to_an_unauthorised_offer() -> None:
    mandate = Mandate(target=R, acceptable=[C], forbidden=[V])
    g = OutputGuard(mandate=mandate)
    g.hear("The best I can do is a travel voucher, valid twelve months. Shall I issue that?")
    blocked = g.check("Sure, that works.")
    assert not blocked.allowed and blocked.reasons == ["commitment:voucher"]
    assert "not agreed to a voucher" in blocked.text
    assert g.check("The passenger is entitled to a cash refund under DGCA rules.").allowed

    g.hear("I can process a full refund to the original card. Is that okay?")
    assert g.check("Yes, please go ahead.").allowed  # the refund is the target

    g = OutputGuard(mandate=Mandate(target=R, acceptable=[], forbidden=[V]))
    g.hear("I could arrange a rebooking on tomorrow's flight instead?")
    assert not g.check("Okay, that's fine.").allowed  # not in the mandate: needs the passenger
    g.approve(OutcomeType.REBOOKING)
    assert g.check("Okay, that's fine.").allowed


def test_guard_without_mandate_or_disabled_does_not_judge_offers() -> None:
    g = OutputGuard()
    g.hear("Shall I issue a voucher?")
    assert g.check("Yes, please.").allowed
    g = OutputGuard(mandate=Mandate(target=R), enabled=False)
    g.hear("Shall I issue a voucher?")
    assert g.check("Yes, please.").allowed


def test_waiving_a_claim_goes_to_the_passenger() -> None:
    flags = input_guard.scan("I can do this as a full and final settlement if you confirm now.")
    assert input_guard.Flag.COMMITMENT in flags
    assert input_guard.requires_handoff(flags, [])
