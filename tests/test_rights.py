from datetime import datetime
from decimal import Decimal

import pytest

from vocalis.core.geo import great_circle_km
from vocalis.core.models import (
    AlternativeOffer,
    Case,
    Disruption,
    DisruptionType,
    Fare,
    FlightSegment,
    Passenger,
)
from vocalis.rights.engine import assess
from vocalis.rights.models import EntitlementKind, Regime


def make_case(
    carrier: str,
    origin: str,
    dest: str,
    disruption: Disruption,
    block_h: float = 2.0,
    fare: Fare | None = None,
    dep_hour: int = 10,
) -> Case:
    dep = datetime(2026, 10, 1, dep_hour, 0)
    arr = datetime.fromtimestamp(dep.timestamp() + block_h * 3600)
    return Case(
        passenger=Passenger(first_name="Asha", last_name="Rao"),
        booking_reference="X7K2QB",
        airline=carrier,
        segments=[
            FlightSegment(
                carrier=carrier,
                flight_number="101",
                origin=origin,
                destination=dest,
                scheduled_departure=dep,
                scheduled_arrival=arr,
            )
        ],
        disruption=disruption,
        fare=fare,
    )


def comp(assessment, regime: Regime):
    regs = {r.regime: r for r in assessment.regimes}
    return [e for e in regs[regime].entitlements if e.kind == EntitlementKind.COMPENSATION]


def test_distance_sanity():
    assert 6600 < great_circle_km("DEL", "LHR") < 6800
    assert 1100 < great_circle_km("DEL", "BOM") < 1200


def test_uk261_long_haul_cancellation_short_notice():
    c = make_case(
        "BA", "LHR", "DEL", Disruption(type=DisruptionType.CANCELLATION, notice_hours=20), block_h=9
    )
    a = assess(c)
    [e] = comp(a, Regime.UK261)
    assert e.amount == Decimal(520) and e.currency == "GBP"
    assert not {r.regime for r in a.applicable} & {Regime.EU261}


def test_uk261_arrival_into_uk_on_non_uk_carrier_does_not_apply():
    c = make_case("AI", "DEL", "LHR", Disruption(type=DisruptionType.CANCELLATION, notice_hours=5), block_h=9)
    a = assess(c)
    regs = {r.regime: r for r in a.regimes}
    assert not regs[Regime.UK261].applies
    assert regs[Regime.DGCA].applies  # Indian carrier


def test_uk261_arrival_into_uk_on_uk_carrier_applies():
    c = make_case(
        "BA", "DEL", "LHR", Disruption(type=DisruptionType.DELAY, arrival_delay_minutes=200), block_h=9
    )
    [e] = comp(assess(c), Regime.UK261)
    assert e.amount == Decimal(520)


def test_uk261_notice_over_14_days_no_compensation():
    c = make_case(
        "BA", "LHR", "EDI", Disruption(type=DisruptionType.CANCELLATION, notice_hours=15 * 24), block_h=1.2
    )
    assert comp(assess(c), Regime.UK261) == []


def test_uk261_7_to_13_days_with_close_reroute_exempt():
    d = Disruption(
        type=DisruptionType.CANCELLATION,
        notice_hours=9 * 24,
        alternative=AlternativeOffer(departs_minutes_after_original=-60, arrives_minutes_after_original=180),
    )
    c = make_case("BA", "LHR", "EDI", d, block_h=1.2)
    assert comp(assess(c), Regime.UK261) == []


def test_uk261_delay_under_3h_no_compensation():
    c = make_case("BA", "LHR", "MAN", Disruption(type=DisruptionType.DELAY, arrival_delay_minutes=170))
    assert comp(assess(c), Regime.UK261) == []


def test_extraordinary_circumstances_blocks_compensation_not_refund():
    c = make_case(
        "BA",
        "LHR",
        "DXB",
        Disruption(type=DisruptionType.CANCELLATION, notice_hours=3, extraordinary_circumstances=True),
        block_h=7,
    )
    a = assess(c)
    assert comp(a, Regime.UK261) == []
    kinds = {e.kind for e in a.entitlements}
    assert EntitlementKind.REFUND in kinds


def test_eu261_intra_eu_middle_band():
    c = make_case(
        "LH", "DUB", "FCO", Disruption(type=DisruptionType.CANCELLATION, notice_hours=2), block_h=2.5
    )
    [e] = comp(assess(c), Regime.EU261)
    assert e.amount == Decimal(400) and e.currency == "EUR"


@pytest.mark.parametrize(
    ("block_h", "expected"),
    [(0.9, Decimal(5000)), (1.5, Decimal(7500)), (2.5, Decimal(10000))],
)
def test_dgca_cancellation_bands(block_h, expected):
    c = make_case(
        "6E", "DEL", "BOM", Disruption(type=DisruptionType.CANCELLATION, notice_hours=10), block_h=block_h
    )
    [e] = comp(assess(c), Regime.DGCA)
    assert e.amount == expected and e.currency == "INR"


def test_dgca_cancellation_capped_at_fare_plus_fuel():
    fare = Fare(currency="INR", total=Decimal(4200), base_fare=Decimal(3000), fuel_charge=Decimal(500))
    c = make_case(
        "6E",
        "DEL",
        "BOM",
        Disruption(type=DisruptionType.CANCELLATION, notice_hours=10),
        block_h=2.2,
        fare=fare,
    )
    [e] = comp(assess(c), Regime.DGCA)
    assert e.amount == Decimal(3500)


def test_dgca_rev4_pays_cancellation_compensation_only_under_24h_notice():
    # Rev. 4 para 3.3.1/3.3.2: told between 24 hours and two weeks out -> alternate flight or refund only
    five_days = Disruption(type=DisruptionType.CANCELLATION, notice_hours=5 * 24)
    assert comp(assess(make_case("6E", "DEL", "BOM", five_days)), Regime.DGCA) == []
    [e] = comp(
        assess(make_case("6E", "DEL", "BOM", Disruption(type=DisruptionType.CANCELLATION, notice_hours=20))),
        Regime.DGCA,
    )
    assert e.citation.clause.startswith("para 3.3.2")


def test_dgca_denied_boarding_bands():
    fare = Fare(currency="INR", total=Decimal(6000), base_fare=Decimal(4000), fuel_charge=Decimal(1000))
    within_24 = Disruption(
        type=DisruptionType.DENIED_BOARDING,
        alternative=AlternativeOffer(departs_minutes_after_original=300, arrives_minutes_after_original=300),
    )
    [e] = comp(assess(make_case("AI", "DEL", "BLR", within_24, fare=fare)), Regime.DGCA)
    assert e.amount == Decimal(10000)  # 200% of 5000 = 10000, cap 10000

    beyond = Disruption(type=DisruptionType.DENIED_BOARDING, passenger_declined_alternative=True)
    [e] = comp(assess(make_case("AI", "DEL", "BLR", beyond, fare=fare)), Regime.DGCA)
    assert e.amount == Decimal(20000)


def test_dgca_delay_has_no_cash_but_refund_after_6h():
    c = make_case("6E", "DEL", "BOM", Disruption(type=DisruptionType.DELAY, departure_delay_minutes=400))
    a = assess(c)
    assert comp(a, Regime.DGCA) == []
    kinds = {e.kind for r in a.applicable if r.regime == Regime.DGCA for e in r.entitlements}
    assert {EntitlementKind.REFUND, EntitlementKind.CARE} <= kinds


def test_dgca_six_hour_delay_refund_is_for_domestic_flights_only():
    delay = Disruption(type=DisruptionType.DELAY, departure_delay_minutes=400)
    intl = assess(make_case("6E", "DEL", "DXB", delay, block_h=3.5))
    kinds = {e.kind for r in intl.applicable if r.regime == Regime.DGCA for e in r.entitlements}
    assert EntitlementKind.REFUND not in kinds  # para 3.4.2: "When domestic flight is expected to be delayed"


def test_dgca_no_hotel_for_extraordinary_delay():
    late = Disruption(
        type=DisruptionType.DELAY, departure_delay_minutes=26 * 60, extraordinary_circumstances=True
    )
    a = assess(make_case("6E", "DEL", "BOM", late))
    kinds = {e.kind for r in a.applicable if r.regime == Regime.DGCA for e in r.entitlements}
    assert EntitlementKind.HOTEL not in kinds and EntitlementKind.CARE not in kinds  # para 3.4.4


def _gcaa(a):
    return {e.kind: e for r in a.regimes if r.regime == Regime.GCAA for e in r.entitlements}


def test_uae_cancellation_refund_care_and_damages():
    c = make_case("EK", "DXB", "BOM", Disruption(type=DisruptionType.CANCELLATION, notice_hours=6), block_h=3)
    a = assess(c)
    regs = {r.regime: r for r in a.regimes}
    assert regs[Regime.GCAA].applies
    g = _gcaa(a)
    assert {EntitlementKind.REFUND, EntitlementKind.CARE, EntitlementKind.REROUTING} <= set(g)
    assert g[EntitlementKind.REFUND].citation.clause == "CTL Art. 322"
    # international carriage: delay damages come from the Montreal Convention, not UAE law (CTL Art. 354)
    assert EntitlementKind.DAMAGES_CLAIM not in g
    assert regs[Regime.MONTREAL].applies
    assert regs[Regime.DGCA].applies  # flight to India


def test_uae_welfare_covers_cancellations_within_48_hours():
    early = assess(
        make_case("FZ", "DXB", "BOM", Disruption(type=DisruptionType.CANCELLATION, notice_hours=72))
    )
    assert set(_gcaa(early)) == {EntitlementKind.REFUND}  # PWP.A.005: welfare if cancelled within 48 h


def test_uae_delay_care_starts_at_three_hours_and_hotel_after_eight():
    def kinds(minutes: int) -> set[EntitlementKind]:
        return set(
            _gcaa(
                assess(
                    make_case(
                        "EK",
                        "DXB",
                        "LHR",
                        Disruption(type=DisruptionType.DELAY, departure_delay_minutes=minutes),
                        block_h=7,
                    )
                )
            )
        )

    assert kinds(150) == set()  # 1 to 3 hours: flight information only (PWP.B.003(a))
    assert EntitlementKind.CARE in kinds(200)  # PWP.B.003(b)
    hotel_9h = _gcaa(
        assess(
            make_case(
                "EK",
                "DXB",
                "LHR",
                Disruption(type=DisruptionType.DELAY, departure_delay_minutes=540),
                block_h=7,
            )
        )
    )
    assert hotel_9h[EntitlementKind.HOTEL].condition is None  # PWP.B.003(c): over 8 hours


def test_uae_rules_cover_disruptions_at_uae_airports_only():
    abroad = assess(
        make_case("EK", "BOM", "DXB", Disruption(type=DisruptionType.CANCELLATION, notice_hours=6))
    )
    assert not {r.regime: r for r in abroad.regimes}[Regime.GCAA].applies  # GM1 to PWP.A.003


def test_every_entitlement_has_a_citation():
    c = make_case(
        "6E", "DEL", "DXB", Disruption(type=DisruptionType.CANCELLATION, notice_hours=3), block_h=3.5
    )
    for e in assess(c).entitlements:
        assert e.citation.clause and e.citation.url.startswith("https://")
