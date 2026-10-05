"""India: DGCA CAR Section 3, Series M, Part IV (Rev. 4 dated 25 Jan 2023, effective 15 Feb 2023).

Facilities to be provided to passengers by airlines due to denied boarding, cancellation of flights and
delays in flights. Every clause cited here is in data/policy/dgca.txt, cut from the official PDF.
"""

from __future__ import annotations

from decimal import Decimal

from vocalis.core.geo import airport, carrier_country
from vocalis.core.models import Case, DisruptionType
from vocalis.rights.models import (
    Citation,
    Entitlement,
    EntitlementKind,
    Regime,
    RegimeAssessment,
)

URL = "https://www.dgca.gov.in/digigov-portal/Upload?flag=iframeAttachView&attachId=we1PSlOuQhYdHcwKKrm7ew%3D%3D"
_EXTRAORDINARY = (
    "not owed if the disruption was caused by extraordinary circumstances beyond the airline's control "
    "(para 1.4, 1.5)"
)


def _cite(clause: str) -> Citation:
    return Citation(regime=Regime.DGCA, clause=clause, url=URL)


def _applies(case: Case) -> tuple[bool, str]:
    dep = airport(case.first_segment.origin).country
    arr = airport(case.last_segment.destination).country
    indian_carrier = carrier_country(case.first_segment.carrier) == "IN"
    if dep == "IN" and arr == "IN":
        return True, "domestic flight within India"
    if indian_carrier:
        return True, "operated by an Indian carrier"
    if dep == "IN" or arr == "IN":
        return (
            True,
            "flight to or from India (foreign carriers may apply their home rules instead, para 3.6.1)",
        )
    return False, "not a flight to, from or within India"


def _one_way_fare_plus_fuel(case: Case) -> Decimal | None:
    if case.fare is None or case.fare.base_fare is None:
        return None
    return case.fare.base_fare + (case.fare.fuel_charge or Decimal(0))


def _cancellation_band(block_minutes: int | None) -> Decimal:
    """Para 3.3.2: up to 1 h block time INR 5,000; over 1 h up to 2 h INR 7,500; over 2 h INR 10,000."""
    if block_minutes is None or block_minutes > 120:
        return Decimal(10000)
    if block_minutes > 60:
        return Decimal(7500)
    return Decimal(5000)


def _join(*conds: str | None) -> str | None:
    parts = [c for c in conds if c]
    return "; ".join(parts) or None


def assess(case: Case) -> RegimeAssessment:
    applies, why = _applies(case)
    result = RegimeAssessment(regime=Regime.DGCA, applies=applies, why=why)
    if not applies:
        return result

    d = case.disruption
    seg = case.first_segment
    fare_plus_fuel = _one_way_fare_plus_fuel(case)
    blocked = d.extraordinary_circumstances is True
    cond = None if d.extraordinary_circumstances is False else _EXTRAORDINARY
    domestic = airport(seg.origin).country == "IN" and airport(case.last_segment.destination).country == "IN"
    ents: list[Entitlement] = []

    if d.type is DisruptionType.CANCELLATION:
        ents.append(
            Entitlement(
                kind=EntitlementKind.REFUND,
                description="An alternate flight acceptable to you, or a full refund of the ticket",
                citation=_cite("para 3.3.1"),
            )
        )
        # Compensation only for passengers not informed as para 3.3.1 requires, i.e. told less than
        # 24 hours before departure (Rev. 4 dropped the old two-week/alternate-flight rule).
        short_notice = d.notice_hours is None or d.notice_hours < 24
        if short_notice and not blocked:
            band = _cancellation_band(seg.block_minutes)
            amount = min(band, fare_plus_fuel) if fare_plus_fuel is not None else band
            ents.append(
                Entitlement(
                    kind=EntitlementKind.COMPENSATION,
                    description=(
                        "Compensation on top of a full refund, if you don't take an alternate flight "
                        "(block-time band, or the one-way basic fare plus fuel charge if lower); paid in cash "
                        "or by bank transfer, and in vouchers only with your signed agreement"
                    ),
                    amount=amount,
                    currency="INR",
                    condition=_join(
                        "if you were told less than 24 hours before departure"
                        if d.notice_hours is None
                        else None,
                        cond,
                        "you gave the airline an email or phone number at booking (para 3.3.3)",
                    ),
                    citation=_cite("para 3.3.2 and para 3.7.1"),
                )
            )
            ents.append(
                Entitlement(
                    kind=EntitlementKind.CARE,
                    description="Meals and refreshments while you wait at the airport for the alternate flight",
                    condition="if you had already reported for the original flight",
                    citation=_cite("para 3.3.2 and para 3.8.1"),
                )
            )
    elif d.type is DisruptionType.DENIED_BOARDING:
        alt = d.alternative
        if alt is None or d.passenger_declined_alternative or alt.departs_minutes_after_original > 24 * 60:
            pct, cap = Decimal(4), Decimal(20000)
        elif alt.departs_minutes_after_original > 60:
            pct, cap = Decimal(2), Decimal(10000)
        else:
            pct, cap = Decimal(0), Decimal(0)  # alternate flight within one hour: no compensation
        if pct:
            amount = min(pct * fare_plus_fuel, cap) if fare_plus_fuel is not None else cap
            ents.append(
                Entitlement(
                    kind=EntitlementKind.COMPENSATION,
                    description=(
                        f"Denied-boarding compensation ({int(pct * 100)}% of the one-way basic fare plus fuel "
                        "charge, capped); paid in cash or by bank transfer, and in vouchers only with your "
                        "signed agreement"
                    ),
                    amount=amount,
                    currency="INR",
                    citation=_cite("para 3.2.2 and para 3.7.1"),
                )
            )
        if d.passenger_declined_alternative:
            ents.append(
                Entitlement(
                    kind=EntitlementKind.REFUND,
                    description="Full refund of the ticket, as you did not take the alternate flight",
                    citation=_cite("para 3.2.2"),
                )
            )
    elif d.type is DisruptionType.DELAY:
        block = seg.block_minutes or 0
        meal_threshold = 120 if block <= 150 else 180 if block <= 300 else 240
        delay = d.departure_delay_minutes or 0
        if delay >= meal_threshold and not blocked:
            ents.append(
                Entitlement(
                    kind=EntitlementKind.CARE,
                    description="Meals and refreshments appropriate to the waiting time",
                    condition=_join("if you checked in on time", cond),
                    citation=_cite("para 3.4.1 and para 3.8.1"),
                )
            )
        if domestic and delay > 360:
            ents.append(
                Entitlement(
                    kind=EntitlementKind.REFUND,
                    description="An alternate flight within six hours, or a full refund of the ticket",
                    citation=_cite("para 3.4.2"),
                )
            )
        dep_hour = seg.scheduled_departure.hour
        night = dep_hour >= 20 or dep_hour < 3
        if ((night and delay > 360) or delay > 24 * 60) and not blocked:
            ents.append(
                Entitlement(
                    kind=EntitlementKind.HOTEL,
                    description="Hotel accommodation with transfers",
                    condition=cond,
                    citation=_cite("para 3.4.3 and para 3.8.1"),
                )
            )
    result.entitlements = ents
    return result
