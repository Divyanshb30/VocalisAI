"""India: DGCA CAR Section 3, Series M, Part IV (Revision 4, effective 15 Feb 2023).

Facilities to passengers by airlines due to denied boarding, cancellation and delays.
Source: https://www.dgca.gov.in (Civil Aviation Requirements, Section 3 Series M Part IV)
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

URL = "https://www.dgca.gov.in/digigov-portal/?page=jsp/dgca/InventoryList/dataReg/CAR/"
_EXTRAORDINARY = "unless the disruption was caused by extraordinary circumstances beyond the airline's control"


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
        return True, "flight to or from India (foreign carriers may apply their home rules instead)"
    return False, "not a flight to, from or within India"


def _one_way_fare_plus_fuel(case: Case) -> Decimal | None:
    if case.fare is None or case.fare.base_fare is None:
        return None
    return case.fare.base_fare + (case.fare.fuel_charge or Decimal(0))


def _cancellation_band(block_minutes: int | None) -> Decimal:
    if block_minutes is None or block_minutes > 120:
        return Decimal(10000)
    if block_minutes > 60:
        return Decimal(7500)
    return Decimal(5000)


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
    ents: list[Entitlement] = []

    if d.type is DisruptionType.CANCELLATION:
        ents.append(
            Entitlement(
                kind=EntitlementKind.REFUND,
                description="Alternate flight at no extra cost, or a full refund",
                citation=_cite("para 3.3"),
            )
        )
        notice_h = d.notice_hours
        alt_within_2h = (
            d.alternative is not None and d.alternative.departs_minutes_after_original <= 120
        )
        owed = (
            notice_h is None
            or notice_h < 24
            or (notice_h < 14 * 24 and not alt_within_2h)
        )
        if owed and not blocked:
            band = _cancellation_band(seg.block_minutes)
            amount = min(band, fare_plus_fuel) if fare_plus_fuel is not None else band
            ents.append(
                Entitlement(
                    kind=EntitlementKind.COMPENSATION,
                    description=(
                        "Cancellation compensation (block-time band, or the one-way basic fare "
                        "plus fuel charge if lower)"
                    ),
                    amount=amount,
                    currency="INR",
                    condition=cond,
                    citation=_cite("para 3.3.3"),
                )
            )
    elif d.type is DisruptionType.DENIED_BOARDING:
        alt = d.alternative
        if alt is None or d.passenger_declined_alternative or alt.departs_minutes_after_original > 24 * 60:
            pct, cap = Decimal(4), Decimal(20000)
        elif alt.departs_minutes_after_original > 60:
            pct, cap = Decimal(2), Decimal(10000)
        else:
            pct, cap = Decimal(0), Decimal(0)
        if pct:
            amount = min(pct * fare_plus_fuel, cap) if fare_plus_fuel is not None else cap
            ents.append(
                Entitlement(
                    kind=EntitlementKind.COMPENSATION,
                    description=f"Denied-boarding compensation ({int(pct * 100)}% of the one-way basic fare plus fuel charge, capped)",
                    amount=amount,
                    currency="INR",
                    citation=_cite("para 3.2"),
                )
            )
        if d.passenger_declined_alternative:
            ents.append(
                Entitlement(
                    kind=EntitlementKind.REFUND,
                    description="Full refund of the ticket",
                    citation=_cite("para 3.2"),
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
                    condition=cond,
                    citation=_cite("para 3.4"),
                )
            )
        if delay > 360:
            ents.append(
                Entitlement(
                    kind=EntitlementKind.REFUND,
                    description="An alternate flight within six hours of the original time, or a full refund",
                    citation=_cite("para 3.4"),
                )
            )
        dep_hour = seg.scheduled_departure.hour
        night = dep_hour >= 20 or dep_hour < 3
        if (night and delay > 360) or delay > 24 * 60:
            ents.append(
                Entitlement(
                    kind=EntitlementKind.HOTEL,
                    description="Hotel accommodation with transfers",
                    condition=cond,
                    citation=_cite("para 3.4"),
                )
            )
    result.entitlements = ents
    return result
