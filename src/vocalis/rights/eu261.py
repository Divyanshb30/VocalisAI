"""UK261 (retained Regulation 261/2004) and EU261.

Sources:
- UK CAA: https://www.caa.co.uk/air-passengers/travel-problems-and-rights/flight-delays-and-cancellations/
- Regulation (EC) 261/2004: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32004R0261
"""

from __future__ import annotations

from decimal import Decimal

from vocalis.core.geo import EU_COUNTRIES, airport, carrier_country, great_circle_km
from vocalis.core.models import Case, DisruptionType
from vocalis.rights.models import (
    Citation,
    Entitlement,
    EntitlementKind,
    Regime,
    RegimeAssessment,
)

UK_URL = "https://www.caa.co.uk/air-passengers/travel-problems-and-rights/flight-delays-and-cancellations/"
EU_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32004R0261"

# Article 7(1) bands: (max_km, amount)
_BANDS = {
    Regime.UK261: ("GBP", [(1500, Decimal(220)), (3500, Decimal(350)), (None, Decimal(520))]),
    Regime.EU261: ("EUR", [(1500, Decimal(250)), (3500, Decimal(400)), (None, Decimal(600))]),
}


def _is_uk(country: str) -> bool:
    return country == "GB"


def _applies(regime: Regime, case: Case) -> tuple[bool, str]:
    dep = airport(case.first_segment.origin).country
    arr = airport(case.last_segment.destination).country
    carrier = carrier_country(case.first_segment.carrier) or ""
    if regime is Regime.UK261:
        if _is_uk(dep):
            return True, "flight departs from the UK (Art. 3(1)(a))"
        if _is_uk(arr) and (carrier == "GB" or carrier in EU_COUNTRIES):
            return True, "flight arrives in the UK on a UK or EU carrier (Art. 3(1)(b))"
        return False, "does not depart the UK, and is not a UK/EU carrier arriving in the UK"
    if dep in EU_COUNTRIES:
        return True, "flight departs from the EU/EEA (Art. 3(1)(a))"
    if arr in EU_COUNTRIES and carrier in EU_COUNTRIES:
        return True, "flight arrives in the EU on an EU carrier (Art. 3(1)(b))"
    return False, "does not depart the EU, and is not an EU carrier arriving in the EU"


def _band(regime: Regime, km: float, intra_eu: bool) -> tuple[Decimal, str]:
    currency, bands = _BANDS[regime]
    # Art. 7(1)(b): intra-Community flights over 1500 km are in the middle band.
    if intra_eu and km > 1500:
        return bands[1][1], currency
    for max_km, amount in bands:
        if max_km is None or km <= max_km:
            return amount, currency
    raise AssertionError("unreachable")


def _cancellation_exempt_by_notice(case: Case) -> bool:
    """Art. 5(1)(c): no compensation if informed early enough with a close alternative."""
    d = case.disruption
    if d.notice_hours is None:
        return False
    days = d.notice_hours / 24
    if days >= 14:
        return True
    alt = d.alternative
    if alt is None:
        return False
    if days >= 7:
        return alt.departs_minutes_after_original >= -120 and alt.arrives_minutes_after_original < 240
    return alt.departs_minutes_after_original >= -60 and alt.arrives_minutes_after_original < 120


def assess(regime: Regime, case: Case) -> RegimeAssessment:
    applies, why = _applies(regime, case)
    result = RegimeAssessment(regime=regime, applies=applies, why=why)
    if not applies:
        return result

    url = UK_URL if regime is Regime.UK261 else EU_URL
    km = great_circle_km(case.first_segment.origin, case.last_segment.destination)
    dep_c = airport(case.first_segment.origin).country
    arr_c = airport(case.last_segment.destination).country
    intra = regime is Regime.EU261 and dep_c in EU_COUNTRIES and arr_c in EU_COUNTRIES
    amount, currency = _band(regime, km, intra)
    d = case.disruption
    extraordinary_cond = (
        None
        if d.extraordinary_circumstances is False
        else "unless caused by extraordinary circumstances such as weather or air traffic control"
    )
    comp_blocked = d.extraordinary_circumstances is True
    ents: list[Entitlement] = []

    def cite(clause: str) -> Citation:
        return Citation(regime=regime, clause=clause, url=url)

    if d.type is DisruptionType.CANCELLATION:
        ents.append(
            Entitlement(
                kind=EntitlementKind.REFUND,
                description="Full refund of the ticket, paid within 7 days, or re-routing at no cost",
                citation=cite("Art. 5(1)(a) and Art. 8"),
            )
        )
        if not comp_blocked and not _cancellation_exempt_by_notice(case):
            ents.append(
                Entitlement(
                    kind=EntitlementKind.COMPENSATION,
                    description=f"Cancellation compensation for a {km:,.0f} km flight",
                    amount=amount,
                    currency=currency,
                    condition=extraordinary_cond,
                    citation=cite("Art. 5(1)(c) and Art. 7"),
                )
            )
    elif d.type is DisruptionType.DELAY:
        arr_delay = d.arrival_delay_minutes or 0
        if arr_delay >= 180 and not comp_blocked:
            ents.append(
                Entitlement(
                    kind=EntitlementKind.COMPENSATION,
                    description=f"Delay compensation for arriving {arr_delay // 60} hours late",
                    amount=amount,
                    currency=currency,
                    condition=extraordinary_cond,
                    citation=cite("Art. 7 (Sturgeon, C-402/07: 3+ hour arrival delay)"),
                )
            )
        if (d.departure_delay_minutes or 0) >= 300:
            ents.append(
                Entitlement(
                    kind=EntitlementKind.REFUND,
                    description="Refund if you choose not to travel after a 5+ hour delay",
                    citation=cite("Art. 6(1)(c)(iii) and Art. 8"),
                )
            )
    elif d.type is DisruptionType.DENIED_BOARDING:
        ents.append(
            Entitlement(
                kind=EntitlementKind.COMPENSATION,
                description="Denied-boarding compensation",
                amount=amount,
                currency=currency,
                citation=cite("Art. 4(3) and Art. 7"),
            )
        )
        ents.append(
            Entitlement(
                kind=EntitlementKind.REFUND,
                description="Refund or re-routing",
                citation=cite("Art. 4(3) and Art. 8"),
            )
        )

    if d.type in (DisruptionType.CANCELLATION, DisruptionType.DELAY, DisruptionType.DENIED_BOARDING):
        care_threshold = 120 if km <= 1500 else 180 if km <= 3500 else 240
        if d.type is not DisruptionType.DELAY or (d.departure_delay_minutes or 0) >= care_threshold:
            ents.append(
                Entitlement(
                    kind=EntitlementKind.CARE,
                    description="Meals and refreshments while waiting, and a hotel if an overnight stay is needed",
                    citation=cite("Art. 9"),
                )
            )
    result.entitlements = ents
    return result
