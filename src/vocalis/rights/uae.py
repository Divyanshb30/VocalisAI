"""UAE: GCAA passenger welfare obligations, the UAE Commercial Transactions Law, and the
Montreal Convention for international carriage.

The UAE has no fixed cash-compensation schedule; the enforceable levers are the duty of
care, the right to refund or rebooking on cancellation, and damages for provable loss.

Sources:
- UAE GCAA consumer protection: https://www.gcaa.gov.ae
- Federal Decree-Law No. 50 of 2022 (Commercial Transactions Law), Art. 357
- Montreal Convention 1999, Art. 19; ICAO revised limits effective 28 Dec 2024
  https://www.icao.int/sites/default/files/secretariat/legal/LEB%20Treaty%20Collection%20Documents/2024_Revised_Limits_of_Liability_Under_the_Montreal_Convention_of_1999_en.pdf
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

GCAA_URL = "https://www.gcaa.gov.ae"
MONTREAL_URL = (
    "https://www.icao.int/sites/default/files/secretariat/legal/LEB%20Treaty%20Collection%20Documents/"
    "2024_Revised_Limits_of_Liability_Under_the_Montreal_Convention_of_1999_en.pdf"
)
MONTREAL_DELAY_LIMIT_SDR = Decimal(6303)


def assess_gcaa(case: Case) -> RegimeAssessment:
    dep = airport(case.first_segment.origin).country
    uae_carrier = carrier_country(case.first_segment.carrier) == "AE"
    applies = dep == "AE" or uae_carrier
    why = (
        "flight departs the UAE"
        if dep == "AE"
        else "operated by a UAE carrier"
        if uae_carrier
        else "neither departs the UAE nor operated by a UAE carrier"
    )
    result = RegimeAssessment(regime=Regime.GCAA, applies=applies, why=why)
    if not applies:
        return result

    def cite(clause: str) -> Citation:
        return Citation(regime=Regime.GCAA, clause=clause, url=GCAA_URL)

    d = case.disruption
    ents: list[Entitlement] = []
    if d.type is DisruptionType.CANCELLATION:
        ents.append(
            Entitlement(
                kind=EntitlementKind.REFUND,
                description="Choice of a full refund of the unused ticket or rebooking at no extra cost",
                citation=cite("Passenger Welfare Programme: cancellations"),
            )
        )
    delay = d.departure_delay_minutes or 0
    if d.type in (DisruptionType.CANCELLATION, DisruptionType.DENIED_BOARDING) or delay >= 120:
        ents.append(
            Entitlement(
                kind=EntitlementKind.CARE,
                description="Meals and refreshments after two hours, and access to calls or internet after three",
                citation=cite("Passenger Welfare Programme: duty of care"),
            )
        )
    if d.type is not DisruptionType.NONE and (delay >= 360 or d.type is DisruptionType.CANCELLATION):
        ents.append(
            Entitlement(
                kind=EntitlementKind.HOTEL,
                description="Hotel accommodation and transfers when the wait is six hours or overnight",
                citation=cite("Passenger Welfare Programme: accommodation"),
            )
        )
    if d.type in (DisruptionType.DELAY, DisruptionType.CANCELLATION):
        ents.append(
            Entitlement(
                kind=EntitlementKind.DAMAGES_CLAIM,
                description="Damages for provable financial loss caused by the delay",
                condition="the carrier is not liable if it took all reasonable measures",
                citation=Citation(
                    regime=Regime.GCAA,
                    clause="Commercial Transactions Law (Decree-Law 50/2022) Art. 357",
                    url=GCAA_URL,
                ),
            )
        )
    result.entitlements = ents
    return result


def assess_montreal(case: Case) -> RegimeAssessment:
    dep = airport(case.first_segment.origin).country
    arr = airport(case.last_segment.destination).country
    international = dep != arr
    result = RegimeAssessment(
        regime=Regime.MONTREAL,
        applies=international and case.disruption.type in (DisruptionType.DELAY, DisruptionType.CANCELLATION),
        why="international carriage" if international else "domestic carriage",
    )
    if result.applies:
        result.entitlements = [
            Entitlement(
                kind=EntitlementKind.DAMAGES_CLAIM,
                description="Damages for loss caused by delay, up to the Convention limit",
                amount=MONTREAL_DELAY_LIMIT_SDR,
                currency="SDR",
                condition="provable loss; not owed if the carrier took all reasonable measures",
                citation=Citation(regime=Regime.MONTREAL, clause="Art. 19 and Art. 22(1)", url=MONTREAL_URL),
            )
        ]
    return result
