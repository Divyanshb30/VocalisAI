"""UAE: GCAA CAR-PWP (Passenger Welfare Program, Issue 02, applicable 1 Jan 2025), the UAE Commercial
Transactions Law (Federal Decree-Law No. 50 of 2022) and the Montreal Convention for international carriage.

The UAE has no fixed cash-compensation schedule. CAR-PWP sets care and re-routing duties for disruptions
at UAE airports; the Commercial Transactions Law covers the fare and delay damages, without prejudice to
the international conventions (Art. 354), so the Montreal Convention governs delay damages on
international flights. Every clause cited here is in data/policy/{gcaa,uae_ctl,montreal}.txt.
"""

from __future__ import annotations

from decimal import Decimal

from vocalis.core.geo import airport
from vocalis.core.models import Case, DisruptionType
from vocalis.rights.models import (
    Citation,
    Entitlement,
    EntitlementKind,
    Regime,
    RegimeAssessment,
)

GCAA_URL = (
    "https://www.gcaa.gov.ae/en/epublication/EPublications/Civil%20Aviation%20Regulations%20(CARs)/"
    "CAR%20III%20-%20GENERAL%20REGULATIONS/CAR-PWP%20-%20PASSENGER%20WELFARE%20PROGRAM%20-%20ISSUE%2002.pdf"
)
CTL_URL = "https://uaelegislation.gov.ae/en/legislations/1610"
MONTREAL_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:22001A0718(01)"
MONTREAL_DELAY_LIMIT_SDR = Decimal(6303)  # Art. 22(1) as revised with effect from 28 Dec 2024
_PWP_SCOPE = "for transit passengers and flights that do not start in your home city (PWP.B.006(b))"


def assess_gcaa(case: Case) -> RegimeAssessment:
    dep = airport(case.first_segment.origin).country
    arr = airport(case.last_segment.destination).country
    # CAR-PWP covers disruptions at UAE airports; disruptions elsewhere follow local rules (GM1 to PWP.A.003)
    applies = dep == "AE"
    why = "the disruption is at a UAE airport" if applies else "the disruption is not at a UAE airport"
    result = RegimeAssessment(regime=Regime.GCAA, applies=applies, why=why)
    if not applies:
        return result

    def pwp(clause: str) -> Citation:
        return Citation(regime=Regime.GCAA, clause=clause, url=GCAA_URL)

    def ctl(clause: str) -> Citation:
        return Citation(regime=Regime.GCAA, clause=f"CTL {clause}", url=CTL_URL)

    d = case.disruption
    ents: list[Entitlement] = []
    if d.type is DisruptionType.CANCELLATION:
        # welfare covers cancellations within 48 hours of departure (PWP.A.005)
        if d.notice_hours is None or d.notice_hours <= 48:
            within = (
                "if the flight was cancelled within 48 hours of departure; " if d.notice_hours is None else ""
            )
            ents.append(
                Entitlement(
                    kind=EntitlementKind.REROUTING,
                    description=(
                        "Carriage on another flight as soon as practicable, or a return flight to your first "
                        "point of departure, or re-routing on another airline subject to availability"
                    ),
                    condition=within + _PWP_SCOPE,
                    citation=pwp("PWP.B.006(a) and PWP.A.005"),
                )
            )
            ents.append(
                Entitlement(
                    kind=EntitlementKind.CARE,
                    description="Care as appropriate: refreshments, meals, hotel and transfers, means of communication",
                    condition=within + _PWP_SCOPE,
                    citation=pwp("PWP.B.006(a)"),
                )
            )
        ents.append(
            Entitlement(
                kind=EntitlementKind.REFUND,
                description=(
                    "Your fare back for a flight that did not operate: the carrier is not entitled to the fare "
                    "for carriage it could not start, even when force majeure prevented it"
                ),
                citation=ctl("Art. 322"),
            )
        )
    elif d.type is DisruptionType.DENIED_BOARDING:
        ents += [
            Entitlement(
                kind=EntitlementKind.REROUTING,
                description="A return flight to your first point of departure, or re-routing to your final destination",
                citation=pwp("PWP.B.002"),
            ),
            Entitlement(
                kind=EntitlementKind.CARE,
                description="Care as appropriate: refreshments, meals, hotel and transfers, means of communication",
                citation=pwp("PWP.B.002"),
            ),
        ]
    elif d.type is DisruptionType.DELAY:
        delay = d.departure_delay_minutes or 0
        if 180 <= delay <= 480:
            ents.append(
                Entitlement(
                    kind=EntitlementKind.CARE,
                    description="Free meals and refreshments for the wait, and a means of communication",
                    citation=pwp("PWP.B.003(b)"),
                )
            )
            ents.append(
                Entitlement(
                    kind=EntitlementKind.HOTEL,
                    description="Hotel accommodation",
                    condition="if you missed your connection and the next one is more than 8 hours away",
                    citation=pwp("PWP.B.003(b)"),
                )
            )
        elif delay > 480:
            ents += [
                Entitlement(
                    kind=EntitlementKind.HOTEL,
                    description="Hotel accommodation with transport to and from the airport",
                    citation=pwp("PWP.B.003(c)"),
                ),
                Entitlement(
                    kind=EntitlementKind.CARE,
                    description="Free meals and refreshments for the wait, and a means of communication",
                    citation=pwp("PWP.B.003(c)"),
                ),
            ]
    if arr == "AE" and d.type in (DisruptionType.DELAY, DisruptionType.CANCELLATION):
        # domestic carriage: the Commercial Transactions Law, not the Montreal Convention, governs delay damages
        ents.append(
            Entitlement(
                kind=EntitlementKind.DAMAGES_CLAIM,
                description="Damages for loss caused by the late arrival",
                condition="the carrier escapes liability only by proving force majeure, your own fault or a third party's act",
                citation=ctl("Art. 357 and Art. 333"),
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
                condition="provable loss; not owed if the carrier took all measures that could reasonably be required",
                citation=Citation(regime=Regime.MONTREAL, clause="Art. 19 and Art. 22(1)", url=MONTREAL_URL),
            )
        ]
    return result
