"""Entry point: assess every regime for a case. See docs/adr/0003-entitlements-in-code.md."""

from __future__ import annotations

from vocalis.core.models import Case
from vocalis.rights import dgca, eu261, uae
from vocalis.rights.models import Regime, RightsAssessment


def assess(case: Case) -> RightsAssessment:
    return RightsAssessment(
        case_id=case.id,
        regimes=[
            eu261.assess(Regime.UK261, case),
            eu261.assess(Regime.EU261, case),
            dgca.assess(case),
            uae.assess_gcaa(case),
            uae.assess_montreal(case),
        ],
    )
