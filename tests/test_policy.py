"""Regulation retrieval: corpus chunking, BM25, and RRF fusion (stub embedder, no model download)."""

import hashlib
from pathlib import Path

import numpy as np

from vocalis.rights.policy import PolicyIndex, load_corpus


class _HashEmbedder:
    """Deterministic bag-of-words vectors: enough to exercise the dense and hybrid code paths offline."""

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(256)
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        return v + 1e-6

    def passage_embed(self, texts: list[str]):  # type: ignore[no-untyped-def]
        return (self._vec(t) for t in texts)

    def query_embed(self, texts: list[str]):  # type: ignore[no-untyped-def]
        return (self._vec(t) for t in texts)


def test_corpus_is_article_paragraphs_with_citations() -> None:
    ps = load_corpus()
    ids = [p.id for p in ps]
    assert len(ids) == len(set(ids)) and len(ps) > 60
    art7 = next(p for p in ps if p.id == "UK261 Art. 7(1)")
    assert (
        "£520" in art7.text
        and art7.jurisdiction == "UK"
        and art7.url.startswith("https://www.legislation.gov.uk")
    )
    assert next(p for p in ps if p.id == "EU261 Art. 7(1)").text.count("EUR") == 3


def test_bm25_finds_exact_clause_words() -> None:
    idx = PolicyIndex(embedder=_HashEmbedder())
    top = idx.search("extraordinary circumstances which could not have been avoided", k=2, mode="bm25")
    assert {h.passage.id for h in top} <= {"EU261 Art. 5(3)", "UK261 Art. 5(3)", "UK261 Art. 6(4)"}


def test_hybrid_fuses_and_filters_by_jurisdiction() -> None:
    idx = PolicyIndex(embedder=_HashEmbedder())
    hits = idx.search("meals and refreshments while waiting", k=3, jurisdiction="UK")
    assert hits and all(h.passage.jurisdiction == "UK" for h in hits)
    assert hits[0].passage.id == "UK261 Art. 9(1)"
    assert [h.rank for h in hits] == [1, 2, 3]


def _grid():
    from tests.test_rights import make_case
    from vocalis.core.models import AlternativeOffer, Disruption, DisruptionType

    routes = [("6E", "DEL", "BOM"), ("AI", "DEL", "LHR"), ("BA", "LHR", "JFK"), ("LH", "FRA", "MAD"),
              ("EK", "DXB", "BOM"), ("FZ", "DXB", "DOH"), ("EK", "BOM", "DXB"), ("VS", "LHR", "DEL")]  # fmt: skip
    alt = AlternativeOffer(departs_minutes_after_original=300, arrives_minutes_after_original=300)
    disruptions = (
        [
            Disruption(type=DisruptionType.CANCELLATION, notice_hours=h)
            for h in (None, 6, 30, 72, 10 * 24, 20 * 24)
        ]
        + [
            Disruption(type=DisruptionType.DELAY, departure_delay_minutes=m, arrival_delay_minutes=m)
            for m in (100, 200, 400, 600, 26 * 60)
        ]
        + [
            Disruption(type=DisruptionType.DENIED_BOARDING, alternative=alt),
            Disruption(type=DisruptionType.DENIED_BOARDING, passenger_declined_alternative=True),
        ]
    )
    return [make_case(c, o, d, dis, block_h=2.5) for c, o, d in routes for dis in disruptions]


def test_every_citation_resolves_to_regulation_text() -> None:
    """Each clause an entitlement cites is a passage of the official text in data/policy/."""
    from pathlib import Path

    from vocalis.rights.citations import unresolved
    from vocalis.rights.engine import assess
    from vocalis.simair.scenario import load_scenarios

    passages = load_corpus()
    cases = _grid() + [sc.case for sc in load_scenarios(Path("evals/scenarios"))]
    missing = {
        (e.citation.regime.value, e.citation.clause, tuple(unresolved(e.citation, passages)))
        for case in cases
        for e in assess(case).entitlements
        if unresolved(e.citation, passages)
    }
    assert not missing, sorted(missing)


def test_spoken_regulation_references_are_checked_against_the_corpus() -> None:
    from vocalis.rights.citations import unknown_spoken_refs

    passages = load_corpus()
    assert unknown_spoken_refs("Under DGCA paragraph 3.3.2 you owe compensation.", passages) == []
    assert unknown_spoken_refs("UK261 Article 7 sets the amount.", passages) == []
    assert unknown_spoken_refs("Under DGCA para 3.9.9 that's required.", passages) == ["DGCA para 3.9.9"]
    assert unknown_spoken_refs("Montreal Convention Article 99 says so.", passages) == ["MONTREAL Art. 99"]
    assert unknown_spoken_refs("the GCAA PWP.B.006 rules apply", passages) == []


def test_briefing_quotes_the_official_wording_and_scopes_lookups() -> None:
    from tests.test_rights import make_case
    from vocalis.agent.briefing import Briefing
    from vocalis.agent.prompts import role_message
    from vocalis.core.models import Disruption, DisruptionType
    from vocalis.guards.canaries import make_canaries
    from vocalis.guards.vault import Vault
    from vocalis.rights.engine import assess
    from vocalis.simair.scenario import load_scenarios

    case = make_case("6E", "DEL", "BOM", Disruption(type=DisruptionType.CANCELLATION, notice_hours=10))
    mandate = load_scenarios(Path("evals/scenarios"))[0].mandate
    b = Briefing(case, assess(case), mandate, Vault(make_canaries(0).as_vault_entries()))
    assert "DGCA para 3.3.2: Passengers who have not been informed" in b.regulation_text
    assert b.scope == {"IN", "INTL"}
    assert b.regulation_text in role_message(b)
    # the amount owed for this 2-hour flight may be spoken; a different amount may not
    guard = b.output_guard()
    assert guard.check("Under para 3.3.2 that is INR 7,500 for this flight.").allowed
    assert not guard.check("Under para 3.3.2 that is INR 12,500 for this flight.").allowed
