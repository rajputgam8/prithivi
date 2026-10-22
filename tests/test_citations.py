"""Citation cross-checking tests.

The label on a claim is the most consequential string this project produces: it
tells the user whether a fact is corroborated or merely asserted by one page. So
each label is pinned here, including the cases where the right answer is
pessimistic — two pages from the same publisher are one source, and a claim with
no attribution at all is UNSUPPORTED rather than quietly accepted.
"""

from __future__ import annotations

from typing import get_args

import pytest

from prithivi.rag.citations import (
    Claim,
    Source,
    Verification,
    content_words,
    cross_check,
    similarity,
    split_sentences,
)

pytestmark = pytest.mark.integration


def source(sid: int, host: str, **kwargs: object) -> Source:
    return Source(id=sid, url=f"https://{host}/page", title=f"{host} page", host=host, **kwargs)  # type: ignore[arg-type]


def claim(text: str, *ids: int) -> Claim:
    return Claim(text=text, source_ids=ids)


# ---------------------------------------------------------------------------
# text utilities
# ---------------------------------------------------------------------------


def test_sentences_shorter_than_a_clause_are_not_cited() -> None:
    assert split_sentences("No. Yes. The release shipped on 7 October 2025.") == [
        "The release shipped on 7 October 2025."
    ]


def test_sentences_survive_empty_and_whitespace_input() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n\t ") == []


def test_similarity_ignores_filler_words() -> None:
    a = "The service costs 30 usd per month for the pro tier"
    b = "It costs 30 usd per month for the pro tier of the service"
    assert similarity(a, b) > 0.7


def test_similarity_of_unrelated_sentences_is_low() -> None:
    assert similarity("The release shipped on 7 October", "Rainfall in Kerala was heavy") < 0.3


def test_content_words_drop_stopwords_and_short_tokens() -> None:
    assert content_words("The a of it Python 3.14") == frozenset({"python", "3.14"})


# ---------------------------------------------------------------------------
# verification labels
# ---------------------------------------------------------------------------


def test_two_independent_sources_agreeing_verifies_the_claim() -> None:
    """This is the acceptance case: agreement across hosts reads as verified."""
    result = cross_check(
        [
            claim("Python 3.14 was released on 7 October 2025.", 1),
            claim("Python 3.14 was released on 7 October 2025, per the release notes.", 2),
        ],
        [source(1, "docs.python.org"), source(2, "blog.example.com")],
    )

    assert len(result.claims) == 1, "one fact stated twice is one claim"
    assert result.claims[0].verification == "VERIFIED"
    assert result.claims[0].source_ids == (1, 2)
    assert result.verified_claims()
    assert result.conflicts == ()


def test_one_source_is_not_verification() -> None:
    result = cross_check(
        [claim("The service costs 30 usd per month for the pro tier.", 1)],
        [source(1, "vendor.example.com")],
    )
    assert result.claims[0].verification == "UNVERIFIED"
    assert result.unverified_claims()


def test_the_same_host_twice_is_still_one_source() -> None:
    """A press release republished across a site's own paths is not corroboration."""
    result = cross_check(
        [
            claim("Revenue grew by 40 percent last year according to the filings.", 1),
            claim("Revenue grew by 40 percent last year, says the annual report.", 2),
        ],
        [source(1, "news.example.com"), source(2, "news.example.com")],
    )
    assert result.claims[0].verification == "UNVERIFIED"


def test_two_hosts_insn_one_syndication_family_count_once() -> None:
    """msn.com and uk.msn.com carry the same wire copy; they are one source."""
    result = cross_check(
        [
            claim("The chip plant will employ two thousand people in the region.", 1),
            claim("The chip plant will employ two thousand people, per the announcement.", 2),
        ],
        [source(1, "msn.com"), source(2, "uk.msn.com")],
    )
    assert result.claims[0].verification == "UNVERIFIED"


def test_a_claim_with_no_attribution_is_unsupported() -> None:
    result = cross_check([claim("This was probably caused by a bad deploy.",)], [source(1, "a.example")])
    assert result.claims[0].verification == "UNSUPPORTED"
    assert not result.verified_claims()


def test_a_single_page_stating_a_fact_twice_is_not_its_own_corroboration() -> None:
    result = cross_check(
        [
            claim("The battery lasts 30 hours in mixed use according to tests.", 1),
            claim("The battery lasts 30 hours in mixed use, measured by testers.", 1),
        ],
        [source(1, "review.example.com")],
    )
    assert len(result.claims) == 1
    assert result.claims[0].verification == "UNVERIFIED"


def test_different_facts_on_one_page_stay_separate_claims() -> None:
    result = cross_check(
        [
            claim("The default timeout is 30 seconds in the service configuration.", 1),
            claim("The log retention period is 7 days for the free tier.", 1),
        ],
        [source(1, "docs.example.com")],
    )
    assert len(result.claims) == 2


# ---------------------------------------------------------------------------
# conflicts
# ---------------------------------------------------------------------------


def test_two_sources_disagreeing_on_a_number_is_a_conflict() -> None:
    result = cross_check(
        [
            claim("The outage lasted 30 minutes according to the status page.", 1),
            claim("The outage lasted 90 minutes, says the incident report.", 2),
        ],
        [source(1, "status.example.net"), source(2, "incident.example.org")],
    )

    assert result.conflicts, "a real disagreement must be surfaced, not averaged away"
    assert result.conflicted_claims()
    assert "minutes" in result.conflicts[0].topic


def test_one_side_being_negated_is_a_conflict() -> None:
    result = cross_check(
        [
            claim("The service does not charge for local traffic in these regions.", 1),
            claim("The service charges for all traffic including local traffic in these regions.", 2),
        ],
        [source(1, "docs.example.com"), source(2, "news.example.org")],
    )
    assert result.conflicts


def test_a_contradiction_is_never_reported_as_verified() -> None:
    result = cross_check(
        [
            claim("The price is 30 usd per month for individuals.", 1),
            claim("The price is 30 usd per month for individuals and teams.", 2),
            claim("The price is 50 usd per month for individuals.", 3),
        ],
        [source(1, "a.example"), source(2, "b.example"), source(3, "c.example")],
    )
    assert "VERIFIED" not in {c.verification for c in result.claims}
    assert result.conflicts


def test_a_hostile_page_alone_cannot_manufacture_a_controversy() -> None:
    """One injected page against four clean ones is an attack, not a debate."""
    result = cross_check(
        [
            claim("The service costs 30 usd per month for the pro tier of the service.", 1),
            claim("The service costs 30 usd per month for the pro tier of the service.", 2),
            claim("The service costs 300 usd per month for the pro tier of the service.", 9),
        ],
        [
            source(1, "vendor.example.com"),
            source(2, "review.example.net"),
            source(9, "spam.example.org", trust=0.2),
        ],
    )
    assert result.conflicts == ()
    assert result.claims[0].verification == "VERIFIED"


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_the_source_list_shows_numbers_hosts_and_low_trust() -> None:
    result = cross_check(
        [claim("The default timeout is 30 seconds in the service configuration.", 1)],
        [source(1, "docs.example.com", trust=0.4, published="2025-09-01")],
    )
    rendered = result.render_sources()

    assert rendered.startswith("[1] docs.example.com page (docs.example.com)")
    assert "trust 40%" in rendered
    assert "2025-09-01" in rendered
    assert "https://docs.example.com/page" in rendered


def test_a_clean_high_trust_source_is_not_annotated_with_a_score() -> None:
    result = cross_check([], [source(1, "docs.python.org")])
    assert "trust" not in result.render_sources()


def test_markers_and_dicts_agree_with_each_other() -> None:
    result = cross_check([claim("Something happened at the plant last Tuesday.", 1, 2)], [source(1, "a.example"), source(2, "b.example")])
    assert result.claims[0].to_dict()["sources"] == [1, 2]
    assert result.to_dict()["claims"][0]["sources"] == [1, 2]


def test_verification_labels_are_the_documented_set() -> None:
    labels: set[str] = set()
    result = cross_check(
        [
            claim("An unattributed statement about the plant expansion project.",),
            claim("The plant will employ two thousand people in the region.", 1),
        ],
        [source(1, "a.example")],
    )
    labels |= {c.verification for c in result.claims}
    assert labels <= set(get_args(Verification))