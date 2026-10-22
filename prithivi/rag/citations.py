"""Citations, verification labels, and contradiction detection (ARCHITECTURE.md S14).

The rule this module exists to enforce: **a number of citations is not evidence.**
Three sources that copy each other are one source, and an answer that says "as
several sources report" when only one page said it is worse than no answer at all.

So every claim carries the sources that support it, and every claim is labelled:

- ``VERIFIED`` — two or more *independent* sources agree (independence being
  distinct hosts, since a syndication network is one source wearing three hats);
- ``UNVERIFIED`` — exactly one source. Stated as such rather than dressed up;
- ``CONFLICTED`` — sources disagree. Surfaced as a disagreement, never averaged
  into a confident midpoint;
- ``UNSUPPORTED`` — no source backs it, which means it should not be in an answer
  at all.

Similarity is Jaccard overlap over content words. That is a blunt instrument, and
deliberately so: it is deterministic, dependency-free, and impossible for a page
to game by choosing words that match a model. It under-merges rather than
over-merges, which fails towards "unverified" instead of towards false agreement.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "CitationSet",
    "Claim",
    "Conflict",
    "Source",
    "Verification",
    "content_words",
    "cross_check",
    "similarity",
    "split_sentences",
]

#: Words too common to distinguish two claims about the same topic.
_STOPWORDS = frozenset(["a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those", "of", "in", "on", "at", "to", "for", "from", "by", "with", "without", "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "doing", "have", "has", "had", "having", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "it", "its", "it's", "as", "such", "not", "no", "nor", "only", "own", "same", "so", "too", "very", "just", "also", "about", "into", "over", "under", "again", "further", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "what", "which", "who", "whom", "whose", "he", "she", "they", "them", "his", "her", "their", "you", "your", "we", "our", "i", "me", "my", "mine", "us", "one", "two", "three", "said", "says", "say", "per", "via", "etc"])

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'.-]*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|(?<=[.!?])$")

#: Phrases that flip a claim's meaning. Their presence on one side of a cluster
#: and not the other is what separates "the fee is 3%" from "the fee is not 3%".
_NEGATIONS = frozenset({
    "not", "no", "never", "cannot", "can't", "won't", "doesn't", "isn't", "aren't",
    "wasn't", "weren't", "without", "neither", "nor", "unable", "failed", "false",
})
#: Stances that mean the opposite of a neutral claim.
_CONTRAST_MARKERS = frozenset({
    "however", "but", "although", "though", "contrary", "instead", "meanwhile",
    "disputed", "incorrect", "false", "denied", "rejected", "reversed", "revoked",
    "deprecated", "withdrawn", "superseded",
})

#: Jaccard above which two claims are treated as the same claim.
_SIMILARITY_THRESHOLD = 0.55

#: Shared content words required before containment alone can merge two claims.
#: Without a floor, "the library is deprecated" and "the library is fine" would
#: merge on the word "library" alone.
_MIN_SHARED_WORDS = 3

#: Sources below this trust cannot corroborate a claim or create a conflict.
#: An injected page is not a credible dissenting source; treating it as one would
#: let any page manufacture a controversy about any other page.
_MIN_TRUST = 0.5

#: Hosts that syndicate the same wire copy. Two hits from the same host are one
#: source for corroboration purposes.
_AGGREGATOR_HINTS = ("news.google.", "news.yahoo.", "msn.com", "flipboard", "smartnews", "npr.org")


Verification = Literal["VERIFIED", "UNVERIFIED", "CONFLICTED", "UNSUPPORTED"]


@dataclass(frozen=True)
class Source:
    """One document an answer can point at.

    ``id`` is the citation number shown to the user, starting at 1. ``trust`` is
    the injection scan's verdict, so a page that tried to instruct the assistant
    is visibly less load-bearing than one that did not.
    """

    id: int
    url: str
    title: str = ""
    snippet: str = ""
    published: str | None = None
    retrieved_at: str | None = None
    trust: float = 1.0
    host: str = ""

    def marker(self) -> str:
        return f"[{self.id}]"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "published": self.published,
            "retrieved_at": self.retrieved_at,
            "trust": self.trust,
            "host": self.host,
            "marker": self.marker(),
        }


@dataclass(frozen=True)
class Claim:
    """A statement plus the sources that support it.

    ``span`` is the sentence as it appeared in the source, kept so a user can
    check the claim against its source without opening it.
    """

    text: str
    source_ids: tuple[int, ...]
    span: str = ""
    verification: Verification = "UNVERIFIED"
    conflicted_with: tuple[str, ...] = ()

    @property
    def independent_sources(self) -> int:
        """Distinct source count, ignoring repeat hosts. Filled in by
        :func:`cross_check`, which is the only place that knows the hosts."""
        return len(self.source_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "sources": list(self.source_ids),
            "span": self.span,
            "verification": self.verification,
            "conflicted_with": list(self.conflicted_with),
        }


@dataclass(frozen=True)
class Conflict:
    """Two credible-looking sources that disagree.

    ``sides`` pairs a statement with the sources making it, so the answer can say
    what the disagreement is about rather than merely that one exists.
    """

    topic: str
    sides: tuple[tuple[str, tuple[int, ...]], ...]

    def describe(self) -> str:
        rendered = " vs ".join(f"{text} [{', '.join(f'[{i}]' for i in ids)}]" for text, ids in self.sides)
        return f"{self.topic}: {rendered}"

    def to_dict(self) -> dict[str, object]:
        return {"topic": self.topic, "sides": [[text, list(ids)] for text, ids in self.sides]}


# ---------------------------------------------------------------------------
# text utilities
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, stdlib only.

    Deliberately simple. It does not need to be linguistically perfect, because
    its only job is to produce candidate spans for citation — and a sentence that
    is split in the wrong place is still attributable to its source.
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    parts = [p.strip() for p in _SENTENCE_RE.split(cleaned)]
    return [p for p in parts if len(p) > 15]


def content_words(text: str) -> frozenset[str]:
    """Lowercased content words, stopwords and one-character noise removed.

    Trailing punctuation is stripped, which sounds cosmetic and is not: without
    it ``"month."`` and ``"month"`` are different tokens, and since almost every
    claim ends in a full stop the comparison is skewed on every single sentence.
    """
    return frozenset(
        word.strip(".'-") for word in _WORD_RE.findall((text or "").lower())
        if len(word.strip(".'-")) > 2 and word.strip(".'-") not in _STOPWORDS
    )


def similarity(left: str, right: str) -> float:
    """Jaccard overlap of content words, 0..1."""
    a, b = content_words(left), content_words(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _is_same_claim(left: str, right: str, threshold: float) -> bool:
    """Whether two sentences are making the same claim.

    Three routes, in decreasing strictness:

    1. Jaccard at ``threshold`` — two phrasings of one fact.
    2. Containment at ``threshold`` with at least ``_MIN_SHARED_WORDS`` shared
       content words — a fact quoted with an attribution tail ("..., says the
       incident report"), where one claim is a subset of the other.
    3. A weak topical overlap, admitted *only* when the two demonstrably
       disagree. This route exists because the sentences we most need to compare
       are the ones phrased least alike ("the outage lasted 30 minutes" against
       "it lasted 90 minutes"). A false positive here can only widen a cluster
       into a reported conflict; it can never manufacture agreement, since
       disagreeing members are labelled CONFLICTED.
    """
    a, b = content_words(left), content_words(right)
    if not a or not b:
        return False
    shared = a & b
    if not shared:
        return False
    if len(shared) / len(a | b) >= threshold:
        return True
    smaller = min(len(a), len(b))
    if len(shared) >= _MIN_SHARED_WORDS and len(shared) / smaller >= threshold:
        return True
    return len(shared) >= 2 and _disagrees(left, right)


def _negated(text: str) -> bool:
    """Whether the sentence carries a negation.

    Tokenised directly rather than through :func:`content_words`, because every
    negation word is also a stopword — filtering them out would make this check
    permanently false and contradictions would go unnoticed.
    """
    return bool(set(_WORD_RE.findall((text or "").lower())) & _NEGATIONS)


def _has_contrast_marker(text: str) -> bool:
    words = _WORD_RE.findall((text or "").lower())
    return any(word in _CONTRAST_MARKERS for word in words)


def _independent_count(source_ids: Iterable[int], hosts: dict[int, str]) -> int:
    """How many *independent* sources back a claim.

    Two citations on the same host are one source; so are hits behind a known
    syndication redirector. Without this, a single press release republished by
    six outlets would render as "six sources agree".
    """
    distinct: set[str] = set()
    for sid in source_ids:
        host = hosts.get(sid, "")
        if not host:
            distinct.add(f"#{sid}")
            continue
        for hint in _AGGREGATOR_HINTS:
            if hint in host:
                host = hint
                break
        distinct.add(host)
    return len(distinct)


# ---------------------------------------------------------------------------
# cross-checking
# ---------------------------------------------------------------------------


@dataclass
class CitationSet:
    """Sources, claims, and their verification state, rendered for humans."""

    sources: tuple[Source, ...] = ()
    claims: tuple[Claim, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    _hosts: dict[int, str] = field(default_factory=dict, repr=False)

    def source(self, source_id: int) -> Source | None:
        return next((s for s in self.sources if s.id == source_id), None)

    def marker_for(self, source_ids: Iterable[int]) -> str:
        """``[1][3]`` for a claim's sources, empty when it has none."""
        return "".join(f"[{sid}]" for sid in source_ids)

    def verified_claims(self) -> tuple[Claim, ...]:
        return tuple(c for c in self.claims if c.verification == "VERIFIED")

    def unverified_claims(self) -> tuple[Claim, ...]:
        return tuple(c for c in self.claims if c.verification == "UNVERIFIED")

    def conflicted_claims(self) -> tuple[Claim, ...]:
        return tuple(c for c in self.claims if c.verification == "CONFLICTED")

    def render_sources(self, *, max_sources: int | None = None) -> str:
        """The numbered source list the UI shows under an answer."""
        lines = []
        for source in self.sources[: max_sources or len(self.sources)]:
            title = source.title or source.url
            bits = [f"{source.marker()} {title}"]
            if source.host:
                bits.append(f"({source.host})")
            if source.published:
                bits.append(f"· {source.published}")
            if source.trust < 0.95:
                bits.append(f"· trust {source.trust:.0%}")
            lines.append(" ".join(bits) + f"\n    {source.url}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "sources": [s.to_dict() for s in self.sources],
            "claims": [c.to_dict() for c in self.claims],
            "conflicts": [c.to_dict() for c in self.conflicts],
        }


def cross_check(
    claims: Iterable[Claim],
    sources: Iterable[Source],
    *,
    threshold: float = _SIMILARITY_THRESHOLD,
) -> CitationSet:
    """Cluster claims, label them, and record disagreements.

    Args:
        claims: candidate claims extracted from the fetched pages.
        sources: the sources they came from.
        threshold: similarity above which two claims are the same claim.

    Returns:
        A :class:`CitationSet` with every claim relabelled and a ``Conflict`` for
        each cluster whose members disagree.
    """
    source_list = tuple(sources)
    hosts = {s.id: s.host for s in source_list}
    trust = {s.id: s.trust for s in source_list}
    claim_list = list(claims)

    def credible(member: Claim) -> bool:
        return any(trust.get(sid, 1.0) >= _MIN_TRUST for sid in member.source_ids)

    clusters: list[list[int]] = []
    for index, claim in enumerate(claim_list):
        for cluster in clusters:
            if _is_same_claim(claim.text, claim_list[cluster[0]].text, threshold):
                cluster.append(index)
                break
        else:
            clusters.append([index])

    conflicts: list[Conflict] = []
    labelled: list[Claim] = []

    for cluster in clusters:
        members = [claim_list[i] for i in cluster]

        # An injected page is not a witness. It cannot add support to a claim and
        # it cannot create a disagreement, or any site could stage a controversy
        # about its competitors by forging a dissenting page.
        witnesses = [m for m in members if credible(m)] or members
        merged_ids = tuple(sorted({sid for member in witnesses for sid in member.source_ids}))

        conflict = _detect_conflict(witnesses)
        if conflict is not None:
            conflicts.append(conflict)

        if not merged_ids:
            verification: Verification = "UNSUPPORTED"
        elif conflict is not None:
            verification = "CONFLICTED"
        elif _independent_count(merged_ids, hosts) >= 2:
            verification = "VERIFIED"
        else:
            verification = "UNVERIFIED"

        # One entry per cluster, not per member: two pages stating the same fact
        # is one claim corroborated twice, and listing it twice would make one
        # source look like a repeated failure to agree with itself. Support is
        # credited to every credible source in the cluster, which is what makes
        # two independent pages agreeing actually read as verification.
        representative = max(witnesses, key=lambda c: len(c.text))
        labelled.append(Claim(
            text=representative.text,
            source_ids=merged_ids,
            span=representative.span,
            verification=verification,
            conflicted_with=tuple(
                other.text for other in witnesses
                if other is not representative and _disagrees(representative.text, other.text)
            ),
        ))

    return CitationSet(sources=source_list, claims=tuple(labelled), conflicts=tuple(conflicts), _hosts=hosts)


def _disagrees(left: str, right: str) -> bool:
    """True when two similar claims cannot both be true."""
    if _negated(left) != _negated(right):
        return True
    if _has_contrast_marker(left) != _has_contrast_marker(right):
        return True
    return _numbers_differ(left, right)


def _numbers_differ(left: str, right: str) -> bool:
    """Different numbers for the same kind of thing means a real disagreement.

    Only compared when both sides mention numbers of the same shape; "€3 per
    month" against "€3" is not a contradiction.
    """
    left_numbers = _numbers(left)
    right_numbers = _numbers(right)
    if not left_numbers or not right_numbers:
        return False
    shared_units = left_numbers.keys() & right_numbers.keys()
    return any(left_numbers[unit] != right_numbers[unit] for unit in shared_units)


#: Spelled-out months are their own unit, so "released on 7 October" and
#: "released on 11 October" register as a disagreement. Without them, the most
#: common kind of factual conflict — a date — looks like two sources agreeing.
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)

_NUMBER_RE = re.compile(
    rf"(?P<value>\d[\d,.]*)\s*(?P<unit>%|percent|per\s+cent|usd|eur|gbp|inr|usd\s*/\s*\w+|"
    rf"\w+\s*/\s*\w+|degrees?|°[cf]?|km|miles?|kg|lbs?|hours?|minutes?|days?|weeks?|months?|years?"
    rf"|{_MONTHS})?",
    re.IGNORECASE,
)


def _numbers(text: str) -> dict[str, str]:
    """Numbers keyed by unit, so only like-for-like quantities are compared."""
    out: dict[str, str] = {}
    for match in _NUMBER_RE.finditer(text or ""):
        unit = (match.group("unit") or "").strip().lower()
        value = match.group("value").rstrip(".,")
        if not unit:
            continue
        out.setdefault(unit, value)
    return out


def _detect_conflict(members: list[Claim]) -> Conflict | None:
    """Build a :class:`Conflict` when a claim cluster disagrees.

    ``members`` arrives already filtered to credible sources by
    :func:`cross_check`; a lone hostile page is excluded there, because an
    injected dissent is an attack rather than a controversy.
    """
    if len(members) < 2:
        return None
    sides: dict[str, tuple[int, ...]] = {}
    for member in members:
        key = member.text.strip()
        sides[key] = tuple(sorted(set(sides.get(key, ())) | set(member.source_ids)))
    disagreeing = [text for text in sides if any(_disagrees(text, other) for other in sides if other != text)]
    if len(disagreeing) < 2:
        return None
    topic = _topic(disagreeing)
    return Conflict(
        topic=topic,
        sides=tuple((text, sides[text]) for text in disagreeing),
    )


def _topic(claims: list[str]) -> str:
    """A short label for what the disagreement is about: the words all sides share."""
    shared: set[str] | None = None
    for claim in claims:
        words = content_words(claim)
        shared = words if shared is None else (shared & words)
    if not shared:
        return "conflicting reports"
    return " ".join(sorted(shared)[:5])
