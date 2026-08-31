"""Shared vocabulary for the review engine.

Ordered scales are IntEnum so `>=` works directly against a declared
threshold — the coverage policy compares detectability to a minimum
constantly, and string comparison would silently do the wrong thing.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class _OrderedScale(IntEnum):
    """An ordered scale that also parses from, and renders as, its label."""

    @classmethod
    def _missing_(cls, value: object) -> _OrderedScale | None:
        if isinstance(value, str):
            for member in cls:
                if member.label == value:
                    return member
        return None

    @property
    def label(self) -> str:
        return self.name.lower()

    def __str__(self) -> str:
        return self.label


class Scale(_OrderedScale):
    """Detectability and suitability share one ordered scale (spec §13)."""

    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3


class Authority(_OrderedScale):
    """How much a rubric rule may influence the outcome (Decision 4)."""

    INERT = 0
    ADVISORY = 1
    BINDING = 2


class FindingState(StrEnum):
    OBSERVED = "observed"
    SUSPECTED = "suspected"
    NOT_OBSERVED = "not_observed"
    NOT_ASSESSABLE = "not_assessable"


class Verdict(StrEnum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    REJECT = "REJECT"
    INSUFFICIENT_IMAGES = "INSUFFICIENT_IMAGES"


class Coverage(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INADEQUATE = "INADEQUATE"


class Psa10Candidate(StrEnum):
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"


class ReviewConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class UndetectabilityClass(StrEnum):
    STRUCTURAL = "structural"
    CIRCUMSTANTIAL = "circumstantial"
    METADATA_RESOLVABLE = "metadata_resolvable"


class RuleEvaluability(StrEnum):
    APPLICABLE = "applicable"
    UNEVALUABLE = "unevaluable"


class Provenance(StrEnum):
    SUPPLIED = "supplied"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class Mode(StrEnum):
    OFF = "off"
    SMART = "smart"
    DEEP = "deep"

    @classmethod
    def default(cls) -> Mode:
        return cls.SMART
