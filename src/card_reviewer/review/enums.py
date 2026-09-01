"""Shared vocabulary for the review engine.

Ordered scales are IntEnum so `>=` works directly against a declared
threshold — the coverage policy compares detectability to a minimum
constantly, and string comparison would silently do the wrong thing.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class _OrderedScale(IntEnum):
    """An ordered scale with ONE persisted representation: its label.

    IntEnum gives us `>=` against a declared threshold, which the policies
    rely on constantly. But Pydantic would then dump the member as an int,
    so a cached stage output would carry `2` while every hand-written
    fixture and reason code carries `"moderate"` — two representations of
    one value, which is how a producer and its consumer drift apart.

    The core schema below fixes the representation at the label in both
    directions, so `model_dump(mode="json")` — exactly what StageRunner
    persists — round-trips through SQLite unchanged.
    """

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

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def _validate(value: Any) -> _OrderedScale:
            return value if isinstance(value, cls) else cls(value)

        return core_schema.no_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda member: member.label, return_schema=core_schema.str_schema()
            ),
        )


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
