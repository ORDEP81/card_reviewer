"""Pydantic models for the video learning pipeline.

These types carry the spec's non-negotiables. Validation lives here so that no
downstream module has to remember them.
"""

from __future__ import annotations

import datetime
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, Field, field_validator

STAGES: tuple[str, ...] = (
    "acquire",
    "transcribe",
    "segment",
    "extract_frames",
    "analyze",
    "validate",
)

# `analyze` (Claude, under skills/learn-video/SKILL.md) and `validate` (the
# mechanical check a human runs via `card-knowledge validate`) are never
# written to a manifest's `stages` — nothing automated advances them. They
# stay in STAGES because they are real stages of the pipeline, but callers
# that render stage status (see `card-knowledge status`) should treat them
# as "not tracked here" rather than as a stalled "pending". See README.md's
# Known limitations section.
MANUAL_STAGES: frozenset[str] = frozenset({"analyze", "validate"})

RULE_ID_RE = re.compile(r"^[A-Z][A-Z0-9_]*[0-9]{3}$")


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class EvidenceType(StrEnum):
    OBJECTIVE = "objective"
    EXPERIENCE_BASED = "experience_based"
    OPINION = "opinion"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RuleStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class Category(StrEnum):
    CENTERING = "centering"
    CORNERS = "corners"
    EDGES = "edges"
    SURFACE = "surface"
    PRINT = "print"
    HANDLING = "handling"
    IMAGE_LIMITATIONS = "image_limitations"
    PROCESS = "process"


class StageState(BaseModel):
    status: StageStatus = StageStatus.PENDING
    at: AwareDatetime | None = None
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class SourceInfo(BaseModel):
    type: Literal["youtube", "skool", "local"]
    url: str | None = None
    title: str
    uploader: str | None = None
    duration_s: float = Field(ge=0)


class FileInfo(BaseModel):
    path: str
    sha256: str
    bytes: int = Field(ge=0)


def _default_stages() -> dict[str, StageState]:
    return {name: StageState() for name in STAGES}


class Manifest(BaseModel):
    video_id: str
    source: SourceInfo
    file: FileInfo | None = None
    stages: dict[str, StageState] = Field(default_factory=_default_stages)
    lesson_id: str | None = None
    rubric_version_at_ingest: str


class Cue(BaseModel):
    """One timestamped utterance from a transcript."""

    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    text: str


class Transcript(BaseModel):
    method: Literal["captions", "mlx-whisper"]
    model: str | None = None
    language: str = "en"
    cues: list[Cue] = Field(default_factory=list)


class Segment(BaseModel):
    """A ranked candidate window worth visual inspection."""

    id: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    score: float
    # categories is list[str] not list[Category]: the segmentation lexicon emits
    # categories ("outcomes", "demonstration") that are outside the rule taxonomy.
    # Category enum is the rule taxonomy only; segment categories are wider.
    categories: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    text: str = ""
    visual_cue: bool = False


class AppliesTo(BaseModel):
    card_types: list[str] = Field(default_factory=list)
    sets: list[str] = Field(default_factory=list)


class RuleSource(BaseModel):
    lesson: str
    video_id: str
    timestamps: list[str] = Field(min_length=1)
    quote: str = ""


class Rule(BaseModel):
    id: str
    category: Category
    statement: str
    evidence_type: EvidenceType
    confidence: Confidence
    applies_to: AppliesTo = Field(default_factory=AppliesTo)
    sources: list[RuleSource] = Field(min_length=1)
    status: RuleStatus = RuleStatus.PENDING
    supersedes: str | None = None
    created: datetime.date
    rubric_version_added: str | None = None
    notes: str | None = None  # rejection reasons and review annotations

    @field_validator("id")
    @classmethod
    def _id_is_slug(cls, value: str) -> str:
        if not RULE_ID_RE.match(value):
            raise ValueError(
                f"rule id {value!r} must be an uppercase slug ending in three "
                "digits, e.g. SURFACE_PRINT_LINE_001"
            )
        return value

    @field_validator("statement")
    @classmethod
    def _statement_is_meaningful(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("statement must not be empty")
        return value.strip()
