"""External input, the resolved core input, and the output record (spec §6, §16).

The boundary that matters here is economic. `CandidateInput` may carry an
asking price, because that is what a listing says. `ResolvedCandidate` — the
type the grading core actually receives — has no price field of any kind, so
non-negotiable rule 10 is structural rather than a discipline every stage
has to keep remembering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

__all__ = ["CandidateInput", "CardReview", "ResolvedCandidate", "ResolvedImage"]


class CandidateInput(BaseModel):
    """What arrives from outside. May carry listing metadata including price."""

    source: str
    title: str = ""
    listing_url: str | None = None
    listing_id: str | None = None
    asking_price: str | None = None
    card_type: str | None = None
    set_name: str | None = None
    #: A caller-supplied stable identity for the physical card. Manual entries
    #: without one get a fresh UUID rather than a title-derived hash.
    candidate_id: str | None = None
    image_paths: list[Path] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    supplied_roles: dict[str, str] = Field(default_factory=dict)


class ResolvedImage(BaseModel):
    image_hash: str
    supplied_role: str | None = None
    source_url: str | None = None
    ordering: int = 0


class ResolvedCandidate(BaseModel):
    """The core's input type. Carries NO price field of any kind."""

    candidate_id: str
    source: str
    title: str = ""
    card_type: str | None = None
    set_name: str | None = None
    images: list[ResolvedImage] = Field(default_factory=list)

    model_config = {"frozen": True}


class CardReview(BaseModel):
    """The complete output record (spec §16).

    Field ownership: `combine` owns verdict, psa10_candidate,
    psa10_rank_score, rankable, estimated_psa_grade, review_confidence and
    reasoning; `coverage` owns coverage, limitations,
    recommended_additional_photos and card_identification_request; the
    remaining blocks are the corresponding stages' stored outputs surfaced
    unchanged.
    """

    review_id: int | None = None
    candidate_id: str
    listing_url: str | None = None
    title: str = ""
    mode: str = ""

    verdict: str
    psa10_candidate: str = ""
    psa10_rank_score: int | None = None
    rankable: bool = False
    estimated_psa_grade: str | None = None
    review_confidence: str = ""

    coverage: str = ""
    coverage_detail: dict[str, Any] = Field(default_factory=dict)
    categories: dict[str, Any] = Field(default_factory=dict)
    image_quality: dict[str, Any] = Field(default_factory=dict)
    roles_and_context: dict[str, Any] = Field(default_factory=dict)

    defects_found: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[dict[str, Any]] = Field(default_factory=list)
    recommended_additional_photos: list[str] = Field(default_factory=list)
    card_identification_request: bool = False

    #: Kept separately recoverable for calibration: measurement and
    #: interpretation must never be collapsed into one another.
    cv_assessment: dict[str, Any] = Field(default_factory=dict)
    vision_assessment: dict[str, Any] | None = None
    reasoning: str = ""
    versions: dict[str, str] = Field(default_factory=dict)
