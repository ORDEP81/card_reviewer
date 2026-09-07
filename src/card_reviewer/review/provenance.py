"""Where a piece of evidence came from.

This module exists so invariant I3 can be enforced as pure logic. Combine
must be able to decide "was this defect visible in something we did not
enhance?" without opening an image file, so every finding carries typed
references and each reference carries its own origin.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class EvidenceOrigin(StrEnum):
    """ORIGINAL and NORMALIZED both count as unenhanced.

    Rectifying a photograph is a geometric resampling: it moves pixels but
    cannot invent local contrast. CLAHE, sharpening and edge-highlighting
    deliberately amplify it, which is exactly why I3 exists.
    """

    ORIGINAL = "original"
    NORMALIZED = "normalized"
    ENHANCED = "enhanced"


class NormalizedBox(BaseModel):
    """A region in normalized card coordinates, [0,1] on both axes.

    Findings carry boxes rather than points so I1's "overlapping location"
    contradiction test is computable (spec §15).
    """

    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> NormalizedBox:
        if self.x1 <= self.x0:
            raise ValueError(f"x1 must exceed x0 — got {self.x0} .. {self.x1}")
        if self.y1 <= self.y0:
            raise ValueError(f"y1 must exceed y0 — got {self.y0} .. {self.y1}")
        return self

    def overlaps(self, other: NormalizedBox) -> bool:
        return not (
            self.x1 <= other.x0
            or other.x1 <= self.x0
            or self.y1 <= other.y0
            or other.y1 <= self.y0
        )


class EvidenceRef(BaseModel):
    """One artifact a finding rests on. Ids only — never pixel data.

    Keeping this a reference rather than an embedded artifact is what stops
    the evidence manifest being duplicated into every stored result.
    """

    artifact_id: str
    image_hash: str
    origin: EvidenceOrigin
    enhancement: str | None = None
    region: NormalizedBox | None = None
    view: str

    @model_validator(mode="after")
    def _enhancement_matches_origin(self) -> EvidenceRef:
        if self.origin is EvidenceOrigin.ENHANCED and not self.enhancement:
            raise ValueError(
                "enhancement is required when origin is ENHANCED — an enhanced "
                "view whose method is unrecorded cannot be reproduced or audited"
            )
        if self.origin is not EvidenceOrigin.ENHANCED and self.enhancement:
            raise ValueError(
                f"enhancement must be absent when origin is {self.origin.value} "
                "— an unenhanced artifact has no enhancement to declare"
            )
        return self

    @property
    def is_enhanced(self) -> bool:
        return self.origin is EvidenceOrigin.ENHANCED
