"""Which photograph is the front (spec §8).

An `unknown` role is a first-class state, not an error: the image still
contributes to any measurement that does not depend on knowing the face, and
is excluded from those that do. Guessing a face wrong would silently
mis-assign every measurement taken from that photograph, so an ambiguous
layout signature resolves to `unknown` rather than to the likelier answer.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from .enums import Provenance
from .versions import RESOLVER_VERSION

__all__ = [
    "BACK_TEXT_DENSITY",
    "FRONT_TEXT_DENSITY",
    "ImageRole",
    "ResolvedRole",
    "RoleInput",
    "resolve_roles",
]

#: A back is dominated by small high-frequency detail — stat lines, the card
#: number, the copyright block. A front is one large image region. The band
#: between the two thresholds is deliberately wide: everything inside it is
#: reported `unknown` rather than assigned to the nearer edge.
BACK_TEXT_DENSITY = 0.45
FRONT_TEXT_DENSITY = 0.15
INFERENCE_CONFIDENCE = 0.7


class ImageRole(StrEnum):
    FRONT = "front"
    BACK = "back"
    UNKNOWN = "unknown"


class RoleInput(BaseModel):
    """Layout features from the `role_features` stage, plus any supplied role."""

    image_hash: str
    supplied_role: str | None = None
    text_density: float = Field(ge=0.0, le=1.0)
    has_central_image_region: bool = True


class ResolvedRole(BaseModel):
    image_hash: str
    role: ImageRole
    provenance: Provenance
    confidence: float = Field(ge=0.0, le=1.0)
    version: str = RESOLVER_VERSION


def resolve_roles(images: list[RoleInput]) -> dict[str, ResolvedRole]:
    return {image.image_hash: _resolve(image) for image in images}


def _resolve(image: RoleInput) -> ResolvedRole:
    # Supplied outranks inferred outranks unknown (spec §8). An unrecognized
    # supplied value is a caller typo, not an assertion — fall through to
    # inference rather than trusting it.
    if image.supplied_role in {ImageRole.FRONT.value, ImageRole.BACK.value}:
        return ResolvedRole(
            image_hash=image.image_hash,
            role=ImageRole(image.supplied_role),
            provenance=Provenance.SUPPLIED,
            confidence=1.0,
        )

    if image.text_density >= BACK_TEXT_DENSITY and not image.has_central_image_region:
        role = ImageRole.BACK
    elif image.text_density <= FRONT_TEXT_DENSITY and image.has_central_image_region:
        role = ImageRole.FRONT
    else:
        return ResolvedRole(
            image_hash=image.image_hash,
            role=ImageRole.UNKNOWN,
            provenance=Provenance.UNKNOWN,
            confidence=0.0,
        )

    return ResolvedRole(
        image_hash=image.image_hash,
        role=role,
        provenance=Provenance.INFERRED,
        confidence=INFERENCE_CONFIDENCE,
    )
