"""The detectability taxonomy (spec §13).

A versioned artifact declaring the defect types, the reason codes that
explain why a defect type is or is not detectable, and each code's class.
Its version participates in the `observability` and `cv_measurements`
producer signatures — adding a defect type genuinely changes what a pixel
measurement must compute, so recomputation is correct. Rubric version does
NOT: that changes policy about what measurements mean, not the measurement.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from .enums import UndetectabilityClass
from .versions import TAXONOMY_VERSION

__all__ = [
    "CATEGORIES",
    "DEFECT_TYPES",
    "REASON_CODES",
    "TAXONOMY_VERSION",
    "DefectTypeSpec",
    "Promotion",
    "class_of",
    "defect_types_for",
    "promotion_of",
]


class Promotion(StrEnum):
    """May a measurement alone raise this finding to `observed`?"""

    MEASUREMENT = "measurement"
    INTERPRETIVE = "interpretive"


class DefectTypeSpec(BaseModel):
    category: str
    name: str
    promotion: Promotion

    @property
    def key(self) -> str:
        return f"{self.category}:{self.name}"


def _spec(category: str, name: str, promotion: Promotion) -> tuple[str, DefectTypeSpec]:
    spec = DefectTypeSpec(category=category, name=name, promotion=promotion)
    return spec.key, spec


_M = Promotion.MEASUREMENT
_I = Promotion.INTERPRETIVE

# `measurement` means a defect type a MEASUREMENT can establish outright —
# not one a heuristic can flag. Only the border ratio qualifies today: it is
# computed from the border segmentation with a declared precision.
#
# Corner rounding, corner whitening and edge whitening were classified
# `measurement` in the first draft, but nothing in the CV layer measures
# them. What exists is a contrast heuristic over a crop, which fires on the
# border-to-artwork transition of any normal card. Leaving them
# `measurement` let that heuristic confirm defects on its own and reject
# clean cards — precisely what "OpenCV emits measurements and candidates, not
# verdicts" forbids. They are interpretive until a validated measurement
# exists, at which point this table is where that changes.
DEFECT_TYPES: dict[str, DefectTypeSpec] = dict(
    [
        _spec("centering", "border_ratio", _M),
        _spec("corners", "whitening", _I),
        _spec("corners", "rounding", _I),
        _spec("corners", "fraying", _I),
        _spec("edges", "whitening", _I),
        _spec("edges", "chipping", _I),
        _spec("edges", "roughness", _I),
        _spec("surface", "scratches", _I),
        _spec("surface", "print_lines", _I),
        _spec("surface", "dimples", _I),
        _spec("surface", "stains", _I),
        _spec("surface", "gloss_break", _I),
        # SURFACE_TECHNICAL_DEFECT_001 (active, objective) names a minor
        # crease and paper loss as grade-limiting, and the spec cites it to
        # justify requiring the back. Interpretive in v1: no validated
        # deterministic measurement separates a crease from a scan line or a
        # fold shadow, so CV may raise the candidate and only the vision
        # layer may confirm it.
        _spec("surface", "crease", _I),
        _spec("surface", "paper_loss", _I),
    ]
)

_S = UndetectabilityClass.STRUCTURAL
_C = UndetectabilityClass.CIRCUMSTANTIAL
_MR = UndetectabilityClass.METADATA_RESOLVABLE

REASON_CODES: dict[str, UndetectabilityClass] = {
    # Structural: the card's own printed design. No photograph resolves these.
    "WHITE_BORDER": _S,
    "BORDERLESS_DESIGN": _S,
    # Circumstantial: this photograph. A better one resolves them.
    "GLARE": _C,
    "BLUR": _C,
    "OCCLUSION": _C,
    "LOW_RESOLUTION": _C,
    "SEVERE_PERSPECTIVE": _C,
    # The rectified border could not be told apart from the printed art, so
    # no border ratio was computed. A straighter, better-lit photograph
    # generally can be, which is what makes this circumstantial rather than
    # a property of the card's design.
    "BORDER_NOT_SEPARABLE_FROM_ART": _C,
    # The opposite of GLARE, and it needs the opposite advice.
    "UNDEREXPOSED": _C,
    "MISSING_FACE": _C,
    # Metadata-resolvable: identifying the card resolves it. Not a photo
    # defect, so it must never generate a photo request.
    "UNKNOWN_PRODUCT_CONTEXT": _MR,
}

CATEGORIES: tuple[str, ...] = ("centering", "corners", "edges", "surface")


def defect_types_for(category: str) -> list[str]:
    return [s.name for s in DEFECT_TYPES.values() if s.category == category]


def promotion_of(category: str, name: str) -> Promotion:
    return DEFECT_TYPES[f"{category}:{name}"].promotion


def class_of(reason_code: str) -> UndetectabilityClass:
    try:
        return REASON_CODES[reason_code]
    except KeyError as exc:
        raise KeyError(
            f"unknown reason code {reason_code!r} — every detectability shortfall "
            "must cite a declared code so its class is never guessed"
        ) from exc
