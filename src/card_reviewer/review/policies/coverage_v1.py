"""EvidenceCoveragePolicy v1 (spec §13).

This is what makes I2 mechanically testable rather than a prose aspiration.
Every threshold below is a declared v1 value, changeable only by a version
bump.

The distinction that carries the most weight: a defect type can be
undetectable because *this photograph* cannot show it (circumstantial — a
better photograph fixes it) or because *no photograph of this card* could
(structural — the card's own design). Demanding evidence no photograph
could ever supply would make PASS unreachable for most white-bordered
cards, which is a false-rejection machine. Demanding evidence a better
photograph would supply is correct and retained in full.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..enums import Coverage, Scale, UndetectabilityClass
from ..roles import ImageRole
from ..taxonomy import CATEGORIES, TAXONOMY_VERSION, class_of, defect_types_for
from ..versions import COVERAGE_POLICY_VERSION

__all__ = [
    "COVERAGE_POLICY_VERSION",
    "MIN_ASSESSED",
    "MIN_FRONT_CATEGORIES_FOR_PARTIAL",
    "REQUIRED_FACES",
    "CoverageResult",
    "Limitation",
    "UnevaluableRule",
    "evaluate_coverage",
]

#: Minimum detectability for a defect type to count as assessed.
MIN_ASSESSED = Scale.MODERATE

#: Both faces are required: SURFACE_TECHNICAL_DEFECT_001 records that a
#: crease or paper loss on the back is grade-limiting.
REQUIRED_FACES = (ImageRole.FRONT, ImageRole.BACK)

#: Coverage is scored PER FACE. A flat count across both faces would make
#: every front-only listing INADEQUATE and drop a large share of real input
#: out of the ranking the score exists to serve.
MIN_FRONT_CATEGORIES_FOR_PARTIAL = 2

PHOTO_REQUESTS: dict[str, str] = {
    "GLARE": "a diffuse-lit photograph of the {face} (avoid direct flash)",
    "BLUR": "a sharper close-up of the {face} {category}",
    "LOW_RESOLUTION": "a higher-resolution close-up of the {face} {category}",
    "OCCLUSION": "the {face} out of its holder, or with the obstruction moved",
    "MISSING_FACE": "a photograph of the {face}",
    "SEVERE_PERSPECTIVE": "a square-on photograph of the {face}",
}


class UnevaluableRule(BaseModel):
    """A rubric rule that could not be applied — not a pixel problem."""

    rule_id: str
    category: str
    reason_code: str


class Limitation(BaseModel):
    face: str
    category: str
    defect_type: str
    reason_code: str
    undetectability_class: UndetectabilityClass


class CoverageResult(BaseModel):
    outcome: Coverage
    rankable: bool
    assessed: dict[str, list[str]] = Field(default_factory=dict)
    limitations: list[Limitation] = Field(default_factory=list)
    recommended_additional_photos: list[str] = Field(default_factory=list)
    card_identification_request: bool = False
    policy_version: str = COVERAGE_POLICY_VERSION
    taxonomy_version: str = TAXONOMY_VERSION


def evaluate_coverage(
    detectability: dict[tuple[ImageRole, str, str], Scale],
    reason_codes: dict[tuple[ImageRole, str, str], str],
    vision_assessability: dict[str, bool],
    faces_present: tuple[ImageRole, ...],
    *,
    unevaluable_rules: list[UnevaluableRule] | None = None,
) -> CoverageResult:
    """`unevaluable_rules` carries rubric gaps that have nothing to do with
    pixels — a product-scoped rule we cannot apply because the card was never
    identified. They are metadata-resolvable limitations and must arrive here
    as themselves, not simulated by lowering some defect type's
    detectability."""
    limitations: list[Limitation] = []
    blocked_categories: set[str] = set()

    # Rubric-level gaps first: these are not photograph defects at all.
    for gap in unevaluable_rules or []:
        limitations.append(
            Limitation(
                face="card", category=gap.category, defect_type="*",
                reason_code=gap.reason_code,
                undetectability_class=class_of(gap.reason_code),
            )
        )
        blocked_categories.add(gap.category)

    assessed: dict[str, list[str]] = {}
    for face in REQUIRED_FACES:
        assessed_here: list[str] = []
        for category in CATEGORIES:
            ok = True
            for defect_type in defect_types_for(category):
                if face not in faces_present:
                    limitations.append(
                        Limitation(
                            face=face.value, category=category,
                            defect_type=defect_type, reason_code="MISSING_FACE",
                            undetectability_class=UndetectabilityClass.CIRCUMSTANTIAL,
                        )
                    )
                    ok = False
                    continue
                if detectability.get((face, category, defect_type), Scale.NONE) >= (
                    MIN_ASSESSED
                ):
                    continue
                code = reason_codes.get((face, category, defect_type),
                                        "LOW_RESOLUTION")
                klass = class_of(code)
                limitations.append(
                    Limitation(
                        face=face.value, category=category,
                        defect_type=defect_type, reason_code=code,
                        undetectability_class=klass,
                    )
                )
                # Structural gaps are reported but do not block: no photograph
                # could ever supply the evidence.
                if klass is not UndetectabilityClass.STRUCTURAL:
                    ok = False
            # Vision may veto a category CV suitability alone allowed.
            if vision_assessability.get(category) is False:
                ok = False
            # So may an unapplied product-scoped rule.
            if category in blocked_categories:
                ok = False
            if ok:
                assessed_here.append(category)
        assessed[face.value] = assessed_here

    front = assessed.get(ImageRole.FRONT.value, [])
    complete = all(
        len(assessed.get(f.value, [])) == len(CATEGORIES) for f in REQUIRED_FACES
    )
    if complete:
        outcome, rankable = Coverage.SUFFICIENT, True
    elif len(front) >= MIN_FRONT_CATEGORIES_FOR_PARTIAL:
        outcome, rankable = Coverage.PARTIAL, True
    else:
        outcome, rankable = Coverage.INADEQUATE, False

    photos, identify = _requests(limitations)
    return CoverageResult(
        outcome=outcome, rankable=rankable, assessed=assessed,
        limitations=limitations, recommended_additional_photos=photos,
        card_identification_request=identify,
    )


def _requests(limitations: list[Limitation]) -> tuple[list[str], bool]:
    """Photo requests come from circumstantial gaps ONLY.

    A structural gap no photograph can close must not generate a request, and
    a metadata-resolvable one asks who the card is rather than for a better
    picture of it.
    """
    photos: list[str] = []
    identify = False
    for lim in limitations:
        if lim.undetectability_class is UndetectabilityClass.METADATA_RESOLVABLE:
            identify = True
            continue
        if lim.undetectability_class is UndetectabilityClass.STRUCTURAL:
            continue
        template = PHOTO_REQUESTS.get(lim.reason_code)
        if template:
            text = template.format(face=lim.face, category=lim.category)
            if text not in photos:
                photos.append(text)
    return photos, identify
