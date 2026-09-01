"""Post-geometry detectability and suitability (spec §7.3).

Detectability is a physical property of the photograph and the card's own
design — what COULD be seen here, independent of any rubric. Rule IDs are
cited as provenance for why it is worth measuring, never as its contract:
that is why taxonomy version, not rubric version, is in this stage's
producer signature.

The distinction that gives detectability its whole point:

    "No whitening was observed"

is not the same claim as

    "Whitening was highly detectable here, and none was observed."
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from ..enums import Scale, UndetectabilityClass
from ..storage.artifacts import ArtifactStore
from ..taxonomy import CATEGORIES, TAXONOMY_VERSION, class_of, defect_types_for
from ..versions import OBSERVABILITY_VERSION
from .geometry import GeometryResult, load_geometry

__all__ = ["OBSERVABILITY_VERSION", "ObservabilityResult", "analyze"]

Key = tuple[str, str, str]

REGIONS = ("top_left", "top_right", "bottom_left", "bottom_right", "center")

#: Detectability is only meaningful where the defect can occur. Emitting
#: ("center", "corners", "whitening") and then taking the max across regions
#: in assembly is what would make a white-bordered card's corner whitening
#: look HIGH — the centre is never bright — so the structural exemption
#: would never fire end to end even though its unit test passed.
REGIONS_FOR_CATEGORY: dict[str, tuple[str, ...]] = {
    "corners": ("top_left", "top_right", "bottom_left", "bottom_right"),
    "edges": ("top_left", "top_right", "bottom_left", "bottom_right"),
    "surface": REGIONS,
    "centering": ("center",),
}

WHITE_BORDER_LUMA = 200.0
GLARE_LUMA = 245.0
OCCLUSION_LUMA = 12.0
GLARE_FRACTION = 0.15

#: How far a region's clipped fraction must rise above the median of its
#: sibling regions before it counts as a specular highlight rather than the
#: card's own brightness. Measured on a white-bordered card: every corner
#: sits at 0.48 unglared, and a flashed corner reaches 0.79.
GLARE_EXCESS_FRACTION = 0.15


class ObservabilityResult(BaseModel):
    """Cache-safe: JSON-serializable scalars plus artifact ids.

    Tuple keys are stored as "region|category|defect_type" strings because
    JSON object keys must be strings; the tuple view every consumer uses is
    exposed as a property.
    """

    detectability_flat: dict[str, str] = Field(default_factory=dict)
    reason_codes_flat: dict[str, str] = Field(default_factory=dict)
    suitability: dict[str, str] = Field(default_factory=dict)
    glare_mask_artifact_id: str | None = None
    occlusion_mask_artifact_id: str | None = None
    version: str = OBSERVABILITY_VERSION
    taxonomy_version: str = TAXONOMY_VERSION

    @staticmethod
    def _key(key: Key) -> str:
        return "|".join(key)

    @staticmethod
    def _unkey(key: str) -> Key:
        region, category, defect_type = key.split("|")
        return (region, category, defect_type)

    @property
    def detectability(self) -> dict[Key, Scale]:
        return {self._unkey(k): Scale(v) for k, v in self.detectability_flat.items()}

    @property
    def reason_codes(self) -> dict[Key, str]:
        return {self._unkey(k): v for k, v in self.reason_codes_flat.items()}

    def reason_class(self, key: Key) -> UndetectabilityClass | None:
        code = self.reason_codes.get(key)
        return class_of(code) if code else None


def analyze(
    geometry: GeometryResult, store: ArtifactStore, image_hash: str
) -> ObservabilityResult:
    import cv2

    if not geometry.usable:
        # No geometry means nothing could be observed anywhere. NONE, not a
        # default that reads as adequate.
        det = {
            (r, c, d): Scale.NONE
            for c in CATEGORIES
            for r in REGIONS_FOR_CATEGORY[c]
            for d in defect_types_for(c)
        }
        return _build(det, {k: "SEVERE_PERSPECTIVE" for k in det}, None, None)

    artifacts = load_geometry(geometry, store)
    normalized = artifacts.normalized
    gray = normalized.mean(axis=2)

    # A white border and a specular highlight are indistinguishable by
    # luminance alone — both are simply bright. What separates them is
    # whether the brightness belongs to the CARD or to the PHOTOGRAPH: a
    # white border is bright everywhere around the card by design, while
    # glare is a localized highlight. Getting this backwards would classify
    # every white-bordered card's corners as circumstantial and ask the owner
    # for a better photograph of something no photograph can change.
    border_is_white = _border_is_white(gray, artifacts.border_mask)

    det: dict[Key, Scale] = {}
    reasons: dict[Key, str] = {}
    for category in CATEGORIES:
        regions = REGIONS_FOR_CATEGORY[category]
        # Glare is what stands out from THIS card, not what crosses a fixed
        # line. On a white-bordered card every corner clips against an
        # absolute threshold together, so an absolute test either flags all
        # four (an impossible photo request) or — as it did — none of them,
        # including a corner the flash genuinely blew out. Measured on a
        # white card: 0.48 clipped at every corner unglared, 0.79 at the
        # glared one. The excess is the signal.
        fractions = {
            region: float((_patch(gray, region) >= GLARE_LUMA).mean())
            for region in regions
        }
        baseline = float(np.median(list(fractions.values())))
        for region in regions:
            patch = _patch(gray, region)
            clipped = fractions[region] > GLARE_FRACTION
            stands_out = fractions[region] > baseline + GLARE_EXCESS_FRACTION
            bright = float(patch.mean()) >= WHITE_BORDER_LUMA
            for defect_type in defect_types_for(category):
                key = (region, category, defect_type)
                if defect_type == "whitening" and (bright or clipped) and (
                    border_is_white
                ):
                    # A white corner cannot show whitening. Structural: no
                    # photograph of THIS card could ever show it.
                    det[key], reasons[key] = Scale.LOW, "WHITE_BORDER"
                elif stands_out or (clipped and not border_is_white):
                    # `stands_out` catches the highlight a white border used
                    # to hide; the absolute arm still catches a dark card
                    # blown out so evenly that nothing stands out from it.
                    det[key], reasons[key] = Scale.LOW, "GLARE"
                elif category == "centering" and not geometry.has_reliable_border:
                    det[key], reasons[key] = Scale.LOW, "BORDERLESS_DESIGN"
                else:
                    det[key] = Scale.HIGH

    # Masks are pixel data, so they go to the store and the output carries ids.
    glare = ((gray >= GLARE_LUMA).astype(np.uint8) * 255)
    occlusion = ((gray <= OCCLUSION_LUMA).astype(np.uint8) * 255)
    return _build(
        det, reasons,
        store.put_derived(image_hash, "masks", "glare.png",
                          cv2.imencode(".png", glare)[1].tobytes()),
        store.put_derived(image_hash, "masks", "occlusion.png",
                          cv2.imencode(".png", occlusion)[1].tobytes()),
    )


def _border_is_white(gray: np.ndarray, border_mask: np.ndarray | None) -> bool:
    """Is the card's border white by design, rather than glared in one spot?

    The median over the whole band answers that: a white border is bright
    all the way round, while a highlight on one corner leaves the median
    where it was.
    """
    if border_mask is None:
        return False
    values = gray[border_mask > 0]
    return bool(values.size and float(np.median(values)) >= WHITE_BORDER_LUMA)


def _patch(gray: np.ndarray, region: str) -> np.ndarray:
    h, w = gray.shape
    match region:
        case "top_left":
            return gray[: h // 5, : w // 5]
        case "top_right":
            return gray[: h // 5, -w // 5 :]
        case "bottom_left":
            return gray[-h // 5 :, : w // 5]
        case "bottom_right":
            return gray[-h // 5 :, -w // 5 :]
        case _:
            return gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]


def _build(
    det: dict[Key, Scale],
    reasons: dict[Key, str],
    glare_id: str | None,
    occlusion_id: str | None,
) -> ObservabilityResult:
    return ObservabilityResult(
        detectability_flat={
            ObservabilityResult._key(k): v.label for k, v in det.items()
        },
        reason_codes_flat={ObservabilityResult._key(k): v for k, v in reasons.items()},
        suitability={c: v.label for c, v in _suitability(det).items()},
        glare_mask_artifact_id=glare_id,
        occlusion_mask_artifact_id=occlusion_id,
    )


def _suitability(det: dict[Key, Scale]) -> dict[str, Scale]:
    """Worst case per category: a photograph is only as good for a purpose as
    its weakest relevant region."""
    out: dict[str, Scale] = {}
    for category in CATEGORIES:
        values = [v for (_r, c, _d), v in det.items() if c == category]
        out[category] = Scale(min(values)) if values else Scale.NONE
    return out
