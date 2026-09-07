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

#: Clipping so heavy that the region is blown out whatever the card's design.
#: The relative test above is blind once HALF the regions are glared — the
#: median moves with them and nothing stands out — and the old absolute arm
#: was switched off on white borders, which is exactly the population that
#: clips. Measured over 6 seeds x 3 border colours: clean regions top out at
#: 0.480 and every glared region starts at 0.695.
HEAVY_CLIP_FRACTION = 0.60

#: Fraction of a region that must be under an obstruction before the region
#: stops being assessable. A thumb, a finger, a holder's clip: opaque, and no
#: amount of interpretation recovers what is behind it. Measured over 8 seeds
#: x 3 border colours, clean regions reach 0.014 and an obstructed corner
#: starts at 0.189 — a clean separation, so the threshold sits between them.
OCCLUSION_FRACTION = 0.10

#: Darkness so complete that nothing is behind it but the obstruction. The
#: fraction alone cannot be absolute the way it was: a card with dark artwork
#: reads 0.42-0.49 of its own regions as "dark" and was reported obstructed
#: everywhere, turning the card's design into a photograph problem. Excess
#: Occlusion is judged ONLY between regions of the same kind — the four
#: corners against each other. The centre is artwork and a corner region
#: contains border, so comparing them compares two materials: a card with
#: dark artwork reads 0.92 at its centre against 0.51 at its corners and was
#: reported obstructed there, with a photo request to move an obstruction
#: that does not exist. No absolute arm either, for the same reason — a dark
#: card is uniformly dark, and every absolute threshold low enough to catch a
#: real obstruction also catches the design.
COMPARABLE_OCCLUSION_REGIONS = ("top_left", "top_right",
                                "bottom_left", "bottom_right")

# There is deliberately NO region-level blur test here.
#
# Whole-image softness is preflight's job and it does the job: a 31-pixel
# blur takes global sharpness to 0.7 against a floor of 25. What preflight
# cannot see is a card sharp overall and soft in one place, and two measures
# were tried for it. Both fail on the same rock — a region can be smooth
# BY DESIGN, and neither measure separates that from out of focus:
#
#     Laplacian variance vs sibling regions
#         clean minimum ratio 0.137, genuinely soft corner 0.489
#         (the clean card looks SOFTER than the blurred one)
#     Laplacian variance normalised by the region's own contrast
#         clean minimum 0.0475, genuinely soft corner up to 0.0937
#
# Either threshold would report BLUR on clean cards and ask their owners to
# re-photograph something already in focus, or sit low enough to never fire.
# Occlusion and resolution are kept because they DO separate: occlusion by a
# factor of 13, resolution because it is an exact pixel count rather than an
# estimate.

#: Minimum ORIGINAL pixels across a region's short side. Effective
#: resolution, not rectified size: the rectified card is always NORM_W x
#: NORM_H, so measuring the normalized patch asks a constant and answers
#: itself. What matters is how many photographed pixels back that patch —
#: a card occupying a small part of a low-resolution photo is upscaled into
#: the same rectangle, and the detail is not there however large the crop.
REGION_MIN_PX = 24


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

    # Original pixels per rectified pixel. Below 1.0 the warp upscaled, and
    # a region's apparent size overstates the detail actually captured.
    scale = _capture_scale(geometry, gray.shape)

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
        # Only the corner regions, and only against each other.
        dark = {
            region: float((_patch(gray, region) <= OCCLUSION_LUMA).mean())
            for region in regions if region in COMPARABLE_OCCLUSION_REGIONS
        }
        dark_baseline = float(np.median(list(dark.values()))) if dark else 0.0
        for region in regions:
            patch = _patch(gray, region)
            clipped = fractions[region] > GLARE_FRACTION
            stands_out = fractions[region] > baseline + GLARE_EXCESS_FRACTION
            bright = float(patch.mean()) >= WHITE_BORDER_LUMA
            # Everything that used to fall through to HIGH. The occlusion
            # mask was already being computed here and then thrown away.
            # Relative to the card's own darkness, plus an absolute arm for
            # the case the relative test cannot see — every region obstructed
            # at once, where there is no unobstructed sibling to stand out
            # from.
            occluded = (
                region in dark
                and dark[region] > dark_baseline + OCCLUSION_FRACTION
            )
            too_small = min(patch.shape[:2]) * scale < REGION_MIN_PX
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
                elif occluded:
                    det[key], reasons[key] = Scale.LOW, "OCCLUSION"
                elif too_small:
                    det[key], reasons[key] = Scale.LOW, "LOW_RESOLUTION"
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


def _capture_scale(geometry: "GeometryResult", normalized_shape) -> float:
    """How many photographed pixels back one rectified pixel.

    Taken from the detected quad's area against the rectified area, so a card
    photographed small — or a thumbnail padded up to a usable size — reports
    the resolution it actually has rather than the resolution it was
    stretched to.
    """
    quad = np.asarray(geometry.quad or [], dtype=float)
    if quad.shape != (4, 2):
        return 1.0
    x, y = quad[:, 0], quad[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    rectified = float(normalized_shape[0] * normalized_shape[1])
    if rectified <= 0 or area <= 0:
        return 1.0
    return (area / rectified) ** 0.5
