"""Corner crops and anomaly CANDIDATES — explicitly not defects.

OpenCV emits measurements and candidates. Turning a high-contrast patch into
a confirmed defect is the heuristic layer's decision, and only for defect
types a measurement can establish.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...provenance import EvidenceOrigin, EvidenceRef, NormalizedBox
from ...storage.artifacts import ArtifactStore
from ..geometry import GeometryResult, load_geometry

__all__ = [
    "ANOMALY_CONTRAST",
    "CONFIDENT_CONTRAST",
    "CornerResult",
    "border_reference",
    "departure_from",
    "measure_corners",
]

CORNER_FRACTION = 0.12
ANOMALY_CONTRAST = 45.0

#: Width of the border band, as a fraction of the card, that a corner is read
#: against. Corner wear shows up in the BORDER at the corner, so the reading
#: is taken there — the old whole-crop contrast was dominated by the
#: border/artwork edge running through the crop, which is a property of the
#: card's design and not of its condition.
BORDER_BAND_FRACTION = 0.045

#: Outermost fraction of the rectified card ignored when reading the border.
#: A detected boundary is approximate, so the rectified image routinely
#: carries a sliver of background at the very edge — measured at 2 columns of
#: backdrop on a clean synthetic card, which reads as a departure of 189 and
#: fabricated a severe corner defect on a pristine card. Centering trims for
#: the same reason.
EDGE_INSET_FRACTION = 0.012

#: Contrast well past the detection threshold is a confident measurement;
#: just past it is not. Declared here rather than inlined so the heuristic's
#: MIN_CONFIDENCE_FOR_OBSERVED has something meaningful to compare against —
#: an anomaly with no confidence key defaults to 0.0 and could never be
#: promoted, making the promotion limit dead in the real pipeline.
CONFIDENT_CONTRAST = 70.0

CORNERS = {
    "top_left": (0.0, 0.0),
    "top_right": (1.0, 0.0),
    "bottom_left": (0.0, 1.0),
    "bottom_right": (1.0, 1.0),
}


class CornerResult(BaseModel):
    crops: dict[str, str] = Field(default_factory=dict)
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


def confidence_for(contrast: float) -> float:
    span = max(CONFIDENT_CONTRAST - ANOMALY_CONTRAST, 1e-6)
    return round(min(1.0, max(0.0, (contrast - ANOMALY_CONTRAST) / span)), 3)


def severity_for(contrast: float) -> str:
    if contrast > CONFIDENT_CONTRAST * 1.5:
        return "severe"
    return "moderate" if contrast > CONFIDENT_CONTRAST else "minor"


def measure_corners(
    geometry: GeometryResult, store: ArtifactStore, image_hash: str
) -> CornerResult:
    import cv2

    result = CornerResult()
    if not geometry.usable:
        return result

    artifacts = load_geometry(geometry, store)
    img = artifacts.normalized
    border_mask = artifacts.border_mask
    height, width = img.shape[:2]
    size_x, size_y = int(width * CORNER_FRACTION), int(height * CORNER_FRACTION)
    band_x = max(2, int(width * BORDER_BAND_FRACTION))
    band_y = max(2, int(height * BORDER_BAND_FRACTION))
    reference = border_reference(img, band_x, band_y, border_mask)

    for name, (fx, fy) in CORNERS.items():
        x0 = 0 if fx == 0.0 else width - size_x
        y0 = 0 if fy == 0.0 else height - size_y
        patch = img[y0 : y0 + size_y, x0 : x0 + size_x]

        artifact_id = store.put_derived(
            image_hash, "corners", f"{name}.png",
            cv2.imencode(".png", patch)[1].tobytes(),
        )
        result.crops[name] = artifact_id
        result.evidence_refs.append(
            EvidenceRef(
                artifact_id=artifact_id, image_hash=image_hash,
                origin=EvidenceOrigin.NORMALIZED,
                # Its own region, so two corners never share a location and
                # fusion cannot merge them into one defect.
                region=NormalizedBox(
                    x0=x0 / width, y0=y0 / height,
                    x1=(x0 + size_x) / width, y1=(y0 + size_y) / height,
                ),
                view=f"corner_{name}",
            )
        )

        # Read the border band at this corner, against the same band at the
        # middle of the card's edges. Corner wear is a local departure from
        # how THIS card's border looks; whole-crop contrast instead measured
        # whether a border/artwork edge crossed the crop, so it fired on
        # every white-bordered card and scored a destroyed card lower than a
        # pristine one.
        contrast = _corner_departure(img, fx, fy, band_x, band_y, reference,
                                     border_mask)
        if contrast > ANOMALY_CONTRAST:
            result.anomalies.append(
                {
                    "kind": "candidate", "region": name, "category": "corners",
                    "defect_type": "rounding", "contrast": contrast,
                    "confidence": confidence_for(contrast),
                    "severity": severity_for(contrast),
                    "artifact_id": artifact_id,
                }
            )
    return result


def _inside_border(mask, region_slices, height, width):
    """The border pixels inside a region, or None when there is no mask.

    Geometry already segments the border, so guessing its width from a
    fraction of the card is unnecessary — and wrong on any card whose border
    is thinner than the guess, where the band spills into the artwork and
    reads a design as damage.
    """
    if mask is None or mask.shape[:2] != (height, width):
        return None
    ys, xs = region_slices
    return mask[ys, xs] > 0


def border_reference(img: "Any", band_x: int, band_y: int,
                      mask=None) -> float:
    """How this card's own border reads, away from the corners.

    Taken from the middle of all four edges, so a card worn at every corner
    still has an undamaged baseline to be compared against.
    """
    import numpy as np

    gray = img.mean(axis=2)
    height, width = gray.shape
    inset_x = max(1, int(width * EDGE_INSET_FRACTION))
    inset_y = max(1, int(height * EDGE_INSET_FRACTION))
    quarter_w, quarter_h = width // 4, height // 4
    regions = [
        (slice(inset_y, inset_y + band_y), slice(quarter_w, width - quarter_w)),
        (slice(height - inset_y - band_y, height - inset_y),
         slice(quarter_w, width - quarter_w)),
        (slice(quarter_h, height - quarter_h), slice(inset_x, inset_x + band_x)),
        (slice(quarter_h, height - quarter_h),
         slice(width - inset_x - band_x, width - inset_x)),
    ]
    values = []
    for region in regions:
        patch = gray[region]
        keep = _inside_border(mask, region, height, width)
        values.append(patch[keep].ravel() if keep is not None and keep.any()
                      else patch.ravel())
    return float(np.median(np.concatenate(values)))


def _corner_departure(
    img: "Any", fx: float, fy: float, band_x: int, band_y: int,
    reference: float, mask=None,
) -> float:
    """How far this corner's border departs from the card's own border.

    The L-shaped band that actually turns the corner, not the square crop:
    the crop reaches into the artwork, and the artwork is not evidence about
    condition.
    """
    import numpy as np

    gray = img.mean(axis=2)
    height, width = gray.shape
    inset_x = max(1, int(width * EDGE_INSET_FRACTION))
    inset_y = max(1, int(height * EDGE_INSET_FRACTION))
    reach_x, reach_y = int(width * CORNER_FRACTION), int(height * CORNER_FRACTION)

    xs = (slice(inset_x, reach_x) if fx == 0.0
          else slice(width - reach_x, width - inset_x))
    ys = (slice(inset_y, reach_y) if fy == 0.0
          else slice(height - reach_y, height - inset_y))
    horizontal = (slice(inset_y, inset_y + band_y) if fy == 0.0
                  else slice(height - inset_y - band_y, height - inset_y))
    vertical = (slice(inset_x, inset_x + band_x) if fx == 0.0
                else slice(width - inset_x - band_x, width - inset_x))

    return max(departure_from(img, (horizontal, xs), reference, mask),
               departure_from(img, (ys, vertical), reference, mask))


def departure_from(img: "Any", region, reference: float, mask=None) -> float:
    """How far a region's BORDER pixels depart from the card's own border.

    The 95th percentile rather than the mean: wear occupies a fraction of the
    band, and averaging it against clean border would hide it.
    """
    import numpy as np

    gray = img.mean(axis=2)
    height, width = gray.shape
    patch = gray[region]
    keep = _inside_border(mask, region, height, width)
    edge = _off_the_edge(region, height, width)
    if keep is None:
        keep = edge
    else:
        keep = keep & edge
    band = patch[keep].ravel() if keep.any() else patch.ravel()
    if band.size == 0:
        return 0.0
    return float(np.percentile(np.abs(band - reference), 95))


def _off_the_edge(region, height: int, width: int):
    """Pixels far enough in from the frame to be card rather than backdrop.

    A detected boundary is approximate, so the rectified image carries a
    sliver of background at its outer edge — and geometry's border mask
    includes it. Left in, that backdrop reads as an enormous departure from
    the border and fabricates a defect on a clean card.
    """
    import numpy as np

    inset_y = max(1, int(height * EDGE_INSET_FRACTION))
    inset_x = max(1, int(width * EDGE_INSET_FRACTION))
    ok = np.zeros((height, width), dtype=bool)
    ok[inset_y : height - inset_y, inset_x : width - inset_x] = True
    return ok[region]
