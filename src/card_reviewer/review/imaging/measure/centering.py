"""Centering: measurement, never acceptability (spec §7.4).

Reports the ratio and the precision the method actually supports. Whether
54/46 passes on Prizm and fails on Bowman Chrome is the heuristic layer's
decision — CENTERING_PRODUCT_LENIENCY_001 does not apply in this module,
and no field here says whether a card passes.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from ...storage.artifacts import ArtifactStore
from ..geometry import GeometryResult, load_geometry

__all__ = ["PRECISION_PP", "CenteringMeasurement", "measure_centering"]

#: The precision this method genuinely supports, in percentage points.
#: CENTERING_NO_OVERMEASURE_001's corollary: never report more precision
#: than the measurement can carry.
PRECISION_PP = 1.5

#: A column counts as printed art when its vertical variance rises this far
#: above the quietest column — a border column is uniform, an art column is
#: not.
INK_VARIANCE_FRACTION = 0.25

#: The percentile of interior column variance the ink threshold is taken
#: against. The peak column is not a safe reference: warp resampling on a
#: tilted photograph puts a single narrow variance spike in the profile, and
#: a quarter of that spike sits above the real artwork — so the "ink band"
#: collapses onto the spike and the border widths become fiction. A high
#: percentile describes the artwork's typical variance instead of its most
#: extreme column, which is what the border is actually being compared to.
INK_REFERENCE_PERCENTILE = 75.0

#: Fraction of each edge ignored when locating the printed art. A detected
#: card boundary is approximate by nature, so the rectified image can carry
#: a sliver of background at the very edge — high-variance, and otherwise
#: read as artwork reaching the trim, which makes the border measure as zero
#: and the card as unmeasurable. Trimming is symmetric and the border widths
#: are still measured against the full width, so the ratio is unbiased.
EDGE_TRIM_FRACTION = 0.02

#: Centering is read from the middle of each edge, using this fraction of the
#: perpendicular span. Corners are where damage lives, and a column's variance
#: is taken down its whole height — so a single chewed corner would otherwise
#: make the entire leading border read as printed art and the card as
#: unmeasurable. It is also how a person eyeballs centering.
CENTRAL_BAND_FRACTION = 0.6



class CenteringMeasurement(BaseModel):
    measurable: bool
    horizontal: float | None = None
    vertical: float | None = None
    method: str | None = None
    precision_pp: float = PRECISION_PP
    reason: str | None = None


def measure_centering(
    geometry: GeometryResult, store: ArtifactStore
) -> CenteringMeasurement:
    if not geometry.usable or not geometry.has_reliable_border:
        # Never force a border ratio onto a design that has no border
        # reference (CENTERING_BORDERLESS_001).
        return CenteringMeasurement(
            measurable=False, reason="BORDERLESS_DESIGN"
        )

    gray = load_geometry(geometry, store).normalized.mean(axis=2)
    horizontal, h_reason = _ratio(_central(gray, axis=0).std(axis=0))
    vertical, v_reason = _ratio(_central(gray, axis=1).std(axis=1))
    if horizontal is None or vertical is None:
        return CenteringMeasurement(
            measurable=False,
            reason=h_reason or v_reason or "BORDERLESS_DESIGN",
        )

    return CenteringMeasurement(
        measurable=True, horizontal=horizontal, vertical=vertical,
        method="border_geometry",
    )


def _central(gray: np.ndarray, axis: int) -> np.ndarray:
    """The middle band of the span perpendicular to the one being measured."""
    length = gray.shape[1 - axis] if axis == 0 else gray.shape[1 - axis]
    margin = int(length * (1.0 - CENTRAL_BAND_FRACTION) / 2.0)
    if margin < 1:
        return gray
    return gray[margin:-margin, :] if axis == 0 else gray[:, margin:-margin]


def _ratio(variance: np.ndarray) -> tuple[float | None, str | None]:
    """Locate the printed-art band, then express the leading border's share.

    50.0 means centred; above 50 means the leading border is wider. Returns
    `(None, reason)` when the border cannot support a number at all — the
    only honest answer when the band we would measure against is not
    distinguishable from the art.
    """
    trim = max(1, int(variance.size * EDGE_TRIM_FRACTION))
    interior = variance[trim:-trim]
    if interior.size < 2:
        return None, "BORDERLESS_DESIGN"
    threshold = (float(np.percentile(interior, INK_REFERENCE_PERCENTILE))
                 * INK_VARIANCE_FRACTION)
    inked = np.where(interior > threshold)[0] + trim
    if inked.size < 2:
        return None, "BORDERLESS_DESIGN"

    # The trim is a tolerance for an approximate boundary, NOT a border. If
    # the ink band runs into it, no border was located on that side and the
    # width we would report is the trim's width rather than the card's.
    # Reporting it anyway is how a 50/50 card at a 10-degree tilt came out as
    # a 23/77 miscut, and centering is the one measurement that can reject.
    if inked[0] <= trim or inked[-1] >= variance.size - trim - 1:
        return None, "BORDER_NOT_SEPARABLE_FROM_ART"

    leading = float(inked[0])
    trailing = float(variance.size - inked[-1] - 1)
    total = leading + trailing
    if total <= 0:
        return None, "BORDERLESS_DESIGN"
    return round(100.0 * leading / total, 2), None
