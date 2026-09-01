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

#: Fraction of each edge ignored when locating the printed art. A detected
#: card boundary is approximate by nature, so the rectified image can carry
#: a sliver of background at the very edge — high-variance, and otherwise
#: read as artwork reaching the trim, which makes the border measure as zero
#: and the card as unmeasurable. Trimming is symmetric and the border widths
#: are still measured against the full width, so the ratio is unbiased.
EDGE_TRIM_FRACTION = 0.02


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
            measurable=False, reason="BORDERLESS_OR_NO_RELIABLE_REFERENCE"
        )

    gray = load_geometry(geometry, store).normalized.mean(axis=2)
    horizontal = _ratio(gray.std(axis=0))
    vertical = _ratio(gray.std(axis=1))
    if horizontal is None or vertical is None:
        return CenteringMeasurement(
            measurable=False, reason="BORDERLESS_OR_NO_RELIABLE_REFERENCE"
        )

    return CenteringMeasurement(
        measurable=True, horizontal=horizontal, vertical=vertical,
        method="border_geometry",
    )


def _ratio(variance: np.ndarray) -> float | None:
    """Locate the printed-art band, then express the leading border's share.

    50.0 means centred; above 50 means the leading border is wider.
    """
    trim = max(1, int(variance.size * EDGE_TRIM_FRACTION))
    interior = variance[trim:-trim]
    if interior.size < 2:
        return None
    threshold = interior.max() * INK_VARIANCE_FRACTION
    inked = np.where(interior > threshold)[0] + trim
    if inked.size < 2:
        return None
    leading = float(inked[0])
    trailing = float(variance.size - inked[-1] - 1)
    total = leading + trailing
    if total <= 0:
        return None
    return round(100.0 * leading / total, 2)
