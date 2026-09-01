"""Surface views: deterministic enhancements ALONGSIDE the preserved original.

Every anomaly candidate records the enhancement level that surfaced it and
whether it is visible in the unenhanced view. That record is what lets I3 be
enforced later as pure logic rather than by re-examining pixels — provenance
has to survive all the way to the code enforcing the invariant.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ...provenance import EvidenceOrigin, EvidenceRef
from ...storage.artifacts import ArtifactStore
from ..geometry import GeometryResult, load_geometry
from .corners import confidence_for, severity_for

__all__ = ["ENHANCEMENTS", "SurfaceResult", "measure_surface"]

CLAHE_CLIP = 2.0
CLAHE_GRID = 8
SHARPEN_AMOUNT = 1.5
ANOMALY_CONTRAST = 38.0

#: Per-view thresholds on the local-outlier reading. One number cannot serve
#: all four: a Canny view of ordinary artwork sits near 60 while the same
#: card's unenhanced view sits near 21.
#:
#: Calibrated over 8 seeds x 3 border colours, clean against two rendered
#: scratches:
#:
#:     view      clean max   clean p50   scratched min   scratched p50
#:     original       44.5        20.7            24.7            37.2
#:     clahe          53.0        25.1            18.7            27.9
#:     sharpen        64.6        25.2            33.4            54.3
#:     edge           76.1        59.2            66.5            73.2
#:
#: Read that honestly: the two populations OVERLAP on every view. There is no
#: threshold on this measure that separates a scratched card from a clean
#: one, so this producer cannot detect scratches, and pretending otherwise
#: would mean accusing clean cards. The thresholds therefore sit above the
#: observed clean maximum: the producer stays silent on everything the corpus
#: covers and speaks only for a gross departure.
#:
#: That is the right division of labour rather than a workaround. Every
#: surface defect type is INTERPRETIVE in the taxonomy — CV cannot establish
#: one — and this producer's real output is the four VIEWS the vision layer
#: inspects, together with the provenance I3 depends on. Missing a candidate
#: is safe; inventing one is not.
VIEW_THRESHOLDS: dict[str, float] = {
    "original": 50.0,
    "clahe": 58.0,
    "sharpen": 70.0,
    "edge": 82.0,
}

#: Reproducible parameters, recorded on every enhanced reference. An
#: enhancement whose method is unrecorded cannot be audited or repeated.
ENHANCEMENTS = {
    "clahe": f"clahe:clip={CLAHE_CLIP},grid={CLAHE_GRID}",
    "sharpen": f"sharpen:amount={SHARPEN_AMOUNT}",
    "edge": "edge:canny:50,150",
}


class SurfaceResult(BaseModel):
    crops: dict[str, str] = Field(default_factory=dict)
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


def measure_surface(
    geometry: GeometryResult, store: ArtifactStore, image_hash: str
) -> SurfaceResult:
    import cv2

    result = SurfaceResult()
    if not geometry.usable:
        return result

    img = load_geometry(geometry, store).normalized
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    original_id = store.put_derived(
        image_hash, "surface", "original.png",
        cv2.imencode(".png", img)[1].tobytes(),
    )
    result.crops["original"] = original_id
    result.evidence_refs.append(
        EvidenceRef(
            artifact_id=original_id, image_hash=image_hash,
            origin=EvidenceOrigin.NORMALIZED, view="surface_original",
        )
    )
    # NOT gray.std(). The standard deviation of a printed card measures how
    # busy its artwork is, so every card cleared the threshold and surface
    # candidates were raised on all of them. A scratch is a LOCAL departure
    # from the surrounding surface, so that is what is measured.
    original_contrast = _local_outlier(gray)
    original_exceeds = original_contrast > VIEW_THRESHOLDS["original"]

    views = {
        "clahe": cv2.createCLAHE(CLAHE_CLIP, (CLAHE_GRID, CLAHE_GRID)).apply(gray),
        "sharpen": cv2.addWeighted(
            gray, 1 + SHARPEN_AMOUNT, cv2.GaussianBlur(gray, (0, 0), 3),
            -SHARPEN_AMOUNT, 0,
        ),
        "edge": cv2.Canny(gray, 50, 150),
    }

    for name, view in views.items():
        artifact_id = store.put_derived(
            image_hash, "surface", f"{name}.png",
            cv2.imencode(".png", view)[1].tobytes(),
        )
        result.crops[name] = artifact_id
        result.evidence_refs.append(
            EvidenceRef(
                artifact_id=artifact_id, image_hash=image_hash,
                origin=EvidenceOrigin.ENHANCED, enhancement=ENHANCEMENTS[name],
                view=f"surface_{name}",
            )
        )

        contrast = _local_outlier(np.asarray(view))
        if contrast > VIEW_THRESHOLDS[name]:
            visible_in_original = original_exceeds
            result.anomalies.append(
                {
                    "kind": "candidate", "category": "surface",
                    "defect_type": "scratches", "region": "center",
                    "contrast": contrast,
                    "confidence": confidence_for(contrast),
                    "severity": severity_for(contrast),
                    # The record I3 depends on: which view surfaced this, and
                    # whether an unenhanced view shows it at all.
                    "surfaced_by": "original" if visible_in_original else name,
                    "visible_in_original": visible_in_original,
                    "artifact_id": artifact_id,
                }
            )
    return result


#: Side of the square tile the surface is read in, as a fraction of the card.
#: Small enough that a scratch dominates its own tile, large enough to carry
#: texture rather than noise.
TILE_FRACTION = 0.08

#: Margin excluded from the surface reading, as a fraction of the card. The
#: border/artwork boundary is the strongest texture edge on a plain card and
#: it is DESIGN, not damage — left in, it is always the outlier. Edge and
#: corner condition is measured by their own producers, against the border.
SURFACE_MARGIN_FRACTION = 0.18


def _local_outlier(view: "np.ndarray") -> float:
    """How far the most unusual tile departs from the card's typical tile.

    Tiling and comparing against the median tile is what separates damage
    from design: printed artwork raises EVERY tile's texture together and so
    moves the median with it, while a scratch, crease or print line raises
    one tile above its neighbours. A global standard deviation cannot tell
    those apart — it reports a busy card and a damaged one identically.
    """
    import numpy as np

    data = np.asarray(view, dtype=float)
    if data.ndim > 2:
        data = data.mean(axis=2)
    margin_y = int(data.shape[0] * SURFACE_MARGIN_FRACTION)
    margin_x = int(data.shape[1] * SURFACE_MARGIN_FRACTION)
    if margin_y and margin_x:
        data = data[margin_y:-margin_y, margin_x:-margin_x]
    height, width = data.shape
    tile = max(8, int(min(height, width) * TILE_FRACTION))
    rows, cols = height // tile, width // tile
    if rows < 3 or cols < 3:
        return 0.0

    trimmed = data[: rows * tile, : cols * tile]
    tiles = trimmed.reshape(rows, tile, cols, tile).swapaxes(1, 2)
    texture = tiles.reshape(rows, cols, -1).std(axis=2)

    baseline = float(np.median(texture))
    spread = float(np.median(np.abs(texture - baseline))) * 1.4826
    if spread <= 1e-6:
        # A perfectly uniform card: any tile that differs at all is the
        # outlier, and there is no scale to express it against.
        return float(texture.max() - baseline)
    return float((texture.max() - baseline) / spread)
