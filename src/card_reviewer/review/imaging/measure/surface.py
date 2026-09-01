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
    original_contrast = float(gray.std())

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

        contrast = float(np.asarray(view).std())
        if contrast > ANOMALY_CONTRAST:
            visible_in_original = original_contrast > ANOMALY_CONTRAST
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
