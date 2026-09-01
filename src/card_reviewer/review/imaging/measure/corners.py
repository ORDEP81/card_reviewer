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
    "measure_corners",
]

CORNER_FRACTION = 0.12
ANOMALY_CONTRAST = 45.0

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

    img = load_geometry(geometry, store).normalized
    height, width = img.shape[:2]
    size_x, size_y = int(width * CORNER_FRACTION), int(height * CORNER_FRACTION)

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

        contrast = float(patch.mean(axis=2).std())
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
