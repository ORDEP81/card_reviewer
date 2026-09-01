"""Edge strips and anomaly candidates. Same contract as corners."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...provenance import EvidenceOrigin, EvidenceRef, NormalizedBox
from ...storage.artifacts import ArtifactStore
from ..geometry import GeometryResult, load_geometry
from .corners import (
    ANOMALY_CONTRAST, border_reference, confidence_for, departure_from,
    severity_for,
)

__all__ = ["EdgeResult", "measure_edges"]

EDGE_FRACTION = 0.06


class EdgeResult(BaseModel):
    crops: dict[str, str] = Field(default_factory=dict)
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


def measure_edges(
    geometry: GeometryResult, store: ArtifactStore, image_hash: str
) -> EdgeResult:
    import cv2

    result = EdgeResult()
    if not geometry.usable:
        return result

    artifacts = load_geometry(geometry, store)
    img = artifacts.normalized
    border_mask = artifacts.border_mask
    height, width = img.shape[:2]
    band_y, band_x = int(height * EDGE_FRACTION), int(width * EDGE_FRACTION)
    reference = border_reference(
        img, max(2, int(width * 0.045)), max(2, int(height * 0.045)), border_mask)

    slices = {
        "top": (slice(0, band_y), slice(0, width)),
        "bottom": (slice(height - band_y, height), slice(0, width)),
        "left": (slice(0, height), slice(0, band_x)),
        "right": (slice(0, height), slice(width - band_x, width)),
    }
    for name, (ys, xs) in slices.items():
        patch = img[ys, xs]
        artifact_id = store.put_derived(
            image_hash, "edges", f"{name}.png",
            cv2.imencode(".png", patch)[1].tobytes(),
        )
        result.crops[name] = artifact_id
        result.evidence_refs.append(
            EvidenceRef(
                artifact_id=artifact_id, image_hash=image_hash,
                origin=EvidenceOrigin.NORMALIZED,
                region=NormalizedBox(
                    x0=xs.start / width, y0=ys.start / height,
                    x1=xs.stop / width, y1=ys.stop / height,
                ),
                view=f"edge_{name}",
            )
        )

        # Against this card's own border, not the strip's internal spread.
        # An edge strip crosses the border/artwork boundary on every bordered
        # card, so its standard deviation described the design rather than
        # the condition and fired on clean cards.
        contrast = departure_from(img, (ys, xs), reference, border_mask)
        if contrast > ANOMALY_CONTRAST:
            result.anomalies.append(
                {
                    "kind": "candidate", "region": name, "category": "edges",
                    "defect_type": "chipping", "contrast": contrast,
                    "confidence": confidence_for(contrast),
                    "severity": severity_for(contrast),
                    "artifact_id": artifact_id,
                }
            )
    return result
