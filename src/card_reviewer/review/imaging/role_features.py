"""Layout signatures used to infer which face a photograph shows.

Separate from cv_measurements on purpose: this is about layout, not
defects, and role resolution consumes it before any defect analysis runs.
Keeping it apart means a defect-analyzer bump does not invalidate role
resolution, and a layout change does not invalidate defect measurements.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from ..storage.artifacts import ArtifactStore
from ..versions import ROLE_FEATURES_VERSION
from .geometry import GeometryResult, load_geometry

__all__ = ["ROLE_FEATURES_VERSION", "RoleFeatures", "extract_role_features"]

#: A back's detail is small and high-frequency — stat lines, the card
#: number, the copyright block. A front's is one large image region.
EDGE_DENSITY_SCALE = 0.25
#: Below this share of edge pixels, the centre reads as one coherent image
#: rather than as text. A v1 declared threshold: card backs are text-dense
#: (edge density typically well above this), while a front's central artwork
#: is comparatively smooth. Expected to be revisited against the golden
#: real-image fixtures, where genuine photographs replace synthetic ground
#: truth. A wrong threshold degrades to `unknown` rather than to a wrong
#: face, because the role resolver's ambiguous band is deliberately wide.
CENTRAL_REGION_MAX_EDGE_DENSITY = 0.05


class RoleFeatures(BaseModel):
    text_density: float = Field(default=0.0, ge=0.0, le=1.0)
    has_central_image_region: bool = False
    layout_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    version: str = ROLE_FEATURES_VERSION


def extract_role_features(
    geometry: GeometryResult, store: ArtifactStore
) -> RoleFeatures:
    import cv2

    if not geometry.usable:
        # Unknown is a first-class state (spec §8). Never guess a face from
        # an image whose geometry could not be established.
        return RoleFeatures()

    normalized = load_geometry(geometry, store).normalized
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    text_density = float(min(1.0, (edges > 0).mean() / EDGE_DENSITY_SCALE))

    height, width = gray.shape
    centre = gray[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    centre_density = float((cv2.Canny(centre, 60, 160) > 0).mean())

    return RoleFeatures(
        text_density=text_density,
        has_central_image_region=centre_density <= CENTRAL_REGION_MAX_EDGE_DENSITY,
        layout_confidence=float(geometry.boundary_confidence),
    )
