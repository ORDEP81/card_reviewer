"""The cv_measurements stage output: one cached document per image.

The four measurement functions need a single declared aggregate, because
the stage caches one JSON document rather than four loose return values.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...storage.artifacts import ArtifactStore
from ...versions import CV_VERSION
from ..geometry import GeometryResult
from .centering import CenteringMeasurement, measure_centering
from .corners import CornerResult, measure_corners
from .edges import EdgeResult, measure_edges
from .surface import SurfaceResult, measure_surface

__all__ = [
    "CenteringMeasurement",
    "CornerResult",
    "CvMeasurements",
    "EdgeResult",
    "SurfaceResult",
    "measure_all",
    "measure_centering",
    "measure_corners",
    "measure_edges",
    "measure_surface",
]


class CvMeasurements(BaseModel):
    centering: dict[str, Any] = Field(default_factory=dict)
    corners: CornerResult = Field(default_factory=CornerResult)
    edges: EdgeResult = Field(default_factory=EdgeResult)
    surface: SurfaceResult = Field(default_factory=SurfaceResult)
    version: str = CV_VERSION

    @property
    def anomalies(self) -> list[dict[str, Any]]:
        """All candidates from every region, in one list for assembly."""
        return [*self.corners.anomalies, *self.edges.anomalies,
                *self.surface.anomalies]


def measure_all(
    geometry: GeometryResult, store: ArtifactStore, image_hash: str
) -> CvMeasurements:
    return CvMeasurements(
        centering=measure_centering(geometry, store).model_dump(),
        corners=measure_corners(geometry, store, image_hash),
        edges=measure_edges(geometry, store, image_hash),
        surface=measure_surface(geometry, store, image_hash),
    )
