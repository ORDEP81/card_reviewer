"""Candidate-level evidence assembly (model here; `assemble` in Task 28).

`Assembled` is a cached stage output stored as JSON in SQLite, so its tuple
keys are flattened to "role|category|defect_type" strings. Every consumer
reads the tuple view through a property, so the flattening never leaks.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .enums import Scale
from .provenance import EvidenceRef
from .roles import ImageRole
from .versions import ASSEMBLY_VERSION

__all__ = ["ASSEMBLY_VERSION", "Assembled", "ImageStageOutputs"]


class Assembled(BaseModel):
    detectability_flat: dict[str, str] = Field(default_factory=dict)
    reason_codes_flat: dict[str, str] = Field(default_factory=dict)
    provenance_flat: dict[str, str] = Field(default_factory=dict)
    best_for: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: dict[str, list[EvidenceRef]] = Field(default_factory=dict)
    faces_present: list[str] = Field(default_factory=list)
    centering: dict[str, Any] = Field(default_factory=dict)
    version: str = ASSEMBLY_VERSION

    @staticmethod
    def key(role: ImageRole, category: str, defect_type: str) -> str:
        return f"{role.value}|{category}|{defect_type}"

    @staticmethod
    def _unkey(k: str) -> tuple[ImageRole, str, str]:
        role, category, defect_type = k.split("|")
        return (ImageRole(role), category, defect_type)

    @property
    def detectability(self) -> dict[tuple[ImageRole, str, str], Scale]:
        return {self._unkey(k): Scale(v) for k, v in self.detectability_flat.items()}

    @property
    def reason_codes(self) -> dict[tuple[ImageRole, str, str], str]:
        return {self._unkey(k): v for k, v in self.reason_codes_flat.items()}

    @property
    def provenance(self) -> dict[tuple[ImageRole, str, str], str]:
        return {self._unkey(k): v for k, v in self.provenance_flat.items()}

    @property
    def faces(self) -> tuple[ImageRole, ...]:
        return tuple(ImageRole(f) for f in self.faces_present)


class ImageStageOutputs(BaseModel):
    """Every image-tier stage's output for one photograph, as cached dicts.

    The pipeline collects these; keeping the raw stage outputs means the
    assembly fingerprint is over exactly what the stages produced.
    """

    image_hash: str
    preflight: dict[str, Any] = Field(default_factory=dict)
    geometry: dict[str, Any] | None = None
    observability: dict[str, Any] | None = None
    cv_measurements: dict[str, Any] | None = None
    role_features: dict[str, Any] | None = None

    @property
    def usable(self) -> bool:
        return self.geometry is not None
