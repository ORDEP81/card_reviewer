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
from .roles import ImageRole, ResolvedRole
from .versions import ASSEMBLY_VERSION

__all__ = [
    "ASSEMBLY_VERSION",
    "Assembled",
    "ImageEvidence",
    "ImageStageOutputs",
    "assemble",
    "to_image_evidence",
]


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


class ImageEvidence(BaseModel):
    """One image's contribution to assembly, already role-independent."""

    image_hash: str
    detectability: dict[tuple[str, str, str], Scale] = Field(default_factory=dict)
    #: Carried through from ObservabilityResult. Dropping these is why
    #: WHITE_BORDER would never reach the coverage policy, silently turning
    #: every structural limitation into a circumstantial one and making the
    #: structural exemption unreachable end to end.
    reason_codes: dict[tuple[str, str, str], str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    sharpness: float = 0.0
    centering: dict[str, Any] = Field(default_factory=dict)
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: dict[str, list[EvidenceRef]] = Field(default_factory=dict)


#: Two centering readings from different photographs of the same face further
#: apart than this are a genuine disagreement, preserved rather than averaged.
CONFLICT_THRESHOLD_PP = 5.0


def assemble(
    images: list[ImageEvidence], roles: dict[str, ResolvedRole]
) -> Assembled:
    """Fuse per-image results into one view of the card (spec §9).

    A corner glared in one photo and clear in another is OBSERVABLE, and the
    assembly records which image established that. Conflicting measurements
    are preserved rather than averaged away — the disagreement is
    information.
    """
    out = Assembled()
    faces: set[ImageRole] = set()

    for image in images:
        role = roles[image.image_hash].role
        if role is ImageRole.UNKNOWN:
            # Still contributes face-independent work, but never claims a face.
            out.anomalies.extend(image.anomalies)
            continue

        faces.add(role)
        for (region, category, defect_type), value in image.detectability.items():
            key = Assembled.key(role, category, defect_type)
            # Best-of across images: a defect visible in ANY photo is
            # observable, and the reason travels with the value it explains.
            if value > Scale(out.detectability_flat.get(key, Scale.NONE.label)):
                out.detectability_flat[key] = value.label
                out.provenance_flat[key] = image.image_hash
                code = image.reason_codes.get((region, category, defect_type))
                if code and value < Scale.MODERATE:
                    out.reason_codes_flat[key] = code
                else:
                    # A better photograph resolved it; it is no longer a
                    # limitation on this card.
                    out.reason_codes_flat.pop(key, None)

        out.limitations.extend(image.limitations)
        out.anomalies.extend(image.anomalies)
        for purpose, refs in image.evidence_refs.items():
            out.evidence_refs.setdefault(purpose, []).extend(refs)

    out.faces_present = sorted(face.value for face in faces)
    out.best_for = _best_for(images, roles)
    out.conflicts = _conflicts(images, roles)
    fronts = [i for i in images if roles[i.image_hash].role is ImageRole.FRONT]
    out.centering = fronts[0].centering if fronts else {}
    return out


def _best_for(
    images: list[ImageEvidence], roles: dict[str, ResolvedRole]
) -> dict[str, str]:
    fronts = [i for i in images if roles[i.image_hash].role is ImageRole.FRONT]
    if not fronts:
        return {}
    sharpest = max(fronts, key=lambda i: i.sharpness)
    return {"surface": sharpest.image_hash, "centering": fronts[0].image_hash}


def _conflicts(
    images: list[ImageEvidence], roles: dict[str, ResolvedRole]
) -> list[dict[str, Any]]:
    values = [
        (i.image_hash, i.centering.get("horizontal"))
        for i in images
        if roles[i.image_hash].role is ImageRole.FRONT
        and i.centering.get("horizontal") is not None
    ]
    if len(values) < 2:
        return []
    numbers = [v for _, v in values]
    if max(numbers) - min(numbers) <= CONFLICT_THRESHOLD_PP:
        return []
    return [
        {"field": "centering.horizontal", "values": numbers,
         "images": [h for h, _ in values]}
    ]


def to_image_evidence(image_outputs: list[ImageStageOutputs]) -> list[ImageEvidence]:
    """Adapt cached image-tier outputs into assembly inputs.

    This is the one place the measurement modules' `evidence_refs` lists
    become the "category:defect_type" keyed map the heuristic looks up. If
    that mapping were left implicit and the keys failed to line up, no CV
    finding would ever be emitted and every card would silently pass, with
    no error anywhere to notice.
    """
    from .imaging.measure import CvMeasurements
    from .imaging.observability import ObservabilityResult
    from .taxonomy import defect_types_for

    out: list[ImageEvidence] = []
    for image in image_outputs:
        if not image.usable:
            # One bad photograph out of six must not fail the card.
            continue
        obs = ObservabilityResult.model_validate(image.observability or {})
        cv = CvMeasurements.model_validate(image.cv_measurements or {})

        refs: dict[str, list[EvidenceRef]] = {}
        for category, group in (("corners", cv.corners), ("edges", cv.edges),
                                ("surface", cv.surface)):
            for defect_type in defect_types_for(category):
                for ref in group.evidence_refs:
                    # Region-scoped as well as category-scoped: giving every
                    # defect type all of a category's refs would union their
                    # boxes into one location, so anomalies at opposite
                    # corners would fuse into a single defect.
                    region = _region_of(ref.view)
                    if region:
                        refs.setdefault(
                            f"{category}:{defect_type}:{region}", []
                        ).append(ref)
                    refs.setdefault(f"{category}:{defect_type}", []).append(ref)

        detectability = dict(obs.detectability)
        reason_codes = dict(obs.reason_codes)
        if cv.centering.get("measurable"):
            refs["centering:border_ratio"] = [
                r for r in cv.surface.evidence_refs if r.view == "surface_original"
            ]
        else:
            # The measurement declined. Without this the heuristic emits no
            # centering finding and coverage counts centering as assessed —
            # "could not measure" would silently read as "nothing wrong",
            # which is the I2 failure exactly.
            reason = cv.centering.get("reason") or "BORDERLESS_DESIGN"
            for key in list(detectability):
                if key[1] == "centering":
                    detectability[key] = min(detectability[key], Scale.LOW)
                    reason_codes[key] = reason

        out.append(
            ImageEvidence(
                image_hash=image.image_hash,
                detectability=detectability,
                reason_codes=reason_codes,
                sharpness=float(image.preflight.get("global_sharpness", 0.0)),
                centering=cv.centering,
                anomalies=cv.anomalies,
                evidence_refs=refs,
            )
        )
    return out


def _region_of(view: str) -> str | None:
    """The region a measurement view describes, e.g. corner_bottom_left."""
    for prefix in ("corner_", "edge_"):
        if view.startswith(prefix):
            return view[len(prefix):]
    return None
