"""VisionProvider contract. Anthropic is one implementation, not the interface.

Two things this module protects:

Per-category assessability is REQUIRED, not an optional remark. It feeds the
authoritative coverage evaluation, so a provider saying "I could not judge
the surface" must not be lost.

Provenance must survive the round trip. The provider returns bare artifact
ids; rebuilding an EvidenceRef from one — inventing ORIGINAL and an empty
image hash — would silently launder an enhancement-only finding into one
that satisfies I3, defeating the invariant at exactly the point it matters.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from ..enums import FindingState
from ..findings import Finding, FindingProducer, Severity
from ..provenance import EvidenceRef, NormalizedBox
from ..taxonomy import CATEGORIES

__all__ = [
    "Assessment",
    "FakeProvider",
    "GemView",
    "ProviderContractError",
    "VisionFinding",
    "VisionProvider",
    "parse_assessment",
    "resolve_vision_findings",
]


class ProviderContractError(Exception):
    """The provider returned something outside its declared contract."""


class GemView(StrEnum):
    NO_DISQUALIFIER = "no_visible_psa10_disqualifier"
    POSSIBLE_DISQUALIFIER = "possible_psa10_disqualifier"
    VISIBLE_DISQUALIFIER = "visible_psa10_disqualifier"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class VisionFinding(BaseModel):
    defect_type: str
    category: str
    state: FindingState
    confidence: float = Field(ge=0.0, le=1.0)
    #: Advisory only. Our rubric decides PSA-10 relevance; this records what
    #: the provider claimed so the two can be compared.
    psa10_relevant: bool
    evidence_artifact_ids: list[str] = Field(min_length=1)
    severity: Severity | None = None
    #: When omitted, derived from the cited refs' regions — a location is
    #: required downstream for fusion and the contradiction test, so it can
    #: never simply be absent.
    location: NormalizedBox | None = None
    explanation: str = ""


class Assessment(BaseModel):
    findings: list[VisionFinding] = Field(default_factory=list)
    category_assessability: dict[str, bool]
    gem_view: GemView
    disagreements: list[str] = Field(default_factory=list)


class VisionProvider(Protocol):
    def assess(self, evidence_manifest: dict[str, Any]) -> Assessment: ...

    def signature(self) -> dict[str, Any]:
        """Provider identity for the vision stage's producer signature.

        Exactly `provider`, `model`, `prompt_version` and `inference_params`.
        Exposing it here is what lets the pipeline cache a vision result
        without importing or knowing anything about Anthropic.
        """
        ...


def parse_assessment(
    payload: dict[str, Any], allowed_artifact_ids: set[str]
) -> Assessment:
    # No explicit gem_view check: it is a required field, so model_validate
    # already rejects a response without one and names it in the message.
    # A hand-written guard here was redundant — mutation testing showed
    # removing it changed nothing.
    try:
        assessment = Assessment.model_validate(payload)
    except ValidationError as exc:
        raise ProviderContractError(f"malformed provider response: {exc}") from exc

    missing = set(CATEGORIES) - set(assessment.category_assessability)
    if missing:
        raise ProviderContractError(
            f"response omits assessability for {sorted(missing)} — coverage "
            "cannot be evaluated without it"
        )

    for finding in assessment.findings:
        unknown = set(finding.evidence_artifact_ids) - allowed_artifact_ids
        if unknown:
            raise ProviderContractError(
                f"finding cites artifacts {sorted(unknown)} not in the manifest "
                "it was sent"
            )
    return assessment


def resolve_vision_findings(
    assessment: Assessment, manifest_index: dict[str, EvidenceRef]
) -> list[Finding]:
    """Turn provider output into Findings WITHOUT losing provenance.

    Every cited id is resolved against the manifest that was actually sent.
    An id that does not resolve is a contract violation, never a default.
    """
    out: list[Finding] = []
    for vf in assessment.findings:
        refs: list[EvidenceRef] = []
        for artifact_id in vf.evidence_artifact_ids:
            ref = manifest_index.get(artifact_id)
            if ref is None:
                raise ProviderContractError(
                    f"finding cites artifact {artifact_id!r} which is not in "
                    "the manifest that was sent"
                )
            refs.append(ref)
        out.append(
            Finding(
                defect_type=vf.defect_type, category=vf.category, state=vf.state,
                producer=FindingProducer.VISION, confidence=vf.confidence,
                # Provisional: relevance resolution recomputes this from the
                # matched rubric rules and overrides the provider's claim.
                psa10_relevant=vf.psa10_relevant,
                severity=vf.severity,
                location=vf.location or _derive_location(refs),
                evidence=refs, explanation=vf.explanation,
            )
        )
    return out


def _derive_location(refs: list[EvidenceRef]) -> NormalizedBox | None:
    """Union of the cited refs' regions, when the provider gave no location."""
    boxes = [r.region for r in refs if r.region is not None]
    if not boxes:
        return None
    return NormalizedBox(
        x0=min(b.x0 for b in boxes), y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes), y1=max(b.y1 for b in boxes),
    )


class FakeProvider:
    """Deterministic stand-in. Every pipeline test uses this, never the API."""

    def __init__(
        self,
        assessment: Assessment,
        *,
        model: str = "fake-model",
        prompt_version: str = "1.0.0",
    ) -> None:
        self._assessment = assessment
        self._model = model
        self._prompt_version = prompt_version
        self.calls = 0

    def assess(self, evidence_manifest: dict[str, Any]) -> Assessment:
        self.calls += 1
        return self._assessment

    def signature(self) -> dict[str, Any]:
        return {
            "provider": "fake", "model": self._model,
            "prompt_version": self._prompt_version, "inference_params": {},
        }
