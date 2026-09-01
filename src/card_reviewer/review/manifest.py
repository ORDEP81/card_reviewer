"""Deterministic evidence manifest construction (spec §12).

Its own cached stage, not an unnamed step: it is what the `vision` stage
fingerprints, so it must be reproducible independently of whether a call was
ultimately made.

The boundary that matters most here: `payload` is EXACTLY what the provider
consumes, and `builder_meta` holds our own bookkeeping. A builder version
inside the payload would enter the vision fingerprint and re-bill every card
on a bump the provider cannot see.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .enums import Mode
from .provenance import EvidenceRef
from .versions import MANIFEST_BUILDER_VERSION

__all__ = ["BUDGETS", "MANIFEST_BUILDER_VERSION", "BuiltManifest", "build_manifest"]

#: DEEP means maximum USEFUL evidence, not mechanically every artifact.
BUDGETS: dict[Mode, int] = {Mode.OFF: 0, Mode.SMART: 8, Mode.DEEP: 20}

#: Fixed priority so selection is deterministic rather than "whatever fits".
VIEW_PRIORITY = ("surface_original", "front_face", "back_face",
                 "corner_", "edge_", "surface_")


class BuiltManifest(BaseModel):
    """The `manifest` stage's cached output.

    A Pydantic model rather than a tuple, because StageRunner validates it
    and SQLite stores it as JSON — EvidenceRef is itself a model, so the
    index serializes cleanly and provider citations stay resolvable after a
    restart.
    """

    payload: dict[str, Any] = Field(default_factory=dict)
    index: dict[str, EvidenceRef] = Field(default_factory=dict)
    builder_meta: dict[str, Any] = Field(default_factory=dict)


def _rank(view: str) -> int:
    for index, prefix in enumerate(VIEW_PRIORITY):
        if view.startswith(prefix):
            return index
    return len(VIEW_PRIORITY)


def build_manifest(assembled: Any, mode: Mode, rubric_rules: list) -> BuiltManifest:
    seen: set[str] = set()
    candidates: list[EvidenceRef] = []
    for refs in assembled.evidence_refs.values():
        for ref in refs:
            if ref.artifact_id in seen:
                continue
            seen.add(ref.artifact_id)
            candidates.append(ref)

    candidates.sort(key=lambda r: (_rank(r.view), r.view, r.artifact_id))
    selected = candidates[: BUDGETS[mode]]

    payload = {
        "artifacts": [
            {
                "artifact_id": r.artifact_id, "view": r.view,
                # The provider must be able to tell an enhanced view from an
                # original, or it cannot honour the conservative evidence
                # standard the brief asks of it.
                "origin": r.origin.value, "enhancement": r.enhancement,
                "region": r.region.model_dump() if r.region else None,
            }
            for r in selected
        ],
        # Nested under its own name: flattening it to bare horizontal/vertical
        # loses the semantic path the canonicalizer resolves precision by, and
        # tells the provider less about what the numbers are.
        "measurements": {"centering": dict(assembled.centering)},
        # Detectability and its reason codes: without them the provider reads
        # absence of a defect as absence of the defect.
        "detectability": {f"{k}": str(v) for k, v in assembled.detectability.items()},
        "detectability_reasons": {
            f"{k}": v for k, v in assembled.reason_codes.items()
        },
        "image_limitations": list(assembled.limitations),
        # Disagreements between photographs, preserved not averaged.
        "conflicts": list(assembled.conflicts),
        # Anomaly candidates WITH their enhancement provenance, so the
        # provider can tell "visible in the original" from "only under CLAHE".
        "anomaly_candidates": [
            {
                "category": a.get("category"), "defect_type": a.get("defect_type"),
                "region": a.get("region"), "artifact_id": a.get("artifact_id"),
                "surfaced_by": a.get("surfaced_by", "original"),
                "visible_in_original": a.get("visible_in_original", True),
            }
            for a in assembled.anomalies
        ],
        # Content, not a version string. No pricing field ever appears here
        # (non-negotiable rule 10).
        "rubric_rules": [
            {
                "id": r.id, "category": r.category.value,
                "statement": r.statement,
                "evidence_type": r.evidence_type.value,
                "confidence": r.confidence.value,
            }
            for r in rubric_rules
        ],
    }

    return BuiltManifest(
        payload=payload,
        index={r.artifact_id: r for r in selected},
        builder_meta={
            "builder_version": MANIFEST_BUILDER_VERSION,
            "mode": mode.value, "selected": len(selected),
        },
    )
