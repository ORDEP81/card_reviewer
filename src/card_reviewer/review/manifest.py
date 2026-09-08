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

#: How many entries of VIEW_PRIORITY are whole-card OVERVIEWS. They are
#: pinned ahead of anomaly crops: a card whose every region raised a
#: candidate filled the entire SMART budget with crops and sent no view of
#: the card at all, which left the provider unable to answer for surface —
#: the category with the fewest crops and the most to read from the whole.
OVERVIEW_TIERS = 3

#: How many whole-card views are worth pinning, in total — NOT per
#: photograph. Overviews are emitted per image, so pinning them by tier
#: alone meant a six-photograph listing spent six of SMART's eight slots on
#: six copies of the same view and sent one distinct crop. Two is a front
#: and a back: enough for the provider to see the card, cheap enough to
#: leave the budget to the evidence that differs between regions.
#:
#: Two is only "a front and a back" if the choice knows the faces. Capping
#: at two without that sent BOTH views of one face on an ordinary
#: two-front-two-back listing, leaving the other face — possibly the front,
#: which decides PSA 10 — with no whole-card view at all.
MAX_PINNED_OVERVIEWS = 2


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


def _anomaly_artifacts(assembled: Any) -> set[str]:
    """The artifacts anomaly candidates actually point at.

    These outrank generic views. Ranking by view name alone let an
    anomaly's own artifact fall outside the budget while the payload still
    cited it, so the provider was told "a candidate at artifact X" and
    never given X — a dangling id, and the opposite of sending the relevant
    anomaly views. `corner_` sorts before `edge_` and then alphabetically,
    so an anomaly on a later corner or any edge was the ordinary case.
    """
    return {a.get("artifact_id") for a in getattr(assembled, "anomalies", [])
            if a.get("artifact_id")}


def build_manifest(assembled: Any, mode: Mode, rubric_rules: list,
                   image_roles: dict | None = None) -> BuiltManifest:
    """`image_roles` maps image_hash to its resolved face.

    `EvidenceRef` carries no role, so without this the pin cannot tell one
    face's overview from another's and both pinned views can come from the
    same face.
    """
    seen: set[str] = set()
    candidates: list[EvidenceRef] = []
    for refs in assembled.evidence_refs.values():
        for ref in refs:
            if ref.artifact_id in seen:
                continue
            seen.add(ref.artifact_id)
            candidates.append(ref)

    # Anomaly-backing artifacts first, then the fixed view priority. Still
    # a total order over the same candidates, so selection stays
    # deterministic and the budget is unchanged — what moves is which
    # evidence is worth the room.
    backing = _anomaly_artifacts(assembled)

    def is_overview(ref: EvidenceRef) -> bool:
        return _rank(ref.view) < OVERVIEW_TIERS

    # Which overviews get pinned, decided before the sort so the rest fall
    # back into ordinary competition rather than being dropped. One per
    # photograph at most, so two pinned views are two different photographs.
    def face_of(ref: EvidenceRef):
        role = (image_roles or {}).get(ref.image_hash)
        return getattr(role, "value", role)

    ordered = sorted(candidates,
                     key=lambda r: (_rank(r.view), r.view, r.artifact_id))
    pinned: list[str] = []
    seen_faces: set[Any] = set()
    seen_images: set[str] = set()
    # One per FACE first, so both faces are represented before either gets a
    # second view.
    for ref in ordered:
        if len(pinned) >= MAX_PINNED_OVERVIEWS:
            break
        face = face_of(ref)
        if is_overview(ref) and face is not None and face not in seen_faces:
            seen_faces.add(face)
            seen_images.add(ref.image_hash)
            pinned.append(ref.artifact_id)
    # Then fill any remaining slot from a photograph not already pinned —
    # a front-only listing has one face, and two views of it still beat one.
    for ref in ordered:
        if len(pinned) >= MAX_PINNED_OVERVIEWS:
            break
        if is_overview(ref) and ref.image_hash not in seen_images:
            seen_images.add(ref.image_hash)
            pinned.append(ref.artifact_id)

    # How many times this view has already appeared ahead of a candidate.
    # Sorting on it puts the FIRST of every view before the second of any,
    # so the budget covers the regions before it repeats one. Without it a
    # six-photograph listing spent every crop slot on `corner_bottom_left`:
    # the same corner six times, and no view of the other three, on a card
    # whose provider is asked to assess all four.
    nth: dict[str, int] = {}
    occurrence: dict[str, int] = {}
    for ref in sorted(candidates, key=lambda r: (_rank(r.view), r.view,
                                                 r.artifact_id)):
        occurrence[ref.artifact_id] = nth.get(ref.view, 0)
        nth[ref.view] = nth.get(ref.view, 0) + 1

    candidates.sort(key=lambda r: (r.artifact_id not in pinned,
                                   r.artifact_id not in backing,
                                   occurrence[r.artifact_id],
                                   _rank(r.view), r.view, r.artifact_id))
    selected = candidates[: BUDGETS[mode]]
    sent = {r.artifact_id for r in selected}

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
        # The stable flat form, NOT f-string of the tuple. That produced
        # "(<ImageRole.FRONT: 'front'>, 'corners', 'whitening')" — a CPython
        # enum repr, which is not a declared part of this system's cache
        # identity and has changed between releases, so a Python upgrade
        # would silently re-bill every card. It is also unreadable for the
        # provider that has to act on it.
        "detectability": dict(assembled.detectability_flat),
        "detectability_reasons": dict(assembled.reason_codes_flat),
        "image_limitations": list(assembled.limitations),
        # Disagreements between photographs, preserved not averaged.
        "conflicts": list(assembled.conflicts),
        # Anomaly candidates WITH their enhancement provenance, so the
        # provider can tell "visible in the original" from "only under CLAHE".
        "anomaly_candidates": [
            {
                "category": a.get("category"), "defect_type": a.get("defect_type"),
                "region": a.get("region"),
                # Only when the block was actually sent. An id here names
                # something the provider received; citing one it did not
                # get is a dangling reference, and the ids are required to
                # map deterministically onto the blocks in the payload.
                #
                # The candidate itself stays either way — its category,
                # region and provenance are real information, and deleting
                # it because the budget was tight would hide a limitation
                # rather than report it.
                "artifact_id": (a.get("artifact_id")
                                if a.get("artifact_id") in sent else None),
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
