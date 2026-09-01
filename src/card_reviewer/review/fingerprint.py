"""Cache identity: (stage, input_fingerprint, producer_signature).

The rule that makes reuse safe: a downstream stage fingerprints upstream
output VALUES, never upstream producer signatures. Bumping the CV analyzer
creates a new cv_measurements row, but if the measurements a later stage
received are unchanged, that stage's fingerprint is unchanged and its
stored result — including an expensive vision assessment — is reused.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .canonical import (
    CANON_SCHEME_VERSION,
    SIGNATURE_SCHEME_VERSION,
    canonicalize,
    canonicalize_config,
)

__all__ = [
    "STAGE_FINGERPRINT_INPUTS",
    "STAGE_SIGNATURE_INPUTS",
    "fingerprint",
    "signature_for",
]

# Which version keys form each stage's producer signature — the stage's own
# implementation and configuration, which can change its output for identical
# input. Read together with spec §4; this is the executable form of that table.
STAGE_SIGNATURE_INPUTS: dict[str, tuple[str, ...]] = {
    "preflight": ("preflight_version", "config"),
    "geometry": ("geometry_version", "config"),
    # Taxonomy, not rubric: adding a defect type changes what pixels must be
    # measured; changing a rubric rule changes what the measurement means.
    "observability": ("observability_version", "taxonomy_version", "config"),
    "cv_measurements": ("cv_version", "taxonomy_version", "config"),
    "role_features": ("role_features_version", "config"),
    "role_context": ("resolver_version", "vocabulary_version"),
    "evidence_assembly": ("assembly_version",),
    # Taxonomy, not authority: the heuristic asks `promotion_of` whether a
    # defect type may reach OBSERVED. Authority is read by relevance and
    # scoring, which run later — declaring it here would invalidate stored
    # heuristic results on a policy bump this stage never reads.
    "heuristic": ("scorer_version", "taxonomy_version", "weights"),
    "coverage_provisional": ("coverage_policy_version", "taxonomy_version"),
    # Mode is deliberately absent here: it is data the stage consumes, not
    # part of its implementation identity, so it belongs in the FINGERPRINT.
    "routing": ("routing_policy_version",),
    "manifest": ("manifest_builder_version",),
    "vision": ("provider", "model", "prompt_version", "inference_params"),
    "coverage": ("coverage_policy_version", "taxonomy_version"),
    # Combine runs relevance, fusion and scoring inside itself, so every one
    # of those policies can change its output for identical inputs. Taxonomy
    # is included because relevance decides psa10_relevant from CATEGORIES
    # and fusion correlates on declared defect types.
    "combine": (
        "combination_policy_version",
        "scoring_policy_version",
        "relevance_policy_version",
        "authority_policy_version",
        "fusion_version",
        "taxonomy_version",
    ),
}

# Which data each stage consumes. Distinct from the signature: this is the
# input, that is the implementation.
STAGE_FINGERPRINT_INPUTS: dict[str, tuple[str, ...]] = {
    "preflight": ("image_hash",),
    "geometry": ("image_hash", "preflight_output"),
    "observability": ("image_hash", "geometry_output"),
    "cv_measurements": ("image_hash", "geometry_output", "observability_output"),
    "role_features": ("image_hash", "geometry_output"),
    "role_context": (
        "image_hashes",
        "per_image_role_features",
        "listing_title",
        "card_identification_text",
        "supplied_card_type",
        "supplied_set",
        "supplied_roles",
    ),
    "evidence_assembly": ("roles", "context", "per_image_outputs"),
    "heuristic": ("assembled_evidence", "applicable_rubric_rules"),
    "coverage_provisional": ("assembled_detectability", "applicable_rubric_rules"),
    "routing": (
        "mode",
        "heuristic_output",
        "provisional_coverage",
        "assembled_observability",
        "detectability",
    ),
    "manifest": (
        "mode_budget",
        "assembled_evidence",
        "routing_decision",
        "applicable_rubric_rule_content",
    ),
    # ONLY what the provider actually consumes. The builder's own version
    # lives in the MANIFEST stage's signature, never here: a builder bump
    # producing identical provider-visible content must not re-bill a call.
    "vision": ("provider_evidence_payload",),
    "coverage": (
        "assembled_detectability",
        "vision_category_assessability",
        "applicable_rubric_rules",
    ),
    # Everything `combine` reads. The three stage outputs are the obvious
    # part; the rest arrive as separate arguments and are just as capable of
    # changing the verdict, so they belong in the cache key too:
    #   - rule content drives relevance and therefore authority
    #   - detectability drives I1's adequacy prong
    #   - card_context_known and required_face_missing drive review_confidence
    #   - manifest_index resolves provider citations back to provenance, so a
    #     different index can change whether a finding satisfies I3
    "combine": (
        "heuristic_output",
        "vision_output",
        "coverage_output",
        "applicable_rubric_rule_content",
        "detectability",
        "card_context_known",
        "required_face_missing",
        "manifest_index",
    ),
}


def fingerprint(payload: Any) -> str:
    """Cache identity of a stage's INPUTS — evidence, quantized semantically."""
    body = f"{CANON_SCHEME_VERSION}|{canonicalize(payload)}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def signature_for(stage: str, versions: dict[str, Any]) -> str:
    try:
        keys = STAGE_SIGNATURE_INPUTS[stage]
    except KeyError as exc:
        raise KeyError(
            f"unknown stage {stage!r} — every cached stage must declare its "
            "producer signature inputs"
        ) from exc
    missing = [k for k in keys if k not in versions]
    if missing:
        raise KeyError(
            f"stage {stage!r} signature requires {missing} — an omitted version "
            "key would silently make two different implementations look identical"
        )
    # Producer configuration is rendered EXACTLY, never through the
    # measurement quantizer: `temperature=0.2` and `temperature=0.204` are
    # different behaviour, and a provider signature carrying a float must not
    # need a declared measurement precision to be hashable at all.
    body = (
        f"{SIGNATURE_SCHEME_VERSION}|"
        f"{canonicalize_config({'stage': stage, **{k: versions[k] for k in keys}})}"
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()
