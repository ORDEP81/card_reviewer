"""Every declared version, in one place.

Each is re-exported by the module that owns the behaviour, so a stage's
producer signature and this table can never disagree.
"""

from __future__ import annotations

import json

PREFLIGHT_VERSION = "1.0.0"
GEOMETRY_VERSION = "1.0.0"
OBSERVABILITY_VERSION = "1.0.0"
CV_VERSION = "1.0.0"
ROLE_FEATURES_VERSION = "1.0.0"
RESOLVER_VERSION = "1.0.0"
VOCABULARY_VERSION = "1.0.0"
ASSEMBLY_VERSION = "1.0.0"
SCORER_VERSION = "1.0.0"
AUTHORITY_POLICY_VERSION = "1.0.0"
RELEVANCE_POLICY_VERSION = "1.0.0"
COVERAGE_POLICY_VERSION = "1.0.0"
ROUTING_POLICY_VERSION = "1.0.0"
MANIFEST_BUILDER_VERSION = "1.0.0"
COMBINATION_POLICY_VERSION = "1.0.0"
SCORING_POLICY_VERSION = "1.0.0"
FUSION_VERSION = "1.0.0"
TAXONOMY_VERSION = "1.0.0"
CANON_SCHEME_VERSION = "1.0.0"

#: Stamped onto every CardReview (spec §16). Keyed by STAGE, so it can be
#: compared directly against STAGE_SIGNATURE_INPUTS — a component-keyed map
#: could not be, and the drift would be invisible.
VERSIONS: dict[str, str] = {
    "preflight": PREFLIGHT_VERSION,
    "geometry": GEOMETRY_VERSION,
    "observability": OBSERVABILITY_VERSION,
    "cv_measurements": CV_VERSION,
    "role_features": ROLE_FEATURES_VERSION,
    "role_context": RESOLVER_VERSION,
    "evidence_assembly": ASSEMBLY_VERSION,
    "heuristic": SCORER_VERSION,
    "coverage_provisional": COVERAGE_POLICY_VERSION,
    "routing": ROUTING_POLICY_VERSION,
    "manifest": MANIFEST_BUILDER_VERSION,
    "vision": "provider-supplied",  # comes from VisionProvider.signature()
    "coverage": COVERAGE_POLICY_VERSION,
    "combine": COMBINATION_POLICY_VERSION,
}

#: Cross-cutting versions that are not themselves stages.
SUPPORTING_VERSIONS: dict[str, str] = {
    "taxonomy": TAXONOMY_VERSION,
    "vocabulary": VOCABULARY_VERSION,
    "authority": AUTHORITY_POLICY_VERSION,
    "relevance": RELEVANCE_POLICY_VERSION,
    "scoring": SCORING_POLICY_VERSION,
    "fusion": FUSION_VERSION,
    "canonicalization": CANON_SCHEME_VERSION,
}


#: What `VERSIONS["vision"]` holds statically. It is not a version — the
#: vision stage's identity is supplied by the provider at run time — so it
#: must never reach a stamped review.
VISION_PLACEHOLDER = "provider-supplied"

#: Recorded when routing decided not to call, or no provider was available.
#: Explicit, because "vision did not run" and "vision ran with some unknown
#: model" are different facts and calibration has to tell them apart.
VISION_NOT_RUN = "not_run"

#: The four values that identify a vision run (spec §4).
VISION_SIGNATURE_KEYS = ("provider", "model", "prompt_version", "inference_params")


def format_vision_version(signature: dict[str, object]) -> str:
    """Render a provider signature as the version string stamped on a review."""
    missing = [k for k in VISION_SIGNATURE_KEYS if k not in signature]
    if missing:
        raise KeyError(
            f"vision signature is missing {missing} — a run stamped without its "
            "provider, model, prompt version and inference parameters cannot be "
            "compared against the PSA outcome it predicted"
        )
    # Render each value as canonical JSON rather than with str(), so a nested
    # parameter dict produces one stable string whatever order it was built
    # in — otherwise the same run reads as two different ones in the
    # calibration record.
    params = signature["inference_params"] or {}
    rendered = ",".join(
        f"{k}={json.dumps(v, sort_keys=True, separators=(',', ':'))}"
        for k, v in sorted(dict(params).items())
    )
    return (
        f"{signature['provider']}/{signature['model']}"
        f"@{signature['prompt_version']}[{rendered}]"
    )


def effective_versions(
    *, vision_signature: dict[str, object] | None = None
) -> dict[str, str]:
    """The versions that ACTUALLY ran, for stamping onto a CardReview.

    `VERSIONS` is a static declaration and cannot describe the vision stage,
    whose identity comes from the provider at run time. Writing it verbatim
    would stamp every review with a placeholder that names nothing — so the
    review carries this map instead, with vision resolved to either the real
    provider identity or an explicit "did not run".
    """
    stamped = dict(VERSIONS)
    stamped.update(SUPPORTING_VERSIONS)
    stamped["vision"] = (
        VISION_NOT_RUN
        if vision_signature is None
        else format_vision_version(vision_signature)
    )
    return stamped
