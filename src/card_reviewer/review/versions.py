"""Every declared version, in one place.

Each is re-exported by the module that owns the behaviour, so a stage's
producer signature and this table can never disagree.
"""

from __future__ import annotations

import json

# 1.1.0: the same — the module drifted from the constant declared before
# it existed.
PREFLIGHT_VERSION = "1.1.0"
# 1.1.0: GeometryResult gained boundary_observed, which changes whether the
# border-relative producers run at all. Cached rows from 1.0.0 cannot answer
# the question, so they must be recomputed rather than defaulted.
# 1.2.0: _detect_quad returns `observed` itself, so a real boundary whose
# rectangularity is exactly ASSUMED_BOUNDARY_CONFIDENCE flips False -> True.
GEOMETRY_VERSION = "1.2.0"
# 1.1.0: each edge region gained a real band (they all read the card's
# centre), and an assumed boundary now records BOUNDARY_NOT_OBSERVED. Its
# fingerprint contains geometry's OUTPUT, which also moved — but that is
# luck, not a guarantee, and a stage whose behaviour changed bumps.
OBSERVABILITY_VERSION = "1.1.0"
# 1.1.0: corners and edges now consult boundary_observed before concluding.
CV_VERSION = "1.1.0"
ROLE_FEATURES_VERSION = "1.0.0"
RESOLVER_VERSION = "1.0.0"
VOCABULARY_VERSION = "1.0.0"
# 1.1.0: best_for["centering"] now names the image the measurement was
# taken from rather than fronts[0], and an unknown-role image contributes
# its evidence refs.
ASSEMBLY_VERSION = "1.1.0"
# 1.2.0: heuristic.py changed again after the 1.1.0 bump — the cross-face
# evidence fallback removed and centering refs narrowed to one image.
SCORER_VERSION = "1.2.0"
AUTHORITY_POLICY_VERSION = "1.0.0"
RELEVANCE_POLICY_VERSION = "1.0.0"
# 1.1.0: a category with a structural waiver is no longer penalized twice.
COVERAGE_POLICY_VERSION = "1.1.0"
# 1.1.0: the policy changed after this constant was first written and was
# never moved with it.
ROUTING_POLICY_VERSION = "1.1.0"
# 1.1.0: artifacts backing anomaly candidates outrank generic views, so
# the provider is never told about an anomaly whose picture it was not sent.
# 1.2.0: whole-card overviews are pinned ahead of those crops, and an
# anomaly whose crop did not fit stops claiming one.
# 1.3.0: pinned overviews capped at two TOTAL rather than one per
# photograph, and crops cover each region before repeating one.
# 1.4.0: the two pinned overviews are one per FACE where both exist —
# capping at two without knowing the faces sent both views of one face.
# 1.5.0: an unresolved photograph no longer takes a pinned slot from an
# identified face — UNKNOWN is not a third face.
# 1.6.0: VIEW_PRIORITY dropped `front_face`/`back_face`, which no producer
# ever emitted, so OVERVIEW_TIERS is 1 rather than a tier count nothing
# reached.
MANIFEST_BUILDER_VERSION = "1.6.0"
# 1.1.0: _material_contradiction compares category AND defect_type, and
# the policy grew several arms. The signature was byte-identical across
# both changes, so cached combine rows kept the older adjudication.
COMBINATION_POLICY_VERSION = "1.1.0"
# 1.1.0: estimated_grade returns "9-10" where it returned "10" for a card
# with open questions — a different answer for the same evidence.
SCORING_POLICY_VERSION = "1.1.0"
# 1.1.0: correlation needs positive evidence of a shared face, and a group
# is a clique rather than everything matching its first member.
# 1.2.0: two NAMED regions are two places. This changed combine's output on
# 62 of 117 corpus photographs and the constant did not move with it, so a
# cached row kept reporting a minor top edge as severe.
FUSION_VERSION = "1.2.0"
# 1.1.0: corners and edges reclassified MEASUREMENT -> INTERPRETIVE, and
# defect types and reason codes were added. It sits in the signature of
# observability, cv_measurements, heuristic, both coverage stages and
# combine, so every one of them was serving rows built under the old
# promotion rules.
# 1.2.0: VISION_UNAVAILABLE and VISION_FAILED are declared codes rather
# than strings invented at the call site with a guessed class.
TAXONOMY_VERSION = "1.2.0"
# 1.1.0: as above. Largely self-invalidating, since canonicalization
# prefixes the fingerprint, but the rule does not have exceptions.
CANON_SCHEME_VERSION = "1.1.0"
#: The derived-artifact-id scheme. Not a stage version: changing it re-keys
#: every stage that reads an artifact id, so it is a migration rather than a
#: bump. It exists so that change cannot happen unnoticed.
ARTIFACT_SCHEME_VERSION = "1.0.0"

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
    # Stamped so a stored review says which id scheme produced its artifact
    # references. A scheme change re-keys every stage that reads one, and a
    # historical record that cannot name its scheme cannot be compared
    # across the migration.
    "artifact_scheme": ARTIFACT_SCHEME_VERSION,
}


#: What `VERSIONS["vision"]` holds statically. It is not a version — the
#: vision stage's identity is supplied by the provider at run time — so it
#: must never reach a stamped review.
VISION_PLACEHOLDER = "provider-supplied"

#: Recorded when routing decided not to call, or no provider was available.
#: Explicit, because "vision did not run" and "vision ran with some unknown
#: model" are different facts and calibration has to tell them apart.
VISION_NOT_RUN = "not_run"

#: The five values that identify a vision run (spec §4).
VISION_SIGNATURE_KEYS = ("provider", "model", "prompt_version",
                         "adapter_version", "inference_params")


def format_vision_version(signature: dict[str, object]) -> str:
    """Render a provider signature as the version string stamped on a review."""
    missing = [k for k in VISION_SIGNATURE_KEYS if k not in signature]
    if missing:
        raise KeyError(
            f"vision signature is missing {missing} — a run stamped without "
            "its provider, model, prompt version, adapter version and "
            "inference parameters cannot be compared against the PSA outcome "
            "it predicted"
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
    # The adapter is rendered, not merely required. Demanding the key and
    # dropping it made the stamp claim a provenance it did not carry: two
    # adapters parsing the same response read as one run in the calibration
    # record, which is the ground truth a later PSA outcome is compared to.
    return (
        f"{signature['provider']}/{signature['model']}"
        f"@{signature['prompt_version']}"
        f"+{signature['adapter_version']}[{rendered}]"
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
    # The rubric is read at RUN TIME, not declared as a constant beside the
    # others: it is whatever Subsystem B currently publishes, and a copy
    # here would go stale silently. Rule 7 names it alongside the model and
    # analyzer versions, and it was the one version a verdict depends on
    # that a stored review could not name.
    from card_reviewer.knowledge import load_active_rubric

    stamped["rubric"] = load_active_rubric().version
    stamped["vision"] = (
        VISION_NOT_RUN
        if vision_signature is None
        else format_vision_version(vision_signature)
    )
    return stamped
