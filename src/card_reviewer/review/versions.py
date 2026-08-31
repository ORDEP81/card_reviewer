"""Every declared version, in one place.

Each is re-exported by the module that owns the behaviour, so a stage's
producer signature and this table can never disagree.
"""

from __future__ import annotations

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
