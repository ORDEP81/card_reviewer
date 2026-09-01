"""Stage execution with content-addressed caching.

Cache identity is (stage, input_fingerprint, producer_signature). A row
exists ONLY for an output that ran to completion AND passed schema
validation. Failures — exceptions, timeouts, malformed provider responses,
schema violations — are recorded as attempts and can never satisfy a lookup.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .fingerprint import fingerprint, signature_for
from .versions import FUSION_VERSION

from .storage.repository import Repository

__all__ = ["MissingProviderError", "ReviewPipeline", "StageRunner",
           "StageValidationError"]


class StageValidationError(Exception):
    """A stage produced output that does not match its declared schema."""


class ProviderCallFailed(Exception):
    """Anything the provider itself raised.

    The boundary is the VisionProvider protocol: whatever comes out of
    `assess()` is the provider's failure, and everything raised by our own
    code around it is ours. Drawing the line anywhere else lets a bug in the
    canonicalizer report as an outage and silently degrade the review — which
    is exactly how one first showed up here.
    """

class StageRunner:
    def __init__(self, repo: Repository) -> None:
        self._repo = repo

    def run_with_id(
        self,
        stage: str,
        inputs: dict[str, Any],
        versions: dict[str, Any],
        compute: Callable[[], dict[str, Any]],
        *,
        schema: type[BaseModel] | None = None,
        image_hash: str | None = None,
        candidate_id: str | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Run or reuse a stage, returning its output and its row id.

        `review` carries foreign keys to the exact `stage_result` rows that
        produced it, so the id travels with the output rather than being
        re-derived from the cache key afterwards.
        """
        fp = fingerprint(inputs)
        # Raises on an unknown stage or a missing version key: an omitted
        # version would silently make two different implementations look
        # identical, which is worse than a hard failure.
        sig = signature_for(stage, versions)

        cached = self._repo.get_stage_result(stage, fp, sig)
        if cached is not None:
            return cached.output, cached.id

        try:
            output = compute()
        except Exception as exc:
            self._repo.record_attempt(
                stage, fp, sig, error_kind=type(exc).__name__,
                error_detail=str(exc), image_hash=image_hash,
                candidate_id=candidate_id,
            )
            raise

        # "Validated successes only" is enforced here, not assumed. An output
        # that does not match its schema is a failure however cleanly the
        # stage returned it — caching it would poison every later run.
        if schema is not None:
            try:
                schema.model_validate(output)
            except ValidationError as exc:
                self._repo.record_attempt(
                    stage, fp, sig, error_kind="StageValidationError",
                    error_detail=str(exc), image_hash=image_hash,
                    candidate_id=candidate_id,
                )
                raise StageValidationError(
                    f"stage {stage!r} output failed validation: {exc}"
                ) from exc

        row_id = self._repo.put_stage_result(
            stage, fp, sig, output, versions, image_hash=image_hash,
            candidate_id=candidate_id,
        )
        return output, row_id

    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """The output alone, for stages whose row id nothing references."""
        return self.run_with_id(*args, **kwargs)[0]


class MissingProviderError(Exception):
    """Routing wanted a vision call and no provider was configured."""


#: Categories whose defect types are interpretive, so a missing vision layer
#: genuinely leaves them unassessed. Centering and corner rounding are
#: measurements CV establishes on its own, and vetoing all four would drop a
#: perfectly photographed card out of the ranked list entirely.
VISION_DEPENDENT_CATEGORIES = ("surface", "edges")


class ReviewPipeline:
    """One card, start to finish.

    Every stage declared in the cache tables executes through StageRunner
    with its declared fingerprint inputs, producer-signature inputs and
    output schema. A stage invoked directly is not a cached stage, however
    the tables describe it.
    """

    def __init__(
        self,
        repo: Repository,
        store: "ArtifactStore",
        rubric: "Rubric | None" = None,
        *,
        require_provider: bool = False,
    ) -> None:
        from card_reviewer.knowledge import load_active_rubric

        self._repo = repo
        self._store = store
        self._require_provider = require_provider
        self._runner = StageRunner(repo)
        # RubricError aborts the run: there is no verdict without a rubric,
        # and guessing at one is worse than declining.
        self._rubric = rubric or load_active_rubric()

    # -- image tier: cached by image hash, shared across candidates --------

    def _image_tier(self, image_hash: str) -> "ImageStageOutputs":
        from .assembly import ImageStageOutputs
        from .imaging.geometry import GEOMETRY_VERSION, GeometryResult
        from .imaging.geometry import analyze as geometry_analyze
        from .imaging.measure import CV_VERSION, CvMeasurements, measure_all
        from .imaging.observability import OBSERVABILITY_VERSION, ObservabilityResult
        from .imaging.observability import analyze as observability_analyze
        from .imaging.preflight import PREFLIGHT_VERSION, PreflightResult
        from .imaging.preflight import analyze as preflight_analyze
        from .imaging.role_features import ROLE_FEATURES_VERSION, RoleFeatures
        from .imaging.role_features import extract_role_features
        from .taxonomy import TAXONOMY_VERSION

        data = self._store.read(image_hash)
        run = self._runner.run

        pre = run("preflight", {"image_hash": image_hash},
                  {"preflight_version": PREFLIGHT_VERSION, "config": {}},
                  lambda: preflight_analyze(data).model_dump(),
                  schema=PreflightResult, image_hash=image_hash)
        if not pre["usable"]:
            return ImageStageOutputs(image_hash=image_hash, preflight=pre)

        geo = run("geometry", {"image_hash": image_hash, "preflight_output": pre},
                  {"geometry_version": GEOMETRY_VERSION, "config": {}},
                  lambda: geometry_analyze(data, self._store,
                                           image_hash).model_dump(),
                  schema=GeometryResult, image_hash=image_hash)
        geometry = GeometryResult.model_validate(geo)
        if not geometry.usable:
            return ImageStageOutputs(image_hash=image_hash, preflight=pre)

        obs = run("observability",
                  {"image_hash": image_hash, "geometry_output": geo},
                  {"observability_version": OBSERVABILITY_VERSION,
                   "taxonomy_version": TAXONOMY_VERSION, "config": {}},
                  lambda: observability_analyze(geometry, self._store,
                                                image_hash).model_dump(),
                  schema=ObservabilityResult, image_hash=image_hash)

        cv = run("cv_measurements",
                 {"image_hash": image_hash, "geometry_output": geo,
                  "observability_output": obs},
                 {"cv_version": CV_VERSION, "taxonomy_version": TAXONOMY_VERSION,
                  "config": {}},
                 lambda: measure_all(geometry, self._store,
                                     image_hash).model_dump(),
                 schema=CvMeasurements, image_hash=image_hash)

        features = run("role_features",
                       {"image_hash": image_hash, "geometry_output": geo},
                       {"role_features_version": ROLE_FEATURES_VERSION,
                        "config": {}},
                       lambda: extract_role_features(geometry,
                                                     self._store).model_dump(),
                       schema=RoleFeatures, image_hash=image_hash)

        return ImageStageOutputs(image_hash=image_hash, preflight=pre,
                                 geometry=geo, observability=obs,
                                 cv_measurements=cv, role_features=features)

    # -- candidate tier ---------------------------------------------------

    def review(self, candidate, mode, provider=None):
        from .assembly import ASSEMBLY_VERSION, Assembled, assemble, to_image_evidence
        from .enums import Coverage, RuleEvaluability
        from .evaluability import applicable, rule_content, scope_rules
        from .heuristic import HeuristicResult, evaluate
        from .policies.authority_v1 import AUTHORITY_POLICY_VERSION
        from .policies.combine_v1 import (
            COMBINATION_POLICY_VERSION, CombinedResult, combine,
        )
        from .policies.coverage_v1 import (
            COVERAGE_POLICY_VERSION, REQUIRED_FACES, CoverageResult,
            UnevaluableRule, evaluate_coverage,
            unevaluable_fingerprint_content,
        )
        from .policies.relevance_v1 import RELEVANCE_POLICY_VERSION
        from .policies.routing_v1 import (
            ROUTING_POLICY_VERSION, RoutingDecision, decide_routing,
        )
        from .policies.scoring_v1 import SCORING_POLICY_VERSION
        from .role_context import RESOLVER_VERSION, RoleContext, resolve_role_context
        from .taxonomy import TAXONOMY_VERSION
        from .versions import SCORER_VERSION, VOCABULARY_VERSION

        run = self._runner.run
        run_id = self._runner.run_with_id
        cid = candidate.candidate_id

        # routing_decision and review both carry NOT NULL foreign keys to
        # candidate, and candidate_image joins images to it. Nothing else
        # writes these rows, so the pipeline must — and before anything
        # references them.
        self._repo.save_candidate(
            id=cid, source=candidate.source, title=candidate.title,
            supplied_card_type=candidate.card_type,
            supplied_set=candidate.set_name)
        for image in candidate.images:
            self._repo.save_image(image.image_hash,
                                  self._store.path_of(image.image_hash))
            self._repo.link_image(cid, image.image_hash,
                                  supplied_role=image.supplied_role,
                                  ordering=image.ordering)

        images = [self._image_tier(i.image_hash) for i in candidate.images]

        rc = run("role_context",
                 {"image_hashes": [i.image_hash for i in images],
                  "per_image_role_features": [i.role_features for i in images],
                  "listing_title": candidate.title,
                  "card_identification_text": candidate.title,
                  "supplied_card_type": candidate.card_type,
                  "supplied_set": candidate.set_name,
                  "supplied_roles": {i.image_hash: i.supplied_role
                                     for i in candidate.images}},
                 {"resolver_version": RESOLVER_VERSION,
                  "vocabulary_version": VOCABULARY_VERSION},
                 lambda: resolve_role_context(candidate, images).model_dump(),
                 schema=RoleContext, candidate_id=cid)
        role_context = RoleContext.model_validate(rc)
        context = role_context.card_context
        scoped = scope_rules(
            self._rubric.for_card(context.canonical_card_types,
                                  context.canonical_sets), context)
        rules = rule_content(scoped)

        asm = run("evidence_assembly",
                  {"roles": rc["roles"], "context": rc["card_context"],
                   "per_image_outputs": [i.model_dump() for i in images]},
                  {"assembly_version": ASSEMBLY_VERSION},
                  lambda: assemble(to_image_evidence(images),
                                   role_context.roles).model_dump(),
                  schema=Assembled, candidate_id=cid)
        assembled = Assembled.model_validate(asm)

        heur, heur_id = run_id(
            "heuristic",
            {"assembled_evidence": asm, "applicable_rubric_rules": rules},
            {"scorer_version": SCORER_VERSION,
             "taxonomy_version": TAXONOMY_VERSION, "weights": {}},
            lambda: evaluate(assembled, scoped).model_dump(),
            schema=HeuristicResult, candidate_id=cid)
        heuristic = HeuristicResult.model_validate(heur)

        # Rubric gaps arrive as themselves, never simulated by lowering some
        # defect type's pixel detectability.
        unevaluable = [
            UnevaluableRule(rule_id=s.rule.id, category=s.rule.category.value,
                            reason_code=s.reason)
            for s in scoped if s.evaluability is RuleEvaluability.UNEVALUABLE
        ]
        unevaluable_content = unevaluable_fingerprint_content(unevaluable)

        prov, prov_id = run_id(
            "coverage_provisional",
            {"assembled_detectability": asm["detectability_flat"],
             "assembled_reason_codes": asm["reason_codes_flat"],
             "applicable_rubric_rules": rules,
             "unevaluable_rubric_rules": unevaluable_content},
            {"coverage_policy_version": COVERAGE_POLICY_VERSION,
             "taxonomy_version": TAXONOMY_VERSION},
            lambda: evaluate_coverage(
                assembled.detectability, assembled.reason_codes, {},
                assembled.faces, unevaluable_rules=unevaluable).model_dump(),
            schema=CoverageResult, candidate_id=cid)
        provisional = CoverageResult.model_validate(prov)

        rout = run("routing",
                   {"mode": mode.value, "heuristic_output": heur,
                    "provisional_coverage": prov,
                    "assembled_observability": asm["detectability_flat"],
                    "detectability": asm["detectability_flat"]},
                   {"routing_policy_version": ROUTING_POLICY_VERSION},
                   lambda: decide_routing(mode, heuristic.findings,
                                          provisional.outcome,
                                          assembled.detectability).model_dump(),
                   schema=RoutingDecision, candidate_id=cid)
        routing = RoutingDecision.model_validate(rout)
        routing_id = self._repo.save_routing_decision(
            candidate_id=cid, policy_version=ROUTING_POLICY_VERSION,
            mode=mode.value, call_vision=routing.call_vision,
            trigger_reasons=routing.trigger_reasons,
            input_fingerprint=fingerprint({"mode": mode.value,
                                           "heuristic_output": heur}))

        vision, index, vision_id, vision_limit = None, {}, None, None
        vision_signature = None
        if routing.call_vision:
            vision, index, vision_id, vision_limit = self._vision(
                cid, mode, assembled, asm, rout, scoped, provider)
            if vision is not None and provider is not None:
                vision_signature = provider.signature()

        if vision is not None:
            assessability = vision.category_assessability
        elif vision_limit is not None:
            # The vision layer was wanted and did not run. Marked HERE, before
            # coverage — appending a limitation after the verdict would let
            # the card PASS as though it had been fully assessed.
            assessability = {c: False for c in VISION_DEPENDENT_CATEGORIES}
        else:
            assessability = {}  # OFF: nothing expected, nothing missing

        cov, cov_id = run_id(
            "coverage",
            {"assembled_detectability": asm["detectability_flat"],
             "vision_category_assessability": assessability,
             "assembled_reason_codes": asm["reason_codes_flat"],
             "applicable_rubric_rules": rules,
             "unevaluable_rubric_rules": unevaluable_content},
            {"coverage_policy_version": COVERAGE_POLICY_VERSION,
             "taxonomy_version": TAXONOMY_VERSION},
            lambda: evaluate_coverage(
                assembled.detectability, assembled.reason_codes, assessability,
                assembled.faces, unevaluable_rules=unevaluable).model_dump(),
            schema=CoverageResult, candidate_id=cid)
        coverage = CoverageResult.model_validate(cov)

        missing_face = any(f not in assembled.faces for f in REQUIRED_FACES)
        comb, comb_id = run_id(
            "combine",
            {"heuristic_output": heur,
             "vision_output": vision.model_dump() if vision else None,
             "coverage_output": cov, "applicable_rubric_rule_content": rules,
             "detectability": asm["detectability_flat"],
             "card_context_known": context.is_known,
             "required_face_missing": missing_face,
             "manifest_index": {k: v.model_dump() for k, v in index.items()}},
            {"combination_policy_version": COMBINATION_POLICY_VERSION,
             "scoring_policy_version": SCORING_POLICY_VERSION,
             "relevance_policy_version": RELEVANCE_POLICY_VERSION,
             "authority_policy_version": AUTHORITY_POLICY_VERSION,
             "fusion_version": FUSION_VERSION,
             "taxonomy_version": TAXONOMY_VERSION},
            lambda: combine(
                heuristic, vision, coverage,
                card_context_known=context.is_known, scoped_rules=scoped,
                manifest_index=index, detectability=assembled.detectability,
                required_face_missing=missing_face).model_dump(),
            schema=CombinedResult, candidate_id=cid)

        return self._persist(
            candidate, mode, routing_id, CombinedResult.model_validate(comb),
            coverage, role_context, images,
            {"heuristic": heur_id, "coverage_provisional": prov_id,
             "coverage": cov_id, "combine": comb_id},
            vision_id, vision_limit, vision_signature)

    def _vision(self, cid, mode, assembled, assembled_json, routing_json,
                scoped, provider):
        """Cache lookup BEFORE any call.

        A provider invoked before the lookup bills every re-review of an
        unchanged card.
        """
        from .evaluability import applicable, rule_content
        from .manifest import BUDGETS, MANIFEST_BUILDER_VERSION, BuiltManifest
        from .manifest import build_manifest
        from .vision.provider import Assessment, ProviderContractError

        man = self._runner.run(
            "manifest",
            {"mode_budget": BUDGETS[mode], "assembled_evidence": assembled_json,
             "routing_decision": routing_json,
             "applicable_rubric_rule_content": rule_content(scoped)},
            {"manifest_builder_version": MANIFEST_BUILDER_VERSION},
            lambda: build_manifest(assembled, mode,
                                   applicable(scoped)).model_dump(),
            schema=BuiltManifest, candidate_id=cid)
        built = BuiltManifest.model_validate(man)

        if provider is None:
            if self._require_provider:
                raise MissingProviderError(
                    "routing requested a vision call and no provider is "
                    "configured")
            # Not configured. This must not silently behave as OFF: the
            # categories the provider would have judged are marked unassessed
            # before coverage runs, so the card cannot reach PASS.
            return None, built.index, None, "VISION_UNAVAILABLE"

        def call_provider() -> dict:
            try:
                return provider.assess(built.payload).model_dump()
            except Exception as exc:
                raise ProviderCallFailed(str(exc)) from exc

        try:
            output, vision_id = self._runner.run_with_id(
                "vision", {"provider_evidence_payload": built.payload},
                provider.signature(), call_provider,
                schema=Assessment, candidate_id=cid)
        except (ProviderCallFailed, StageValidationError):
            # Narrow deliberately. Catching every exception would report a bug
            # in our own code as a provider outage and silently degrade the
            # review — which is how a ValueError from the canonicalizer first
            # showed up here as "the provider is down".
            #
            # StageRunner already recorded the attempt. A failure REMOVES
            # evidence; it never creates any, and an independently
            # established defect may still reject.
            return None, built.index, None, "VISION_FAILED"

        return Assessment.model_validate(output), built.index, vision_id, None

    def _persist(self, candidate, mode, routing_id, combined, coverage,
                 role_context, images, stage_ids, vision_id, vision_limit,
                 vision_signature):
        from .models import CardReview
        from .versions import effective_versions

        limitations = [l.model_dump() for l in coverage.limitations]
        if vision_limit:
            # Recorded as a limitation so the owner sees WHY the card was not
            # fully assessed; the coverage veto above is what actually stops
            # it passing.
            limitations.append({
                "face": "card", "category": "*", "defect_type": "*",
                "reason_code": vision_limit,
                "undetectability_class": "circumstantial"})

        review = CardReview(
            candidate_id=candidate.candidate_id, title=candidate.title,
            mode=mode.value, verdict=combined.verdict.value,
            psa10_candidate=combined.psa10_candidate.value,
            psa10_rank_score=combined.psa10_rank_score,
            rankable=combined.rankable,
            estimated_psa_grade=combined.estimated_psa_grade,
            review_confidence=combined.review_confidence.value,
            coverage=coverage.outcome.value,
            coverage_detail=coverage.assessed,
            roles_and_context=role_context.model_dump(mode="json"),
            # The ADJUDICATED view — after fusion and I3 — because this is
            # what the report renders and what the verdict was actually
            # decided on. Showing the raw producer state made the report
            # assert `observed` for a finding the engine had demoted.
            defects_found=[f.as_finding().model_dump(mode="json")
                           for f in combined.fused],
            # ...and the raw producer findings stay recoverable beside it.
            # Calibrating OpenCV against Claude against the PSA outcome needs
            # both sides intact, so fusion must never be the only record.
            raw_findings=[f.model_dump(mode="json") for f in combined.findings],
            limitations=limitations,
            recommended_additional_photos=coverage.recommended_additional_photos,
            card_identification_request=coverage.card_identification_request,
            image_quality={i.image_hash: i.preflight for i in images},
            # Kept separately recoverable: measurement and interpretation must
            # never be collapsed into one another (calibration depends on it).
            cv_assessment={i.image_hash: i.cv_measurements for i in images
                           if i.cv_measurements},
            vision_assessment=(
                {"fused": [f.model_dump(mode="json") for f in combined.fused]}
                if combined.vision_present else None),
            reasoning="; ".join(combined.reasons),
            versions=effective_versions(vision_signature=vision_signature),
        )

        review.review_id = self._repo.save_review(
            candidate_id=candidate.candidate_id, mode=mode.value,
            routing_decision_id=routing_id, verdict=review.verdict,
            psa10_candidate=review.psa10_candidate,
            psa10_rank_score=review.psa10_rank_score,
            rankable=review.rankable,
            estimated_psa_grade=review.estimated_psa_grade,
            review_confidence=review.review_confidence,
            coverage=review.coverage,
            heuristic_result_id=stage_ids["heuristic"],
            coverage_provisional_result_id=stage_ids["coverage_provisional"],
            coverage_result_id=stage_ids["coverage"],
            combine_result_id=stage_ids["combine"],
            vision_result_id=vision_id,
            rubric_version=self._rubric.version,
            output=review.model_dump(mode="json"))
        return review
