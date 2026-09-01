import itertools

from card_reviewer.review.enums import (
    Authority, Coverage, FindingState, Psa10Candidate, Scale, Verdict,
)
from card_reviewer.review.findings import Finding, FindingProducer
from card_reviewer.review.policies.combine_v1 import decide_verdict, i1_satisfied
from card_reviewer.review.provenance import EvidenceOrigin, EvidenceRef, NormalizedBox


def _f(state=FindingState.OBSERVED, conf=0.95, box=(0.0, 0.0, 0.3, 0.3),
       enhanced=False, relevant=True, producer=FindingProducer.HEURISTIC):
    origin = EvidenceOrigin.ENHANCED if enhanced else EvidenceOrigin.ORIGINAL
    return Finding(
        defect_type="rounding", category="corners", state=state,
        producer=producer, confidence=conf, psa10_relevant=relevant,
        location=NormalizedBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
        evidence=[EvidenceRef(artifact_id="a", image_hash="h", origin=origin,
                              enhancement="clahe:clip=2.0" if enhanced else None,
                              view="front")])


# --- I1 --------------------------------------------------------------------

def test_i1_requires_adequate_detectability_for_the_finding():
    """This prong is what actually stops poor photographs producing
    rejections — the contradiction prong cannot fire on a badly
    photographed card, since nothing there reaches MODERATE."""
    assert i1_satisfied(_f(), Scale.HIGH, []) is True
    assert i1_satisfied(_f(), Scale.LOW, []) is False


def test_i1_requires_the_reject_confidence_floor():
    assert i1_satisfied(_f(conf=0.55), Scale.HIGH, []) is False


def test_i1_fails_on_enhancement_only_evidence():
    assert i1_satisfied(_f(enhanced=True), Scale.HIGH, []) is False


def test_i1_fails_on_a_material_contradiction_at_an_overlapping_location():
    other = _f(state=FindingState.NOT_OBSERVED, box=(0.2, 0.2, 0.5, 0.5))
    assert i1_satisfied(_f(), Scale.HIGH, [(other, Scale.HIGH)]) is False


def test_a_contradiction_elsewhere_on_the_card_is_not_material():
    elsewhere = _f(state=FindingState.NOT_OBSERVED, box=(0.7, 0.7, 0.9, 0.9))
    assert i1_satisfied(_f(), Scale.HIGH, [(elsewhere, Scale.HIGH)]) is True


def test_a_low_detectability_contradiction_does_not_block():
    weak = _f(state=FindingState.NOT_OBSERVED, box=(0.1, 0.1, 0.4, 0.4))
    assert i1_satisfied(_f(), Scale.HIGH, [(weak, Scale.LOW)]) is True


def test_i1_fails_on_a_contradiction_carried_from_fusion():
    """Fusion selects the strongest state, so the contradicting source is no
    longer visible in `others`. The flag is what preserves it."""
    assert i1_satisfied(_f(), Scale.HIGH, [], material_contradiction=True) is False


def test_suspected_findings_never_satisfy_i1():
    assert i1_satisfied(_f(state=FindingState.SUSPECTED), Scale.HIGH, []) is False


def test_cross_producer_disagreement_is_a_material_contradiction():
    vision = _f(state=FindingState.NOT_OBSERVED, box=(0.1, 0.1, 0.4, 0.4),
                producer=FindingProducer.VISION)
    assert i1_satisfied(_f(), Scale.HIGH, [(vision, Scale.HIGH)]) is False


# --- verdict precedence ----------------------------------------------------

def test_an_i1_satisfying_disqualifier_rejects():
    r = decide_verdict([(_f(), Authority.BINDING, Scale.HIGH)],
                       Coverage.SUFFICIENT, ambiguity=False)
    assert r.verdict is Verdict.REJECT


def test_reject_outranks_inadequate_coverage():
    """A crease plainly visible on the front is knowledge, not absence of it.
    A missing back bars passing, not rejecting."""
    r = decide_verdict([(_f(), Authority.BINDING, Scale.HIGH)],
                       Coverage.INADEQUATE, ambiguity=False)
    assert r.verdict is Verdict.REJECT


def test_an_observed_finding_failing_i1_routes_to_review_never_pass():
    """Something looked like a disqualifier and could not be resolved. That
    is an unresolved concern, not an absence of one."""
    r = decide_verdict([(_f(conf=0.5), Authority.BINDING, Scale.HIGH)],
                       Coverage.SUFFICIENT, ambiguity=False)
    assert r.verdict is Verdict.REVIEW


def test_advisory_authority_cannot_reject():
    r = decide_verdict([(_f(), Authority.ADVISORY, Scale.HIGH)],
                       Coverage.SUFFICIENT, ambiguity=False)
    assert r.verdict is Verdict.REVIEW


def test_an_irrelevant_finding_cannot_reject():
    r = decide_verdict([(_f(relevant=False), Authority.BINDING, Scale.HIGH)],
                       Coverage.SUFFICIENT, ambiguity=False)
    assert r.verdict is Verdict.PASS


def test_a_contradicted_finding_does_not_reject_even_at_row_one():
    """REJECT precedence is evaluated first, so the contradiction must be
    visible there — not only in the ambiguity clause at row 3."""
    f = _f()
    r = decide_verdict([(f, Authority.BINDING, Scale.HIGH)], Coverage.SUFFICIENT,
                       ambiguity=False, contradicted={id(f)})
    assert r.verdict is Verdict.REVIEW


def test_inadequate_coverage_without_a_disqualifier_is_insufficient_images():
    r = decide_verdict([], Coverage.INADEQUATE, ambiguity=False)
    assert r.verdict is Verdict.INSUFFICIENT_IMAGES
    assert r.psa10_candidate is Psa10Candidate.UNKNOWN


def test_partial_coverage_reviews():
    assert decide_verdict([], Coverage.PARTIAL, ambiguity=False).verdict is (
        Verdict.REVIEW)


def test_a_clean_fully_covered_card_passes():
    r = decide_verdict([], Coverage.SUFFICIENT, ambiguity=False)
    assert r.verdict is Verdict.PASS
    assert r.psa10_candidate is Psa10Candidate.YES


def test_enhancement_only_evidence_cannot_reject():
    r = decide_verdict([(_f(enhanced=True), Authority.BINDING, Scale.HIGH)],
                       Coverage.SUFFICIENT, ambiguity=False)
    assert r.verdict is Verdict.REVIEW


def test_pass_is_unreachable_without_sufficient_coverage():
    """I2, over every non-sufficient outcome."""
    for coverage in (Coverage.PARTIAL, Coverage.INADEQUATE):
        assert decide_verdict([], coverage, ambiguity=False).verdict is not (
            Verdict.PASS)


def test_the_verdict_function_is_total_over_the_full_cross_product():
    """Coverage x I1-satisfying x I1-failing x ambiguity — every cell maps to
    exactly one verdict, with no hole."""
    axes = itertools.product(list(Coverage), [False, True], [False, True],
                             [False, True])
    for coverage, has_sat, has_unsat, ambiguity in axes:
        findings = []
        if has_sat:
            findings.append((_f(), Authority.BINDING, Scale.HIGH))
        if has_unsat:
            findings.append((_f(conf=0.5), Authority.BINDING, Scale.HIGH))
        r = decide_verdict(findings, coverage, ambiguity=ambiguity)
        assert r.verdict in set(Verdict)
        if has_sat:
            assert r.verdict is Verdict.REJECT
        elif coverage is Coverage.INADEQUATE:
            assert r.verdict is Verdict.INSUFFICIENT_IMAGES
        elif has_unsat or ambiguity or coverage is Coverage.PARTIAL:
            assert r.verdict is Verdict.REVIEW
        else:
            assert r.verdict is Verdict.PASS


def test_psa10_candidate_is_always_derived_from_the_verdict():
    mapping = {Verdict.PASS: Psa10Candidate.YES,
               Verdict.REVIEW: Psa10Candidate.UNCERTAIN,
               Verdict.REJECT: Psa10Candidate.NO,
               Verdict.INSUFFICIENT_IMAGES: Psa10Candidate.UNKNOWN}
    for coverage in Coverage:
        for has_sat in (False, True):
            f = [(_f(), Authority.BINDING, Scale.HIGH)] if has_sat else []
            r = decide_verdict(f, coverage, ambiguity=False)
            assert r.psa10_candidate is mapping[r.verdict]


def test_the_verdict_records_why():
    r = decide_verdict([(_f(), Authority.BINDING, Scale.HIGH)],
                       Coverage.SUFFICIENT, ambiguity=False)
    assert r.reasons and "corners" in r.reasons[0]


def test_heuristic_observed_against_vision_suspected_blocks_a_reject():
    """Isolates the cross-producer clause.

    The NOT_OBSERVED case is already caught by the detectability clause, so
    without this the producer comparison was unguarded. Here CV says the
    defect is established and Claude says only maybe — a genuine
    disagreement about whether it is established at all. CLAUDE.md requires
    preserving that disagreement rather than letting one side silently win,
    and under the recall asymmetry an unresolved one routes to REVIEW.
    """
    vision = _f(state=FindingState.SUSPECTED, box=(0.1, 0.1, 0.4, 0.4),
                producer=FindingProducer.VISION)
    assert i1_satisfied(_f(), Scale.HIGH, [(vision, Scale.HIGH)]) is False


def test_agreement_between_producers_does_not_block_a_reject():
    """The other side of the same clause: corroboration is not contradiction."""
    vision = _f(state=FindingState.OBSERVED, box=(0.1, 0.1, 0.4, 0.4),
                producer=FindingProducer.VISION)
    assert i1_satisfied(_f(), Scale.HIGH, [(vision, Scale.HIGH)]) is True
