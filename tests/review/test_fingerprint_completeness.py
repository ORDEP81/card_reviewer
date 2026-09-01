"""A stage's cache identity must cover everything the stage consumes.

Both defects here were found by independent review, and both are the same
mistake in opposite directions: a value that reaches a stage but not its key.
"""

import pytest

from card_reviewer.review.enums import Coverage, Scale
from card_reviewer.review.fingerprint import fingerprint
from card_reviewer.review.policies.coverage_v1 import REQUIRED_FACES, evaluate_coverage
from card_reviewer.review.taxonomy import CATEGORIES, defect_types_for


def test_a_centering_disagreement_between_two_photos_can_be_fingerprinted():
    """`_conflicts` emits raw floats under a key with no declared precision,
    so the one case it exists to preserve used to abort the whole review with
    an unhandled ValueError before any attempt row was written."""
    payload = {"assembled_evidence": {"conflicts": [
        {"field": "centering.horizontal", "values": [52.0, 61.5],
         "images": ["hash-a", "hash-b"]}]}}
    assert fingerprint(payload)


def test_a_conflict_is_quantized_at_centerings_declared_precision():
    """Conflicting values ARE centering values, so they share its bucket:
    a difference finer than the method resolves is the same evidence."""
    def fp(first, second):
        return fingerprint({"assembled_evidence": {"conflicts": [
            {"field": "centering.horizontal", "values": [first, second],
             "images": ["a", "b"]}]}})

    assert fp(52.0, 61.5) == fp(52.0, 61.5)
    assert fp(52.0, 61.5) != fp(52.0, 71.5)


def _detectability(value=Scale.HIGH):
    return {(f, c, d): value for f in REQUIRED_FACES
            for c in CATEGORIES for d in defect_types_for(c)}


def test_reason_codes_change_coverage_so_they_must_change_its_key():
    """Same detectability, different reason code, different outcome:
    STRUCTURAL does not block a category, CIRCUMSTANTIAL does. Sharing a
    cache key between them serves a glared card a clean card's coverage —
    I2 broken through the cache."""
    detectability = _detectability()
    for face in REQUIRED_FACES:
        for category in ("corners", "edges"):
            detectability[(face, category, "whitening")] = Scale.LOW

    structural = {(f, c, "whitening"): "WHITE_BORDER"
                  for f in REQUIRED_FACES for c in ("corners", "edges")}
    circumstantial = {(f, c, "whitening"): "GLARE"
                      for f in REQUIRED_FACES for c in ("corners", "edges")}

    assert evaluate_coverage(detectability, structural, {},
                             REQUIRED_FACES).outcome is Coverage.SUFFICIENT
    assert evaluate_coverage(detectability, circumstantial, {},
                             REQUIRED_FACES).outcome is not Coverage.SUFFICIENT

    flat = {f"{f.value}|{c}|{d}": v.label
            for (f, c, d), v in detectability.items()}

    def key(reasons):
        return fingerprint({
            "assembled_detectability": flat,
            "assembled_reason_codes": {f"{f.value}|{c}|{d}": code
                                       for (f, c, d), code in reasons.items()},
            "applicable_rubric_rules": [],
        })

    assert key(structural) != key(circumstantial)


@pytest.mark.parametrize("stage", ["coverage_provisional", "coverage"])
def test_both_coverage_stages_declare_the_reason_codes_they_read(stage):
    from card_reviewer.review.fingerprint import STAGE_FINGERPRINT_INPUTS

    assert "assembled_reason_codes" in STAGE_FINGERPRINT_INPUTS[stage]


@pytest.mark.parametrize("stage", ["coverage_provisional", "coverage"])
def test_both_coverage_stages_declare_the_unevaluable_rules_they_read(stage):
    """`unevaluable_rules` adds metadata-resolvable limitations, blocks
    categories and drives the card-identification request."""
    from card_reviewer.review.fingerprint import STAGE_FINGERPRINT_INPUTS

    assert "unevaluable_rubric_rules" in STAGE_FINGERPRINT_INPUTS[stage]


def test_every_stage_passes_exactly_the_inputs_it_declares(tmp_path):
    """The declaration table and the pipeline's call sites must agree.

    Both defects this module guards were a value reaching a stage without
    reaching its key. A table nobody checks against the real calls would let
    the next one through the same way, so this asserts on the fingerprints the
    real pipeline actually builds.
    """
    from card_reviewer.review.enums import Mode
    from card_reviewer.review.fingerprint import STAGE_FINGERPRINT_INPUTS
    from card_reviewer.review.imaging.synthetic import CardSpec, render_png
    from card_reviewer.review.ingest.adapter import ManualAdapter
    from card_reviewer.review.models import CandidateInput
    from card_reviewer.review.pipeline import ReviewPipeline, StageRunner
    from card_reviewer.review.storage.artifacts import ArtifactStore
    from card_reviewer.review.storage.migrations import connect, migrate
    from card_reviewer.review.storage.repository import SqliteRepository

    seen: dict[str, set[str]] = {}
    original = StageRunner.run_with_id

    def recording(self, stage, inputs, signature, compute, **kw):
        seen.setdefault(stage, set()).update(inputs)
        return original(self, stage, inputs, signature, compute, **kw)

    StageRunner.run_with_id = recording
    try:
        store = ArtifactStore(tmp_path / "store")
        conn = connect(tmp_path / "t.db")
        migrate(conn)
        paths = []
        for i, spec in enumerate((CardSpec(), CardSpec(text_heavy=True))):
            path = tmp_path / f"{i}.png"
            path.write_bytes(render_png(spec))
            paths.append(path)
        resolved = ManualAdapter(store).resolve(CandidateInput(
            source="manual", title="2023 Topps Chrome", candidate_id="c",
            image_paths=paths,
            supplied_roles={str(paths[0]): "front", str(paths[1]): "back"}))
        ReviewPipeline(SqliteRepository(conn), store).review(resolved, Mode.OFF)
        conn.close()
    finally:
        StageRunner.run_with_id = original

    assert seen, "no stage was observed running"
    for stage, passed in sorted(seen.items()):
        assert passed == set(STAGE_FINGERPRINT_INPUTS[stage]), (
            f"{stage} declares {sorted(STAGE_FINGERPRINT_INPUTS[stage])} "
            f"but the pipeline passes {sorted(passed)}")


def test_two_readings_in_one_precision_bucket_share_a_conflict_fingerprint():
    """A conflict's values are centering readings, so a difference finer
    than centering resolves is the same evidence and must not split the
    cache."""
    def fp(first):
        return fingerprint({"assembled_evidence": {"conflicts": [
            {"field": "centering.horizontal", "values": [first, 61.5],
             "images": ["a", "b"]}]}})

    assert fp(52.0) == fp(52.1), "0.1pp is well inside the declared 1.5pp"
    assert fp(52.0) != fp(58.0)


def test_the_rubrics_iteration_order_is_not_part_of_a_coverage_key():
    """Unevaluable rules arrive in whatever order the rubric yields them.
    Letting that into the key splits one result across two rows."""
    from card_reviewer.review.policies.coverage_v1 import (
        UnevaluableRule, unevaluable_fingerprint_content,
    )

    gaps = [
        UnevaluableRule(rule_id="SURFACE_SHINY_001", category="surface",
                        reason_code="UNKNOWN_PRODUCT_CONTEXT"),
        UnevaluableRule(rule_id="CENTERING_X_001", category="centering",
                        reason_code="UNKNOWN_PRODUCT_CONTEXT"),
    ]
    assert (fingerprint({"u": unevaluable_fingerprint_content(gaps)})
            == fingerprint({"u": unevaluable_fingerprint_content(gaps[::-1])}))
    # ...but a different gap is still a different key.
    assert (fingerprint({"u": unevaluable_fingerprint_content(gaps)})
            != fingerprint({"u": unevaluable_fingerprint_content(gaps[:1])}))
