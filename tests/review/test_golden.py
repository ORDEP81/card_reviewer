"""Golden regression fixtures.

These assert OBSERVATIONS, never grades. A photograph is never labelled
"this is a PSA 10" — subjective grading opinion must not become fake ground
truth, and actual PSA results belong in the calibration dataset instead.

The fixtures are currently synthetic stand-ins, each flagged `synthetic: true`
so they are replaced with real photographs rather than mistaken for
real-world coverage.
"""

from pathlib import Path

import pytest
import yaml

from card_reviewer.review.enums import Scale
from card_reviewer.review.imaging.geometry import analyze as geom
from card_reviewer.review.imaging.measure.centering import measure_centering
from card_reviewer.review.imaging.observability import analyze as obs
from card_reviewer.review.imaging.preflight import analyze as pre
from card_reviewer.review.storage.artifacts import ArtifactStore

GOLDEN = Path(__file__).parent / "golden"
CASES = yaml.safe_load((GOLDEN / "expectations.yaml").read_text())


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["file"])
def test_golden_observations_hold(case, store):
    data = (GOLDEN / case["file"]).read_bytes()
    preflight = pre(data)
    assert preflight.usable is case["preflight_usable"]
    if not case["preflight_usable"]:
        assert preflight.reason_code == case["preflight_reason"]
        return

    image_hash = store.put_image(data)
    geometry = geom(data, store, image_hash)
    assert (geometry.boundary_confidence > 0.5) is case["boundary_detected"]
    assert geometry.has_reliable_border is case["has_reliable_border"]

    assert measure_centering(geometry, store).measurable is (
        case["centering_measurable"])

    observed = obs(geometry, store, image_hash)
    for key, expected in case["detectability"].items():
        assert observed.detectability[tuple(key.split("."))].label == expected

    if "corner_whitening_reason" in case:
        codes = {v for k, v in observed.reason_codes.items()
                 if k[1] == "corners" and k[2] == "whitening"}
        assert case["corner_whitening_reason"] in codes


def test_no_golden_case_asserts_a_psa_grade():
    """Guard: subjective grading opinion must never become ground truth."""
    forbidden = {"grade", "psa_grade", "expected_grade", "verdict", "is_psa10",
                 "psa10_candidate", "rank_score"}
    for case in CASES:
        assert not (set(case) & forbidden), f"{case['file']} asserts a grade"


def test_every_fixture_is_covered_by_an_expectation():
    """A fixture with no expectations is a file nothing checks."""
    images = {p.name for p in GOLDEN.glob("*.png")}
    assert images == {c["file"] for c in CASES}


def test_synthetic_fixtures_are_flagged_as_such():
    """They stand in until real photographs arrive; a silently unflagged
    synthetic fixture would read as real-world coverage it does not provide."""
    assert all(c.get("synthetic") is True for c in CASES)


def test_the_set_spans_the_conditions_the_policies_branch_on():
    """Each of these drives a different path through coverage or geometry."""
    covered = {c["file"] for c in CASES}
    for required in ("white_border_clean.png", "dark_border_clean.png",
                     "borderless_chrome.png", "glare_heavy.png",
                     "angled_card.png", "thumbnail_too_small.png"):
        assert required in covered
