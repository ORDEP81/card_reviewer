"""Golden regression fixtures.

These assert OBSERVATIONS, never grades. A photograph is never labelled
"this is a PSA 10" — subjective grading opinion must not become fake ground
truth, and actual PSA results belong in the calibration dataset instead.

The fixtures are currently synthetic stand-ins, each flagged `synthetic: true`
so they are replaced with real photographs rather than mistaken for
real-world coverage.

The expectations are DERIVED FROM THE IMPLEMENTATION by golden/generate.py,
which makes almost everything here a regression baseline: it detects change,
and it cannot detect that today's behaviour is wrong. The exception is
`rendered_centering`, which comes from the generator rather than the
measurement — that one assertion compares against ground truth, and it has
already earned its place by catching a 6.4pp error on the card back.
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

    centering = measure_centering(geometry, store)
    assert centering.measurable is case["centering_measurable"]
    if centering.measurable:
        # The VALUE, not merely that one exists. Asserting only
        # `measurable: true` left the miscut fixture indistinguishable from
        # the clean ones, so several cases constrained nothing at all.
        assert round(centering.horizontal, 1) == case["centering_horizontal"]
        assert round(centering.vertical, 1) == case["centering_vertical"]
    else:
        assert centering.reason == case["centering_reason"]

    observed = obs(geometry, store, image_hash)
    for key, expected in case["detectability"].items():
        assert observed.detectability[tuple(key.split("."))].label == expected
    for key, expected in case.get("reason_codes", {}).items():
        assert observed.reason_codes[tuple(key.split("."))] == expected


@pytest.mark.parametrize("case", [c for c in CASES if "rendered_centering" in c
                                  and c.get("centering_measurable")],
                         ids=lambda c: c["file"])
def test_a_measured_centering_matches_what_was_actually_drawn(case, store):
    """The one assertion here that is NOT derived from the implementation.

    `rendered_centering` comes from the generator, so this compares a
    measurement against ground truth rather than against yesterday's output
    — the single check in this file that could catch the implementation
    being wrong rather than merely different.
    """
    data = (GOLDEN / case["file"]).read_bytes()
    geometry = geom(data, store, store.put_image(data))
    measured = measure_centering(geometry, store)
    truth_h, truth_v = case["rendered_centering"]

    assert abs(measured.horizontal - truth_h) <= 3.0, (
        f"{case['file']}: measured {measured.horizontal} against a rendered "
        f"{truth_h}")
    assert abs(measured.vertical - truth_v) <= 3.0


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
                     "angled_card.png", "thumbnail_too_small.png",
                     "miscut.png", "scratched_surface.png",
                     "occluded_corner.png"):
        assert required in covered


def test_no_two_cases_assert_the_same_thing():
    """Four of the eight original cases carried byte-identical expectation
    bodies, so three of them added no constraint while looking like
    coverage. A case that stops distinguishing itself has stopped earning
    its place in the set."""
    bodies = {}
    for case in CASES:
        body = {k: v for k, v in case.items() if k != "file"}
        key = repr(sorted(body.items(), key=str))
        bodies.setdefault(key, []).append(case["file"])

    duplicates = {k: v for k, v in bodies.items() if len(v) > 1}
    assert not duplicates, (
        "these cases assert exactly the same thing: "
        + "; ".join(", ".join(files) for files in duplicates.values()))


def test_the_expectations_disclose_that_they_come_from_the_implementation():
    """A baseline read as ground truth is worse than no baseline."""
    text = (GOLDEN / "expectations.yaml").read_text()
    assert "REGRESSION" in text
    assert "not ground truth" in text
    assert (GOLDEN / "generate.py").exists(), (
        "the fixtures cannot be regenerated, so `synthetic: true` is an "
        "unverifiable claim about opaque PNGs")
