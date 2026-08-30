import datetime

from card_reviewer.knowledge import dedup
from card_reviewer.knowledge.models import Rule, RuleSource


def make(rule_id, statement, category="surface"):
    return Rule(
        id=rule_id,
        category=category,
        statement=statement,
        evidence_type="experience_based",
        confidence="high",
        sources=[RuleSource(lesson="lesson_001", video_id="yt_a", timestamps=["01:00"])],
        created=datetime.date(2026, 8, 28),
    )


def test_normalize_strips_case_and_punctuation():
    assert dedup.normalize("A print LINE, running!") == "a print line running"


def test_identical_statements_are_flagged_duplicate():
    new = make("SURFACE_002", "Vertical print lines prevent a PSA 10.")
    active = [make("SURFACE_001", "Vertical print lines prevent a PSA 10.")]
    flags = dedup.flags_for(new, active)
    assert len(flags) == 1
    assert flags[0].kind == "duplicate"
    assert flags[0].other_id == "SURFACE_001"


def test_negation_mismatch_is_flagged_contradiction():
    new = make("SURFACE_002", "Vertical print lines do not prevent a PSA 10.")
    active = [make("SURFACE_001", "Vertical print lines prevent a PSA 10.")]
    flags = dedup.flags_for(new, active)
    assert flags[0].kind == "contradiction"


def test_different_categories_are_never_compared():
    new = make("CORNERS_001", "Vertical print lines prevent a PSA 10.", category="corners")
    active = [make("SURFACE_001", "Vertical print lines prevent a PSA 10.")]
    assert dedup.flags_for(new, active) == []


def test_unrelated_statements_produce_no_flags():
    new = make("SURFACE_002", "Refractor lines are a factory characteristic, not damage.")
    active = [make("SURFACE_001", "Vertical print lines prevent a PSA 10.")]
    assert dedup.flags_for(new, active) == []


def test_is_negated_detects_common_negations():
    assert dedup.is_negated("This will never gem.")
    assert dedup.is_negated("This does not gem.")
    assert dedup.is_negated("No card with this gems.")
    assert not dedup.is_negated("This card gems reliably.")


def test_flags_are_sorted_most_similar_first():
    new = make("SURFACE_003", "Vertical print lines prevent a PSA 10.")
    active = [
        make("SURFACE_001", "Vertical print lines on chrome prevent a high grade."),
        make("SURFACE_002", "Vertical print lines prevent a PSA 10."),
    ]
    flags = dedup.flags_for(new, active, threshold=0.5)
    assert flags[0].other_id == "SURFACE_002"
    assert flags[0].score >= flags[1].score


def test_is_negated_detects_contraction_negations():
    # Finding 1 fix: contractions without apostrophes should be detected
    assert dedup.is_negated("This card won't gem.")
    assert dedup.is_negated("This card doesn't gem.")


def test_is_negated_uses_word_boundaries_for_multi_word_negations():
    # Finding 2 fix: "does not" should not match inside "does notice"
    assert not dedup.is_negated("The grader does notice surface wear.")
    # Verify "does not" still matches correctly when used as a negation
    assert dedup.is_negated("This card does not gem.")


def test_contraction_negation_is_flagged_contradiction():
    # End-to-end: active affirmative + pending contraction-negated = contradiction
    new = make("SURFACE_002", "Vertical print lines won't prevent a PSA 10.")
    active = [make("SURFACE_001", "Vertical print lines prevent a PSA 10.")]
    flags = dedup.flags_for(new, active)
    assert len(flags) == 1
    assert flags[0].kind == "contradiction"
    assert flags[0].other_id == "SURFACE_001"


# --- A12: NEGATIONS had "wont"/"doesnt" but not the other common
# contractions -- normalize() strips apostrophes, so "don't"/"isn't"/etc.
# are otherwise invisible to is_negated.


def test_is_negated_detects_further_common_contractions():
    assert dedup.is_negated("This card don't gem.")
    assert dedup.is_negated("This isn't a defect.")
    assert dedup.is_negated("This wasn't graded correctly.")
    assert dedup.is_negated("These aren't gem quality.")
    assert dedup.is_negated("This can't be a 10.")


# --- B5: the self-id skip clause in flags_for is otherwise unguarded --
# removing `or other.id == rule.id` would let a rule be flagged against
# itself whenever the exact same Rule object (or an identical statement
# under the same id) shows up in the active list.


def test_flags_for_never_flags_a_rule_against_itself():
    rule = make("SURFACE_001", "Vertical print lines prevent a PSA 10.")
    flags = dedup.flags_for(rule, [rule])
    assert flags == []
