import datetime

import pytest
import yaml

from card_reviewer.knowledge import promote, validate
from card_reviewer.knowledge.models import Rule, RuleSource, RuleStatus
from card_reviewer.knowledge.paths import ProjectPaths


def make(rule_id="SURFACE_PRINT_LINE_001", **over):
    base = dict(
        id=rule_id,
        category="surface",
        statement="Vertical print lines commonly prevent a PSA 10.",
        evidence_type="experience_based",
        confidence="high",
        sources=[RuleSource(lesson="lesson_001", video_id="yt_a", timestamps=["01:00"])],
        created=datetime.date(2026, 8, 28),
    )
    return Rule(**(base | over))


@pytest.fixture
def paths(tmp_path):
    p = ProjectPaths(tmp_path)
    p.pending_rules.mkdir(parents=True)
    p.rules.mkdir(parents=True)
    return p


def write_pending(paths, rule):
    path = paths.pending_rules / f"{rule.id}.yaml"
    path.write_text(yaml.safe_dump(rule.model_dump(mode="json"), sort_keys=False))
    return path


def test_accept_moves_rule_to_category_directory(paths):
    rule = make()
    pending = write_pending(paths, rule)
    dest = promote.accept(paths, rule, "0.2.0")
    assert dest == paths.rules / "surface" / "SURFACE_PRINT_LINE_001.yaml"
    assert dest.exists()
    assert not pending.exists()


def test_accept_sets_status_and_stamps_version(paths):
    rule = make()
    write_pending(paths, rule)
    dest = promote.accept(paths, rule, "0.2.0")
    stored = Rule.model_validate(yaml.safe_load(dest.read_text()))
    assert stored.status is RuleStatus.ACTIVE
    assert stored.rubric_version_added == "0.2.0"


def test_accepted_rule_is_visible_to_load_active(paths):
    rule = make()
    write_pending(paths, rule)
    promote.accept(paths, rule, "0.2.0")
    assert [r.id for r in validate.load_active(paths)] == ["SURFACE_PRINT_LINE_001"]


def test_reject_keeps_the_rule_on_disk_with_reason(paths):
    """Spec §7 and plan §30 rule 9: nothing is ever deleted."""
    rule = make()
    pending = write_pending(paths, rule)
    dest = promote.reject(paths, rule, "instructor opinion, contradicted by lesson_003")
    assert dest.exists()
    assert not pending.exists()
    stored = Rule.model_validate(yaml.safe_load(dest.read_text()))
    assert stored.status is RuleStatus.REJECTED
    assert "contradicted by lesson_003" in stored.notes


def test_rejected_rule_is_not_active(paths):
    rule = make()
    write_pending(paths, rule)
    promote.reject(paths, rule, "opinion")
    assert validate.load_active(paths) == []


def test_supersede_activates_new_and_retires_old(paths):
    old = make("SURFACE_PRINT_LINE_001")
    write_pending(paths, old)
    promote.accept(paths, old, "0.1.0")

    new = make("SURFACE_PRINT_LINE_002", statement="Print lines under 1cm may still gem.")
    write_pending(paths, new)
    new_path, old_path = promote.supersede(paths, new, "SURFACE_PRINT_LINE_001", "1.0.0")

    stored_new = Rule.model_validate(yaml.safe_load(new_path.read_text()))
    stored_old = Rule.model_validate(yaml.safe_load(old_path.read_text()))
    assert stored_new.status is RuleStatus.ACTIVE
    assert stored_new.supersedes == "SURFACE_PRINT_LINE_001"
    assert stored_old.status is RuleStatus.SUPERSEDED
    assert stored_old.id == "SURFACE_PRINT_LINE_001"  # id is never reused
    assert [r.id for r in validate.load_active(paths)] == ["SURFACE_PRINT_LINE_002"]


def test_supersede_unknown_rule_raises(paths):
    new = make("SURFACE_PRINT_LINE_002")
    write_pending(paths, new)
    with pytest.raises(promote.UnknownRule):
        promote.supersede(paths, new, "SURFACE_NOPE_999", "1.0.0")
