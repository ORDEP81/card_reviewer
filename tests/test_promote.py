import datetime

import pytest
import yaml

from card_reviewer.knowledge import promote, validate, version
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


def test_supersede_rejected_rule_raises_and_leaves_it_unchanged(paths):
    """Fix round 1, finding 2: a rejected id must not be silently retired."""
    rule = make("SURFACE_PRINT_LINE_001")
    write_pending(paths, rule)
    dest = promote.reject(paths, rule, "instructor opinion")
    before = dest.read_text()

    new = make("SURFACE_PRINT_LINE_002")
    write_pending(paths, new)
    with pytest.raises(promote.UnknownRule):
        promote.supersede(paths, new, "SURFACE_PRINT_LINE_001", "1.0.0")

    assert dest.read_text() == before
    stored = Rule.model_validate(yaml.safe_load(dest.read_text()))
    assert stored.status is RuleStatus.REJECTED


def test_supersede_already_superseded_rule_raises(paths):
    """Fix round 1, finding 2: a superseded id must not be re-retired."""
    old = make("SURFACE_PRINT_LINE_001")
    write_pending(paths, old)
    promote.accept(paths, old, "0.1.0")

    mid = make("SURFACE_PRINT_LINE_002", statement="Print lines under 1cm may still gem.")
    write_pending(paths, mid)
    promote.supersede(paths, mid, "SURFACE_PRINT_LINE_001", "1.0.0")

    new = make("SURFACE_PRINT_LINE_003", statement="An unrelated later finding about surface.")
    write_pending(paths, new)
    with pytest.raises(promote.UnknownRule):
        promote.supersede(paths, new, "SURFACE_PRINT_LINE_001", "2.0.0")


def test_session_bump_level_supersede_outranks_accept():
    assert promote.session_bump_level(accepted=True, superseded=True) == "major"


def test_session_bump_level_accept_only_is_minor():
    assert promote.session_bump_level(accepted=True, superseded=False) == "minor"


def test_session_bump_level_supersede_only_is_major():
    assert promote.session_bump_level(accepted=False, superseded=True) == "major"


def test_session_bump_level_no_decisions_is_none():
    assert promote.session_bump_level(accepted=False, superseded=False) is None


def test_accept_only_session_bumps_minor_and_stamps_it(paths):
    """Fix round 1, finding 1: models the pattern `review_cmd` follows for an
    accept-only session — one version computed and used for every stamp and
    the one version-file write."""
    version.write(paths, "0.1.0")
    rule = make()
    write_pending(paths, rule)

    level = promote.session_bump_level(accepted=True, superseded=False)
    assert level == "minor"
    new_version = version.bump(version.read(paths), level)
    assert new_version == "0.2.0"

    dest = promote.accept(paths, rule, new_version)
    version.write(paths, new_version)

    stored = Rule.model_validate(yaml.safe_load(dest.read_text()))
    assert stored.rubric_version_added == "0.2.0"
    assert version.read(paths) == "0.2.0"


def test_mixed_accept_and_supersede_session_writes_version_once(paths):
    """Fix round 1, finding 1: a session with both an accept and a supersede
    must produce exactly one version value, stamped on every rule it touches,
    and exactly one write to the version file — never a minor bump for the
    accept followed by a major bump that strands the accept's stamp on a
    version that was never actually current on disk."""
    version.write(paths, "0.1.0")

    old = make("SURFACE_PRINT_LINE_001")
    write_pending(paths, old)
    promote.accept(paths, old, "0.1.0")
    version.write(paths, "0.1.0")

    accept_rule = make(
        "SURFACE_PRINT_LINE_002",
        statement="Whitening confined to a single edge rarely caps the grade alone.",
    )
    write_pending(paths, accept_rule)

    supersede_rule = make(
        "SURFACE_PRINT_LINE_003", statement="Print lines under 1cm may still gem."
    )
    write_pending(paths, supersede_rule)

    # This mirrors what review_cmd now does: collect decisions during the
    # walk, then compute ONE bump level and ONE version for the session.
    level = promote.session_bump_level(accepted=True, superseded=True)
    assert level == "major"
    new_version = version.bump(version.read(paths), level)
    assert new_version == "1.0.0"

    accept_dest = promote.accept(paths, accept_rule, new_version)
    new_path, old_path = promote.supersede(
        paths, supersede_rule, "SURFACE_PRINT_LINE_001", new_version
    )
    version.write(paths, new_version)

    assert version.read(paths) == "1.0.0"
    stamped_accept = Rule.model_validate(yaml.safe_load(accept_dest.read_text()))
    stamped_new = Rule.model_validate(yaml.safe_load(new_path.read_text()))
    assert stamped_accept.rubric_version_added == "1.0.0"
    assert stamped_new.rubric_version_added == "1.0.0"
    # The rubric version on disk is exactly the version every touched rule
    # was stamped with — no rule names a version that never stood on disk.
    assert stamped_accept.rubric_version_added == version.read(paths)
    assert stamped_new.rubric_version_added == version.read(paths)
