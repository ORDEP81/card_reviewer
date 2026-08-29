import datetime

import pytest
import yaml

from card_reviewer.knowledge import rubric, version
from card_reviewer.knowledge.models import Rule, RuleSource
from card_reviewer.knowledge.paths import ProjectPaths


def make(rule_id, category="surface", card_types=None, sets=None, status="active"):
    return Rule(
        id=rule_id,
        category=category,
        statement=f"Statement for {rule_id}.",
        evidence_type="objective",
        confidence="high",
        applies_to={"card_types": card_types or [], "sets": sets or []},
        sources=[RuleSource(lesson="lesson_001", video_id="yt_a", timestamps=["01:00"])],
        status=status,
        created=datetime.date(2026, 8, 28),
        rubric_version_added="0.1.0",
    )


@pytest.fixture
def project(tmp_path):
    p = ProjectPaths(tmp_path)
    for rule in [
        make("SURFACE_001"),
        make("SURFACE_002", card_types=["chrome", "refractor"]),
        make("CORNERS_001", category="corners"),
        make("SURFACE_003", status="rejected"),
    ]:
        path = p.rules / rule.category.value / f"{rule.id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(rule.model_dump(mode="json"), sort_keys=False))
    version.write(p, "0.3.0")
    return p


def test_load_active_rubric_carries_the_version(project):
    r = rubric.load_active_rubric(project.root)
    assert r.version == "0.3.0"


def test_load_active_rubric_excludes_non_active_rules(project):
    r = rubric.load_active_rubric(project.root)
    assert "SURFACE_003" not in {rule.id for rule in r.rules}


def test_by_category_filters(project):
    r = rubric.load_active_rubric(project.root)
    assert {rule.id for rule in r.by_category("surface")} == {"SURFACE_001", "SURFACE_002"}


def test_for_card_includes_unscoped_rules(project):
    r = rubric.load_active_rubric(project.root)
    ids = {rule.id for rule in r.for_card(card_types=["paper"])}
    assert "SURFACE_001" in ids  # unscoped applies to everything
    assert "SURFACE_002" not in ids  # scoped to chrome/refractor


def test_for_card_includes_matching_scoped_rules(project):
    r = rubric.load_active_rubric(project.root)
    ids = {rule.id for rule in r.for_card(card_types=["chrome"])}
    assert {"SURFACE_001", "SURFACE_002", "CORNERS_001"} <= ids


def test_for_card_with_no_arguments_returns_everything(project):
    r = rubric.load_active_rubric(project.root)
    assert len(r.for_card()) == 3


def test_render_includes_version_and_rule_count(project):
    r = rubric.load_active_rubric(project.root)
    text = rubric.render(r)
    assert "0.3.0" in text
    assert "3 active rules" in text


def test_render_groups_by_category(project):
    text = rubric.render(rubric.load_active_rubric(project.root))
    assert "## corners" in text
    assert "## surface" in text


def test_render_warns_against_hand_editing(project):
    """Spec §8: the markdown is a view, the YAML is the source of truth."""
    text = rubric.render(rubric.load_active_rubric(project.root))
    assert "generated" in text.lower()
    assert "do not edit" in text.lower()


def test_build_writes_the_rubric_file(project):
    path = rubric.build(project)
    assert path == project.rubric_file
    assert "0.3.0" in path.read_text()


def test_empty_knowledge_base_renders_without_error(tmp_path):
    p = ProjectPaths(tmp_path)
    p.knowledge.mkdir(parents=True)
    r = rubric.load_active_rubric(tmp_path)
    assert r.rules == []
    assert "0 active rules" in rubric.render(r)
