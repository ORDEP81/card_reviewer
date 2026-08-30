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
    """A13: build() now returns the `Rubric` it built (so a caller like
    build_rubric_cmd doesn't have to re-parse every rule file a second time
    via load_active_rubric just to report version/count) rather than the
    Path it wrote."""
    r = rubric.build(project)
    assert isinstance(r, rubric.Rubric)
    assert r.version == "0.3.0"
    assert len(r.rules) == 3
    assert project.rubric_file.exists()
    assert "0.3.0" in project.rubric_file.read_text()


def test_empty_knowledge_base_renders_without_error(tmp_path):
    p = ProjectPaths(tmp_path)
    p.knowledge.mkdir(parents=True)
    r = rubric.load_active_rubric(tmp_path)
    assert r.rules == []
    assert "0 active rules" in rubric.render(r)


# --- for_card: independent-axis regression (fix round 1) ---
#
# The `project` fixture above has no sets-scoped rule, which is exactly why the
# asymmetry bug (a call that names only one axis silently drops rules scoped on
# the *other*, unnamed axis) went unnoticed. This fixture adds one unscoped
# rule, one card_types-scoped rule, and one sets-scoped rule so both axes can be
# exercised independently.


@pytest.fixture
def scoped_project(tmp_path):
    p = ProjectPaths(tmp_path)
    for rule in [
        make("UNSCOPED_001"),
        make("TYPESCOPED_001", card_types=["chrome"]),
        make("SETSCOPED_001", sets=["2023-topps"]),
    ]:
        path = p.rules / rule.category.value / f"{rule.id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(rule.model_dump(mode="json"), sort_keys=False))
    version.write(p, "0.1.0")
    return p


def test_for_card_naming_only_card_types_keeps_sets_scoped_rule(scoped_project):
    """Not naming `sets` means 'unknown', not 'empty' -- the sets-scoped rule
    must not be silently dropped just because the caller hasn't resolved the
    set yet."""
    r = rubric.load_active_rubric(scoped_project.root)
    ids = {rule.id for rule in r.for_card(card_types=["chrome"])}
    assert ids == {"UNSCOPED_001", "TYPESCOPED_001", "SETSCOPED_001"}


def test_for_card_naming_only_sets_keeps_type_scoped_rule(scoped_project):
    r = rubric.load_active_rubric(scoped_project.root)
    ids = {rule.id for rule in r.for_card(sets=["2023-topps"])}
    assert ids == {"UNSCOPED_001", "SETSCOPED_001", "TYPESCOPED_001"}


def test_for_card_naming_both_axes_returns_all_matching(scoped_project):
    r = rubric.load_active_rubric(scoped_project.root)
    ids = {
        rule.id
        for rule in r.for_card(card_types=["chrome"], sets=["2023-topps"])
    }
    assert ids == {"UNSCOPED_001", "TYPESCOPED_001", "SETSCOPED_001"}


def test_for_card_type_mismatch_still_keeps_sets_scoped_rule(scoped_project):
    """A non-matching card type excludes the type-scoped rule but must not
    touch the sets-scoped rule, which that call said nothing about."""
    r = rubric.load_active_rubric(scoped_project.root)
    ids = {rule.id for rule in r.for_card(card_types=["paper"])}
    assert "TYPESCOPED_001" not in ids
    assert "SETSCOPED_001" in ids
    assert "UNSCOPED_001" in ids


def test_for_card_explicit_empty_card_types_is_a_real_constraint(scoped_project):
    """`card_types=[]` means 'known to have no card types' -- unlike omitting
    the argument, this excludes any rule scoped on that axis."""
    r = rubric.load_active_rubric(scoped_project.root)
    ids = {rule.id for rule in r.for_card(card_types=[])}
    assert ids == {"UNSCOPED_001", "SETSCOPED_001"}


def test_for_card_no_arguments_returns_everything(scoped_project):
    r = rubric.load_active_rubric(scoped_project.root)
    assert len(r.for_card()) == 3


# --- load_active_rubric's failure mode (fix round 2, finding 1) ---
#
# A single corrupt file under knowledge/rules/ used to raise a raw
# yaml.parser.ParserError (or pydantic ValidationError) straight out of the
# public contract subsystem A imports. `load_active_rubric` must instead
# raise one documented exception type, chained from the original, naming
# the offending file.


def test_load_active_rubric_raises_rubric_error_on_corrupt_file(tmp_path):
    p = ProjectPaths(tmp_path)
    bad_dir = p.rules / "surface"
    bad_dir.mkdir(parents=True)
    bad_file = bad_dir / "BROKEN_001.yaml"
    bad_file.write_text("sources: [not, closed\n")  # unterminated flow sequence
    version.write(p, "0.1.0")

    with pytest.raises(rubric.RubricError) as excinfo:
        rubric.load_active_rubric(tmp_path)

    assert str(bad_file) in str(excinfo.value)
    assert excinfo.value.__cause__ is not None


def test_load_active_rubric_raises_rubric_error_on_schema_violation(tmp_path):
    """Not just unparseable YAML -- a file that parses but fails Rule's
    schema (e.g. a missing required field) must also surface as
    RubricError, not a raw pydantic ValidationError."""
    p = ProjectPaths(tmp_path)
    bad_dir = p.rules / "surface"
    bad_dir.mkdir(parents=True)
    bad_file = bad_dir / "INVALID_001.yaml"
    bad_file.write_text(yaml.safe_dump({"id": "INVALID_001"}))
    version.write(p, "0.1.0")

    with pytest.raises(rubric.RubricError) as excinfo:
        rubric.load_active_rubric(tmp_path)

    assert str(bad_file) in str(excinfo.value)
    assert excinfo.value.__cause__ is not None


def test_render_shows_reference_mode_citations():
    """A reference-mode source must appear in the rendered rubric.

    render() joined `lesson` with `timestamps`; a reference source has no
    timestamps, so it rendered as a bare lesson id and the reader could not
    see which document the rule came from. The rules resting on PSA's
    published standard are exactly the ones whose provenance matters most.
    """
    import datetime

    from card_reviewer.knowledge.models import Rule, RuleSource

    rule = Rule(
        id="CENTERING_DOC_001",
        category="centering",
        statement="A documented centering tolerance.",
        evidence_type="objective",
        confidence="high",
        sources=[
            RuleSource(
                lesson="lesson_013",
                reference="PSA published grading standards",
                locator="Grade Definitions, GEM-MT PSA 10",
                quote="within a tolerance not to exceed approximately 55/45 percent",
            )
        ],
        status="active",
        created=datetime.date(2026, 8, 30),
        rubric_version_added="3.0.0",
    )
    text = rubric.render(rubric.Rubric(version="3.0.0", rules=[rule]))
    assert "PSA published grading standards" in text
    assert "Grade Definitions, GEM-MT PSA 10" in text


def test_render_still_shows_video_mode_citations():
    """The video-mode citation format must not regress."""
    import datetime

    from card_reviewer.knowledge.models import Rule, RuleSource

    rule = Rule(
        id="CENTERING_VID_001",
        category="centering",
        statement="A video-sourced claim.",
        evidence_type="experience_based",
        confidence="medium",
        sources=[
            RuleSource(lesson="lesson_001", video_id="yt_abc", timestamps=["04:16-04:28"])
        ],
        status="active",
        created=datetime.date(2026, 8, 30),
        rubric_version_added="3.0.0",
    )
    text = rubric.render(rubric.Rubric(version="3.0.0", rules=[rule]))
    assert "lesson_001" in text
    assert "04:16-04:28" in text


def test_render_uses_the_models_notion_of_an_absent_video_id():
    """_cite must agree with RuleSource about what "absent" means.

    RuleSource treats a whitespace-only string as absent, so a source with
    video_id="" is a valid reference-mode source. Branching on
    `video_id is not None` sent it down the video path and rendered a bare
    lesson id - the same defect this fix set out to remove.
    """
    import datetime

    from card_reviewer.knowledge.models import Rule, RuleSource

    source = RuleSource(
        lesson="lesson_x",
        video_id="",
        reference="PSA published grading standards",
        locator="Grade Definitions, GEM-MT PSA 10",
    )
    rule = Rule(
        id="CENTERING_BLANKVID_001",
        category="centering",
        statement="A reference-mode claim whose video_id is blank rather than absent.",
        evidence_type="objective",
        confidence="high",
        sources=[source],
        status="active",
        created=datetime.date(2026, 8, 30),
        rubric_version_added="4.0.0",
    )
    text = rubric.render(rubric.Rubric(version="4.0.0", rules=[rule]))
    assert "PSA published grading standards" in text
