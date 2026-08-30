import datetime

import pytest
import yaml

from card_reviewer.knowledge import manifest as mf, validate
from card_reviewer.knowledge.models import Manifest, SourceInfo
from card_reviewer.knowledge.paths import ProjectPaths


def rule_dict(**over):
    base = {
        "id": "SURFACE_PRINT_LINE_001",
        "category": "surface",
        "statement": "Vertical print lines commonly prevent a PSA 10.",
        "evidence_type": "experience_based",
        "confidence": "high",
        "applies_to": {"card_types": ["chrome"], "sets": []},
        "sources": [
            {
                "lesson": "lesson_001",
                "video_id": "yt_abc",
                "timestamps": ["05:00-05:30"],
                "quote": "look at that line",
            }
        ],
        "status": "pending",
        "supersedes": None,
        "created": datetime.date(2026, 8, 28),
        "rubric_version_added": None,
    }
    return base | over


@pytest.fixture
def project(tmp_path):
    p = ProjectPaths(tmp_path)
    p.pending_rules.mkdir(parents=True)
    p.rules.mkdir(parents=True)
    p.lessons.mkdir(parents=True)
    p.lesson("lesson_001").write_text("# Lesson 1\n")
    m = Manifest(
        video_id="yt_abc",
        source=SourceInfo(type="youtube", url="u", title="t", duration_s=600.0),
        rubric_version_at_ingest="0.1.0",
    )
    mf.save(p, m)
    return p


def reference_source(**over):
    base = {
        "lesson": "lesson_001",
        "reference": "PSA Grading Standards",
        "locator": "Card Grading Standards, Gem Mint 10",
        "quote": "55/45 or better front centering and 75/25 or better back.",
    }
    return base | over


def write_pending(p, data, name=None):
    path = p.pending_rules / f"{name or data['id']}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_parse_timestamp_range():
    assert validate.parse_timestamp("12:04-12:38") == (724.0, 758.0)


def test_parse_timestamp_single_point():
    assert validate.parse_timestamp("12:04") == (724.0, 724.0)


def test_parse_timestamp_with_hours():
    assert validate.parse_timestamp("1:02:03-1:02:30") == (3723.0, 3750.0)


def test_parse_timestamp_rejects_garbage():
    with pytest.raises(validate.BadTimestamp):
        validate.parse_timestamp("sometime near the end")


def test_valid_rule_passes(project):
    write_pending(project, rule_dict())
    report = validate.run(project)
    assert report.ok
    assert report.errors == {}
    assert report.checked == 1


def test_timestamp_beyond_video_duration_is_rejected(project):
    """The load-bearing check: a citation past the end of the video is fabricated."""
    data = rule_dict()
    data["sources"][0]["timestamps"] = ["59:00-59:30"]  # video is 600s
    write_pending(project, data)
    report = validate.run(project)
    assert not report.ok
    assert any("exceeds" in e for e in report.errors["SURFACE_PRINT_LINE_001"])


def test_missing_lesson_is_rejected(project):
    data = rule_dict()
    data["sources"][0]["lesson"] = "lesson_999"
    write_pending(project, data)
    report = validate.run(project)
    assert any("lesson_999" in e for e in report.errors["SURFACE_PRINT_LINE_001"])


def test_unknown_video_id_is_rejected(project):
    data = rule_dict()
    data["sources"][0]["video_id"] = "yt_nope"
    write_pending(project, data)
    report = validate.run(project)
    assert any("yt_nope" in e for e in report.errors["SURFACE_PRINT_LINE_001"])


def test_id_colliding_with_an_active_rule_is_rejected(project):
    active = rule_dict(status="active", rubric_version_added="0.1.0")
    (project.rules / "surface").mkdir(parents=True)
    (project.rules / "surface" / "SURFACE_PRINT_LINE_001.yaml").write_text(
        yaml.safe_dump(active, sort_keys=False)
    )
    write_pending(project, rule_dict())
    report = validate.run(project)
    assert any(
        "already exists with status 'active'" in e
        for e in report.errors["SURFACE_PRINT_LINE_001"]
    )


def test_id_colliding_with_a_rejected_rule_is_rejected(project):
    """Critical 2: a rejected rule's id must never be silently reused —
    otherwise the rejection reason (and the fact the grader declined the
    claim) is destroyed by whatever pending rule reuses the id."""
    rejected = rule_dict(status="rejected", notes="rejected: instructor opinion")
    (project.rules / "surface").mkdir(parents=True)
    (project.rules / "surface" / "SURFACE_PRINT_LINE_001.yaml").write_text(
        yaml.safe_dump(rejected, sort_keys=False)
    )
    write_pending(project, rule_dict(statement="A totally different claim."))
    report = validate.run(project)
    assert not report.ok
    assert any(
        "already exists with status 'rejected'" in e
        for e in report.errors["SURFACE_PRINT_LINE_001"]
    )
    # The rejected rule on disk must be untouched by the failed validation run.
    stored = yaml.safe_load(
        (project.rules / "surface" / "SURFACE_PRINT_LINE_001.yaml").read_text()
    )
    assert stored["status"] == "rejected"
    assert stored["statement"] == rejected["statement"]


def test_id_colliding_with_a_superseded_rule_is_rejected(project):
    superseded = rule_dict(status="superseded", rubric_version_added="0.1.0")
    (project.rules / "surface").mkdir(parents=True)
    (project.rules / "surface" / "SURFACE_PRINT_LINE_001.yaml").write_text(
        yaml.safe_dump(superseded, sort_keys=False)
    )
    write_pending(project, rule_dict())
    report = validate.run(project)
    assert not report.ok
    assert any(
        "already exists with status 'superseded'" in e
        for e in report.errors["SURFACE_PRINT_LINE_001"]
    )


def test_pending_rule_with_supersedes_set_is_rejected(project):
    """Important 5: supersession is a review-time human decision, not
    something a pending rule (written by Claude) can declare for itself."""
    data = rule_dict(supersedes="CORNERS_OLD_001")
    write_pending(project, data)
    report = validate.run(project)
    assert not report.ok
    assert any("supersedes" in e for e in report.errors["SURFACE_PRINT_LINE_001"])


def test_pending_filename_not_matching_id_is_rejected(project):
    """Important 4: a mismatched filename lets `_drop_pending` promote the
    rule while the real file stays behind, wedging every future `review`."""
    write_pending(project, rule_dict(), name="totally_different_name")
    report = validate.run(project)
    assert not report.ok
    assert any(
        "does not match id" in e for e in report.errors["totally_different_name.yaml"]
    )


def test_duplicate_pending_id_across_two_files_flags_both(project):
    """A rule id must map to exactly one pending file; two files claiming
    the same id are ambiguous about which one promotion should apply."""
    write_pending(project, rule_dict(), name="first")
    write_pending(project, rule_dict(), name="second")
    report = validate.run(project)
    assert not report.ok
    assert "first.yaml" in report.errors
    assert "second.yaml" in report.errors
    assert any("more than one file" in e for e in report.errors["first.yaml"])
    assert any("more than one file" in e for e in report.errors["second.yaml"])


def test_pending_rule_marked_active_is_rejected(project):
    write_pending(project, rule_dict(status="active"))
    report = validate.run(project)
    assert any("status" in e for e in report.errors["SURFACE_PRINT_LINE_001"])


def test_malformed_yaml_is_reported_not_raised(project):
    (project.pending_rules / "broken.yaml").write_text("id: [unclosed\n")
    report = validate.run(project)
    assert not report.ok
    assert "broken.yaml" in report.errors


def test_missing_evidence_type_is_reported(project):
    data = rule_dict()
    del data["evidence_type"]
    write_pending(project, data)
    report = validate.run(project)
    assert not report.ok


def test_load_pending_skips_malformed_file_without_raising(project):
    write_pending(project, rule_dict())
    (project.pending_rules / "broken.yaml").write_text("id: [unclosed\n")
    loaded = validate.load_pending(project)
    assert [rule.id for _, rule in loaded] == ["SURFACE_PRINT_LINE_001"]


def test_load_pending_pairs_paths_with_their_own_rule(project):
    path = write_pending(project, rule_dict())
    loaded = validate.load_pending(project)
    assert loaded == [(path, loaded[0][1])]
    assert loaded[0][0] == path
    assert loaded[0][1].id == "SURFACE_PRINT_LINE_001"


def test_reference_mode_source_validates(project):
    data = rule_dict()
    data["sources"] = [reference_source()]
    write_pending(project, data)
    report = validate.run(project)
    assert report.ok
    assert report.errors == {}
    assert report.checked == 1


def test_rule_mixing_video_and_reference_sources_validates(project):
    data = rule_dict()
    data["sources"] = [rule_dict()["sources"][0], reference_source()]
    write_pending(project, data)
    report = validate.run(project)
    assert report.ok
    assert report.errors == {}


def test_reference_mode_source_with_missing_lesson_is_reported_not_raised(project):
    data = rule_dict()
    data["sources"] = [reference_source(lesson="lesson_999")]
    write_pending(project, data)
    report = validate.run(project)
    assert not report.ok
    assert any("lesson_999" in e for e in report.errors["SURFACE_PRINT_LINE_001"])


def test_reference_mode_source_requires_nonempty_reference_and_locator(project):
    data = rule_dict()
    data["sources"] = [reference_source(reference="   ")]
    write_pending(project, data)
    report = validate.run(project)
    assert not report.ok
    assert any("reference" in e for e in report.errors["SURFACE_PRINT_LINE_001"])


def test_video_mode_out_of_bounds_timestamp_still_rejected_with_mixed_sources(project):
    """Trust boundary: mixing in a reference source must not weaken the
    video-mode duration check."""
    data = rule_dict()
    video_source = rule_dict()["sources"][0]
    video_source["timestamps"] = ["59:00-59:30"]  # video is 600s
    data["sources"] = [video_source, reference_source()]
    write_pending(project, data)
    report = validate.run(project)
    assert not report.ok
    assert any("exceeds" in e for e in report.errors["SURFACE_PRINT_LINE_001"])


def test_unparseable_citation_timestamp_is_rejected(project):
    data = rule_dict()
    data["sources"][0]["timestamps"] = ["sometime near the end"]
    write_pending(project, data)
    report = validate.run(project)
    assert not report.ok
    assert any(
        "sometime near the end" in e
        for e in report.errors["SURFACE_PRINT_LINE_001"]
    )


# --- fix round 2, finding 2: a corrupt file already under knowledge/rules/
# must not be invisible to `validate run`. `load_all` used to catch the
# exact exceptions `run`'s except clause was written to handle, making that
# clause dead and the "active rules are unreadable" message a lie -- a
# broken active rule file was silently skipped and `run` reported clean.
# Chosen fix: `load_all` now reports what it skips instead of only skipping
# it, and `run` surfaces each corrupt file keyed by its path.


def test_load_all_reports_unparseable_files_instead_of_only_skipping(project):
    bad_dir = project.rules / "surface"
    bad_dir.mkdir(parents=True)
    bad_file = bad_dir / "BROKEN_001.yaml"
    bad_file.write_text("sources: [not, closed\n")

    rules, errors = validate.load_all(project)

    assert rules == []
    assert len(errors) == 1
    (key, message) = next(iter(errors.items()))
    assert str(bad_file) in key
    assert "does not parse" in message


def test_run_reports_corrupt_active_rule_file_keyed_by_its_path(project):
    bad_dir = project.rules / "surface"
    bad_dir.mkdir(parents=True)
    bad_file = bad_dir / "BROKEN_001.yaml"
    bad_file.write_text("sources: [not, closed\n")

    report = validate.run(project)

    assert report.ok is False
    assert any(str(bad_file) in key for key in report.errors)


def test_run_still_checks_pending_rules_despite_a_corrupt_active_file(project):
    bad_dir = project.rules / "surface"
    bad_dir.mkdir(parents=True)
    (bad_dir / "BROKEN_001.yaml").write_text("sources: [not, closed\n")
    write_pending(project, rule_dict())

    report = validate.run(project)

    assert report.ok is False  # corrupt active file alone keeps this false
    assert report.checked == 1  # pending validation still ran
