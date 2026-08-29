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
    assert any("already active" in e for e in report.errors["SURFACE_PRINT_LINE_001"])


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
