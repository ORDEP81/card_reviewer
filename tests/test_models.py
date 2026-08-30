"""Models enforce the spec's non-negotiables at the type level."""
import datetime

import pytest
from pydantic import ValidationError

from card_reviewer.knowledge.models import (
    STAGES,
    AppliesTo,
    Confidence,
    EvidenceType,
    Manifest,
    Rule,
    RuleSource,
    RuleStatus,
    SourceInfo,
    StageState,
    StageStatus,
)


def a_source(**over):
    base = dict(
        lesson="lesson_014",
        video_id="yt_abc123",
        timestamps=["12:04-12:38"],
        quote="I've never seen one gem with a line like that.",
    )
    return RuleSource(**(base | over))


def a_reference_source(**over):
    base = dict(
        lesson="lesson_014",
        reference="PSA Grading Standards",
        locator="Card Grading Standards, Gem Mint 10",
        quote="55/45 or better front centering and 75/25 or better back.",
    )
    return RuleSource(**(base | over))


def a_rule(**over):
    base = dict(
        id="SURFACE_PRINT_LINE_001",
        category="surface",
        statement="Vertical print lines commonly prevent a PSA 10.",
        evidence_type=EvidenceType.EXPERIENCE_BASED,
        confidence=Confidence.HIGH,
        sources=[a_source()],
        created=datetime.date(2026, 8, 28),
    )
    return Rule(**(base | over))


def test_stage_order_is_the_spec_order():
    assert STAGES == (
        "acquire",
        "transcribe",
        "segment",
        "extract_frames",
        "analyze",
        "validate",
    )


def test_rule_requires_evidence_type():
    """Spec §7: evidence_type is required with no default (plan §30 rule 11)."""
    with pytest.raises(ValidationError):
        Rule(
            id="SURFACE_001",
            category="surface",
            statement="x",
            confidence=Confidence.HIGH,
            sources=[a_source()],
            created=datetime.date(2026, 8, 28),
        )


def test_rule_requires_at_least_one_source():
    with pytest.raises(ValidationError):
        a_rule(sources=[])


def test_rule_rejects_empty_statement():
    with pytest.raises(ValidationError):
        a_rule(statement="   ")


def test_rule_defaults_to_pending_and_unversioned():
    rule = a_rule()
    assert rule.status is RuleStatus.PENDING
    assert rule.rubric_version_added is None
    assert rule.applies_to == AppliesTo()


def test_rule_id_must_be_uppercase_slug():
    with pytest.raises(ValidationError):
        a_rule(id="surface print line 1")


def test_manifest_starts_every_stage_pending():
    manifest = Manifest(
        video_id="yt_abc123",
        source=SourceInfo(
            type="youtube",
            url="https://youtube.com/watch?v=abc123",
            title="Grading 101",
            uploader="Someone",
            duration_s=3120.0,
        ),
        rubric_version_at_ingest="0.1.0",
    )
    assert set(manifest.stages) == set(STAGES)
    assert all(s.status is StageStatus.PENDING for s in manifest.stages.values())


def test_rule_notes_default_to_none():
    """notes carries rejection reasons; Task 13 writes it."""
    assert a_rule().notes is None


def test_rule_accepts_notes():
    assert a_rule(notes="rejected: opinion").notes == "rejected: opinion"


def test_stage_state_carries_error_on_failure():
    state = StageState(status=StageStatus.FAILED, error="yt-dlp exited 1")
    assert state.error == "yt-dlp exited 1"


def test_stage_state_rejects_naive_datetime():
    """All timestamps in manifests are UTC ISO-8601; naive datetimes are rejected."""
    naive_dt = datetime.datetime(2026, 8, 29, 12, 0, 0)
    with pytest.raises(ValidationError):
        StageState(status=StageStatus.DONE, at=naive_dt)


def test_video_mode_source_still_validates():
    source = a_source()
    assert source.video_id == "yt_abc123"
    assert source.timestamps == ["12:04-12:38"]
    assert source.reference is None
    assert source.locator is None


def test_reference_mode_source_validates():
    source = a_reference_source()
    assert source.reference == "PSA Grading Standards"
    assert source.locator == "Card Grading Standards, Gem Mint 10"
    assert source.video_id is None
    assert source.timestamps == []


def test_rule_may_mix_video_and_reference_sources():
    rule = a_rule(sources=[a_source(), a_reference_source()])
    assert len(rule.sources) == 2


def test_source_rejects_both_modes_at_once():
    with pytest.raises(ValidationError):
        a_source(reference="PSA Grading Standards", locator="p.2")


def test_source_rejects_neither_mode():
    with pytest.raises(ValidationError):
        RuleSource(lesson="lesson_014", quote="unsupported claim")


def test_source_rejects_video_id_without_timestamps():
    with pytest.raises(ValidationError):
        RuleSource(lesson="lesson_014", video_id="yt_abc123")


def test_source_rejects_reference_without_locator():
    with pytest.raises(ValidationError):
        RuleSource(lesson="lesson_014", reference="PSA Grading Standards")


def test_stage_state_accepts_utc_aware_datetime():
    """UTC-aware datetimes are accepted and Task 3's datetime.now(datetime.UTC) works."""
    utc_dt = datetime.datetime(2026, 8, 29, 12, 0, 0, tzinfo=datetime.UTC)
    state = StageState(status=StageStatus.DONE, at=utc_dt)
    assert state.at == utc_dt
