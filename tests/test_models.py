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
