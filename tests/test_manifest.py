import pytest

from card_reviewer.knowledge import manifest as mf
from card_reviewer.knowledge.models import Manifest, SourceInfo, StageStatus
from card_reviewer.knowledge.paths import ProjectPaths


@pytest.fixture
def paths(tmp_path):
    return ProjectPaths(tmp_path)


@pytest.fixture
def packet(paths):
    m = Manifest(
        video_id="yt_abc",
        source=SourceInfo(type="youtube", url="u", title="t", duration_s=100.0),
        rubric_version_at_ingest="0.1.0",
    )
    mf.save(paths, m)
    return m


def test_save_then_load_roundtrips(paths, packet):
    loaded = mf.load(paths, "yt_abc")
    assert loaded.video_id == "yt_abc"
    assert loaded.source.duration_s == 100.0


def test_load_missing_packet_raises(paths):
    with pytest.raises(mf.PacketNotFound):
        mf.load(paths, "nope")


def test_finish_marks_done_and_records_detail(paths, packet):
    m = mf.finish(paths, packet, "acquire", tool="yt-dlp 2026.1.1")
    assert m.stages["acquire"].status is StageStatus.DONE
    assert m.stages["acquire"].at is not None
    assert m.stages["acquire"].detail["tool"] == "yt-dlp 2026.1.1"
    assert mf.is_done(mf.load(paths, "yt_abc"), "acquire")


def test_fail_records_error_and_is_not_done(paths, packet):
    m = mf.fail(paths, packet, "acquire", "yt-dlp exited 1: sign in required")
    assert m.stages["acquire"].status is StageStatus.FAILED
    assert "sign in required" in m.stages["acquire"].error
    assert not mf.is_done(m, "acquire")


def test_require_ready_blocks_when_prerequisite_incomplete(paths, packet):
    """transcribe cannot run before acquire is done."""
    with pytest.raises(mf.StageNotReady, match="acquire"):
        mf.require_ready(packet, "transcribe")


def test_require_ready_passes_when_prerequisites_done(paths, packet):
    m = mf.finish(paths, packet, "acquire")
    mf.require_ready(m, "transcribe")  # must not raise


def test_first_stage_has_no_prerequisites(packet):
    mf.require_ready(packet, "acquire")  # must not raise


def test_failed_prerequisite_also_blocks(paths, packet):
    m = mf.fail(paths, packet, "acquire", "boom")
    with pytest.raises(mf.StageNotReady):
        mf.require_ready(m, "transcribe")
