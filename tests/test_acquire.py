import json
import subprocess

import pytest

from card_reviewer.knowledge import acquire, manifest as mf
from card_reviewer.knowledge.models import StageStatus
from card_reviewer.knowledge.paths import ProjectPaths

METADATA = {
    "id": "abc123",
    "title": "Grading 101",
    "uploader": "Someone",
    "duration": 3120,
    "ext": "mp4",
}


@pytest.fixture
def paths(tmp_path):
    return ProjectPaths(tmp_path)


def recording_runner(calls, *, fail_on=None, stdout=""):
    def run(cmd, **kwargs):
        calls.append(cmd)
        if fail_on and fail_on in " ".join(cmd):
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="ERROR: sign in to confirm"
            )
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    return run


def test_derive_video_id_from_youtube_watch_url():
    assert acquire.derive_video_id(url="https://www.youtube.com/watch?v=abc123") == "yt_abc123"


def test_derive_video_id_from_youtube_short_url():
    assert acquire.derive_video_id(url="https://youtu.be/abc123") == "yt_abc123"


def test_derive_video_id_from_skool_url_is_stable_hash():
    url = "https://www.skool.com/mlp/classroom/xyz"
    first = acquire.derive_video_id(url=url)
    assert first.startswith("skool_")
    assert first == acquire.derive_video_id(url=url)


def test_derive_video_id_from_local_file_hashes_content(tmp_path):
    f = tmp_path / "lesson.mp4"
    f.write_bytes(b"pretend video bytes")
    vid = acquire.derive_video_id(file=f)
    assert vid.startswith("local_")
    assert vid == acquire.derive_video_id(file=f)


def test_derive_video_id_rejects_traversal_shaped_v_param():
    # A `v=` value shaped like a path-traversal payload must never survive
    # into the id unescaped -- it becomes a directory name via ProjectPaths.
    vid = acquire.derive_video_id(url="https://www.youtube.com/watch?v=..%2F..%2Fetc")
    assert "/" not in vid
    assert "\\" not in vid
    assert ".." not in vid


def test_derive_video_id_youtube_id_round_trips_unchanged():
    # Ordinary, well-formed YouTube ids are unaffected by the safety check.
    assert (
        acquire.derive_video_id(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        == "yt_dQw4w9WgXcQ"
    )


def test_from_url_uses_cookies_from_browser_when_requested(paths):
    calls = []
    runner = recording_runner(calls, stdout=json.dumps(METADATA))
    with pytest.raises(acquire.AcquisitionFailed):
        # No real file lands on disk, so this fails at the verify step. We only
        # care that the cookie flag was passed correctly.
        acquire.from_url(
            paths,
            "https://www.skool.com/mlp/x",
            rubric_version="0.1.0",
            browser="chrome",
            runner=runner,
        )
    flat = [" ".join(c) for c in calls]
    assert any("--cookies-from-browser chrome" in f for f in flat)


def test_from_url_never_writes_a_cookie_file(paths):
    calls = []
    runner = recording_runner(calls, stdout=json.dumps(METADATA))
    with pytest.raises(acquire.AcquisitionFailed):
        acquire.from_url(
            paths, "https://www.skool.com/mlp/x", "0.1.0", browser="chrome", runner=runner
        )
    flat = " ".join(" ".join(c) for c in calls)
    assert "--cookies " not in flat
    assert "cookies.txt" not in flat


def test_failed_download_records_failure_and_gives_manual_path(paths):
    calls = []
    runner = recording_runner(calls, fail_on="--dump-json", stdout="")
    url = "https://www.skool.com/mlp/x"
    with pytest.raises(acquire.AcquisitionFailed) as exc:
        acquire.from_url(paths, url, "0.1.0", runner=runner)
    assert "sign in" in str(exc.value)
    assert "--file" in exc.value.guidance

    # The failure must be durable, not just raised: spec §9 requires the
    # stage to record failed with yt-dlp's stderr before it stops.
    video_id = acquire.derive_video_id(url=url)
    recorded = mf.load(paths, video_id)
    assert recorded.stages["acquire"].status is StageStatus.FAILED
    assert "sign in" in recorded.stages["acquire"].error


def test_failed_download_does_not_retry(paths):
    """Spec §9: failure is terminal. Exactly one metadata attempt."""
    calls = []
    runner = recording_runner(calls, fail_on="--dump-json")
    with pytest.raises(acquire.AcquisitionFailed):
        acquire.from_url(paths, "https://www.skool.com/mlp/x", "0.1.0", runner=runner)
    dump_calls = [c for c in calls if "--dump-json" in c]
    assert len(dump_calls) == 1


def test_from_file_adopts_local_video(paths, tmp_path):
    src = tmp_path / "lesson.mp4"
    src.write_bytes(b"pretend video bytes")
    runner = recording_runner([], stdout="42.5")
    m = acquire.from_file(paths, src, rubric_version="0.1.0", runner=runner)
    assert m.source.type == "local"
    assert m.source.duration_s == 42.5
    assert m.file is not None
    assert m.stages["acquire"].status is StageStatus.DONE
    assert paths.manifest(m.video_id).exists()
    # Original is copied into the packet, not moved.
    assert src.exists()


def test_from_file_rerun_preserves_downstream_stages(paths, tmp_path):
    """Critical 1: re-running `acquire` for a packet that has already been
    transcribed/segmented/etc. must not reset those stages to pending —
    only `acquire` itself, and `source`/`file`, may legitimately change."""
    src = tmp_path / "lesson.mp4"
    src.write_bytes(b"pretend video bytes")
    runner = recording_runner([], stdout="42.5")

    m = acquire.from_file(paths, src, rubric_version="0.1.0", runner=runner)
    mf.finish(paths, m, "transcribe")
    mf.finish(paths, m, "segment")
    mf.finish(paths, m, "extract_frames")
    m.lesson_id = "lesson_014"
    mf.save(paths, m)

    # Re-acquire the same file (e.g. a second `card-knowledge run`).
    m2 = acquire.from_file(paths, src, rubric_version="0.1.0", runner=runner)

    assert m2.stages["acquire"].status is StageStatus.DONE
    assert m2.stages["transcribe"].status is StageStatus.DONE
    assert m2.stages["segment"].status is StageStatus.DONE
    assert m2.stages["extract_frames"].status is StageStatus.DONE
    assert m2.lesson_id == "lesson_014"

    reloaded = mf.load(paths, m2.video_id)
    assert reloaded.stages["transcribe"].status is StageStatus.DONE
    assert reloaded.stages["segment"].status is StageStatus.DONE
    assert reloaded.stages["extract_frames"].status is StageStatus.DONE
