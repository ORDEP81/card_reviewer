import json
import subprocess

import pytest
from PIL import Image

from card_reviewer.knowledge import frames, manifest as mf
from card_reviewer.knowledge.models import Manifest, SourceInfo
from card_reviewer.knowledge.paths import ProjectPaths


def fake_ffmpeg(produced_names):
    """Simulate ffmpeg by writing the files it would have produced."""

    def run(cmd, **kwargs):
        out_pattern = cmd[-1]
        out_dir = __import__("pathlib").Path(out_pattern).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in produced_names:
            (out_dir / name).write_bytes(b"fake-jpeg")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return run


def test_sample_invokes_ffmpeg_with_the_right_window(tmp_path):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        (tmp_path / "out").mkdir(exist_ok=True)
        (tmp_path / "out" / "frame_0001.jpg").write_bytes(b"x")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    out = frames.sample(
        tmp_path / "video.mp4", tmp_path / "out", 120.0, 150.0, fps=1.0, runner=run
    )
    flat = " ".join(calls[0])
    assert "-ss 120.0" in flat
    assert "-t 30.0" in flat
    assert "fps=1.0" in flat
    assert len(out) == 1


def test_sample_caps_frame_count(tmp_path):
    run = fake_ffmpeg([f"frame_{i:04d}.jpg" for i in range(1, 51)])
    out = frames.sample(
        tmp_path / "v.mp4", tmp_path / "out", 0.0, 100.0, cap=20, runner=run
    )
    assert len(out) == 20
    # Frames beyond the cap are deleted, not merely hidden.
    assert len(list((tmp_path / "out").glob("*.jpg"))) == 20


def test_dedupe_removes_near_identical_frames(tmp_path):
    paths_ = []
    for i in range(4):
        p = tmp_path / f"f{i}.jpg"
        p.write_bytes(b"x")
        paths_.append(p)

    # f0, f1, f2 are near-identical; f3 differs.
    fake_hashes = {paths_[0]: 0, paths_[1]: 1, paths_[2]: 2, paths_[3]: 999}
    survivors = frames.dedupe(
        paths_, threshold=5, hasher=lambda p: fake_hashes[p]
    )
    assert len(survivors) == 2
    assert paths_[0] in survivors
    assert paths_[3] in survivors
    assert not paths_[1].exists()


def test_dedupe_keeps_everything_when_all_differ(tmp_path):
    paths_ = []
    for i in range(3):
        p = tmp_path / f"f{i}.jpg"
        p.write_bytes(b"x")
        paths_.append(p)
    survivors = frames.dedupe(paths_, threshold=5, hasher=lambda p: hash(p.name) % 10000)
    assert len(survivors) == 3


def test_run_blocks_before_segment_stage(tmp_path):
    p = ProjectPaths(tmp_path)
    m = Manifest(
        video_id="yt_abc",
        source=SourceInfo(type="youtube", url="u", title="t", duration_s=100.0),
        rubric_version_at_ingest="0.1.0",
    )
    mf.save(p, m)
    with pytest.raises(mf.StageNotReady):
        frames.run(p, "yt_abc")


def test_sample_clears_stale_frames_before_extracting(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "frame_0099.jpg").write_bytes(b"stale")

    run = fake_ffmpeg(["frame_0001.jpg", "frame_0002.jpg"])
    out = frames.sample(tmp_path / "v.mp4", out_dir, 0.0, 10.0, runner=run)

    names = sorted(p.name for p in out_dir.glob("*.jpg"))
    assert names == ["frame_0001.jpg", "frame_0002.jpg"]
    assert len(out) == 2
    assert not (out_dir / "frame_0099.jpg").exists()


def test_sample_raises_runtime_error_with_ffmpeg_stderr(tmp_path):
    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no such filter: bogus")

    with pytest.raises(RuntimeError, match="no such filter: bogus"):
        frames.sample(tmp_path / "v.mp4", tmp_path / "out", 0.0, 10.0, runner=run)


def recording_ffmpeg(images_per_call=1):
    """Simulate ffmpeg by writing real (tiny, valid) images, so the default
    perceptual hasher can run against them without invoking real ffmpeg."""
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        out_pattern = cmd[-1]
        out_dir = __import__("pathlib").Path(out_pattern).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_call):
            color = ((i * 37) % 256, (i * 61) % 256, (i * 89) % 256)
            Image.new("RGB", (8, 8), color=color).save(out_dir / f"frame_{i + 1:04d}.jpg")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    run.calls = calls
    return run


def _ready_manifest(paths_, video_id="yt_abc", duration_s=100.0):
    """A manifest with every stage before extract_frames marked done, plus a
    source video file on disk — the minimum frames.run() needs to proceed."""
    m = Manifest(
        video_id=video_id,
        source=SourceInfo(type="youtube", url="u", title="t", duration_s=duration_s),
        rubric_version_at_ingest="0.1.0",
    )
    mf.save(paths_, m)
    for stage in ("acquire", "transcribe", "segment"):
        m = mf.finish(paths_, m, stage)
    paths_.source_dir(video_id).mkdir(parents=True, exist_ok=True)
    (paths_.source_dir(video_id) / "video.mp4").write_bytes(b"fake-video")
    return m


def _write_segments(paths_, video_id, n):
    segments = [
        {"id": f"seg_{i:03d}", "start_s": float(i * 10), "end_s": float(i * 10 + 5), "score": float(n - i)}
        for i in range(n)
    ]
    paths_.segments(video_id).write_text(
        json.dumps({"lexicon_version": "1.0.0", "total_cues": n, "segments": segments})
    )


def test_run_ranked_branch_honors_top_n_and_marks_stage_done(tmp_path):
    p = ProjectPaths(tmp_path)
    _ready_manifest(p, duration_s=200.0)
    _write_segments(p, "yt_abc", n=5)

    runner = recording_ffmpeg()
    total = frames.run(p, "yt_abc", top_n=3, runner=runner)

    created = sorted(d.name for d in p.frames("yt_abc").iterdir())
    assert created == ["seg_000", "seg_001", "seg_002"]
    assert total == 3  # one distinct frame kept per segment

    reloaded = mf.load(p, "yt_abc")
    assert mf.is_done(reloaded, "extract_frames")


def test_run_at_does_not_mark_stage_done(tmp_path):
    p = ProjectPaths(tmp_path)
    _ready_manifest(p, duration_s=200.0)

    runner = recording_ffmpeg()
    frames.run(p, "yt_abc", at=5.0, window_s=10.0, runner=runner)

    reloaded = mf.load(p, "yt_abc")
    assert not mf.is_done(reloaded, "extract_frames")


def test_run_uniform_branch_stops_at_video_duration(tmp_path):
    p = ProjectPaths(tmp_path)
    _ready_manifest(p, duration_s=100.0)

    runner = recording_ffmpeg()
    frames.run(p, "yt_abc", top_n=12, uniform=True, runner=runner)

    starts = []
    ends = []
    for cmd in runner.calls:
        ss = float(cmd[cmd.index("-ss") + 1])
        t = float(cmd[cmd.index("-t") + 1])
        starts.append(ss)
        ends.append(ss + t)

    # 100s of video at the floor step (30s) fits only 4 windows (0/30/60/90);
    # a 5th at 120 would start past the end and must not be produced.
    assert len(runner.calls) == 4
    assert all(s < 100.0 for s in starts)
    assert all(e <= 100.0 for e in ends)
