import subprocess

import pytest

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
