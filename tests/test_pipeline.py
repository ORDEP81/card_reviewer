import pytest

from card_reviewer.knowledge import manifest as mf, pipeline
from card_reviewer.knowledge.models import Manifest, SourceInfo
from card_reviewer.knowledge.paths import ProjectPaths


@pytest.fixture
def paths(tmp_path):
    return ProjectPaths(tmp_path)


def fake_steps(paths, calls):
    def make_packet(*a, **k):
        m = Manifest(
            video_id="yt_abc",
            source=SourceInfo(type="youtube", url="u", title="t", duration_s=100.0),
            rubric_version_at_ingest="0.1.0",
        )
        mf.save(paths, m)
        calls.append("acquire")
        return mf.finish(paths, m, "acquire")

    def stage(name):
        def go(p, video_id, **kwargs):
            calls.append(name)
            m = mf.load(p, video_id)
            mf.finish(p, m, name)

        return go

    return {
        "acquire": make_packet,
        "transcribe": stage("transcribe"),
        "segment": stage("segment"),
        "extract_frames": stage("extract_frames"),
    }


def test_run_all_advances_every_deterministic_stage(paths):
    calls = []
    m = pipeline.run_all(paths, url="https://youtu.be/abc", steps=fake_steps(paths, calls))
    assert calls == ["acquire", "transcribe", "segment", "extract_frames"]
    assert all(mf.is_done(m, s) for s in pipeline.DETERMINISTIC_STAGES)


def test_run_all_stops_before_analyze(paths):
    calls = []
    m = pipeline.run_all(paths, url="https://youtu.be/abc", steps=fake_steps(paths, calls))
    assert "analyze" not in calls
    assert not mf.is_done(m, "analyze")


def test_completed_stages_are_skipped_on_rerun(paths):
    calls = []
    steps = fake_steps(paths, calls)
    pipeline.run_all(paths, url="https://youtu.be/abc", steps=steps)
    calls.clear()
    pipeline.run_all(paths, url="https://youtu.be/abc", steps=steps)
    assert calls == ["acquire"]  # acquire re-runs to locate the packet; rest skipped


def test_force_reruns_completed_stages(paths):
    calls = []
    steps = fake_steps(paths, calls)
    pipeline.run_all(paths, url="https://youtu.be/abc", steps=steps)
    calls.clear()
    pipeline.run_all(paths, url="https://youtu.be/abc", steps=steps, force=True)
    assert calls == ["acquire", "transcribe", "segment", "extract_frames"]


def test_status_lists_all_packets(paths):
    for vid in ("yt_a", "yt_b"):
        mf.save(
            paths,
            Manifest(
                video_id=vid,
                source=SourceInfo(type="youtube", url="u", title=vid, duration_s=1.0),
                rubric_version_at_ingest="0.1.0",
            ),
        )
    assert {m.video_id for m in pipeline.status(paths)} == {"yt_a", "yt_b"}


def test_status_for_one_packet(paths):
    mf.save(
        paths,
        Manifest(
            video_id="yt_a",
            source=SourceInfo(type="youtube", url="u", title="t", duration_s=1.0),
            rubric_version_at_ingest="0.1.0",
        ),
    )
    assert len(pipeline.status(paths, "yt_a")) == 1


def test_run_all_requires_a_source(paths):
    with pytest.raises(ValueError):
        pipeline.run_all(paths)
