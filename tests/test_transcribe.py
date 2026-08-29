import subprocess

import pytest

from card_reviewer.knowledge import manifest as mf, transcribe
from card_reviewer.knowledge.models import Manifest, SourceInfo
from card_reviewer.knowledge.paths import ProjectPaths

VTT = """WEBVTT

00:00:01.000 --> 00:00:04.500
Look right here at this corner.

00:00:04.500 --> 00:00:08.000
You can see the whitening on the edge.
"""


def test_parse_vtt_extracts_cues_with_seconds():
    cues = transcribe.parse_vtt(VTT)
    assert len(cues) == 2
    assert cues[0].start_s == 1.0
    assert cues[0].end_s == 4.5
    assert cues[0].text == "Look right here at this corner."
    assert cues[1].start_s == 4.5


def test_parse_vtt_handles_hourless_timestamps():
    cues = transcribe.parse_vtt("WEBVTT\n\n01:02.000 --> 01:05.000\nHello.\n")
    assert cues[0].start_s == 62.0
    assert cues[0].end_s == 65.0


def test_parse_vtt_joins_multiline_cue_text():
    cues = transcribe.parse_vtt(
        "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nfirst line\nsecond line\n"
    )
    assert cues[0].text == "first line second line"


def test_parse_vtt_ignores_cue_identifiers_and_notes():
    body = "WEBVTT\n\nNOTE something\n\ncue-7\n00:00:01.000 --> 00:00:02.000\nText.\n"
    cues = transcribe.parse_vtt(body)
    assert len(cues) == 1
    assert cues[0].text == "Text."


def test_parse_vtt_on_empty_input_returns_no_cues():
    assert transcribe.parse_vtt("WEBVTT\n") == []


def test_fetch_captions_returns_none_on_nonzero_returncode(tmp_path):
    def runner(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no such video")

    assert transcribe.fetch_captions("https://x/video", tmp_path, runner=runner) is None


def test_fetch_captions_returns_none_when_no_vtt_written(tmp_path):
    def runner(cmd, **kwargs):
        # Succeeds but the video genuinely has no captions.
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    assert transcribe.fetch_captions("https://x/video", tmp_path, runner=runner) is None


def test_fetch_captions_returns_parsed_cues_when_vtt_is_written(tmp_path):
    def runner(cmd, **kwargs):
        (tmp_path / "captions.en.vtt").write_text(VTT)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    cues = transcribe.fetch_captions("https://x/video", tmp_path, runner=runner)
    assert cues is not None
    assert len(cues) == 2
    assert cues[0].text == "Look right here at this corner."


def test_fetch_captions_returns_none_when_vtt_parses_to_zero_cues(tmp_path):
    def runner(cmd, **kwargs):
        (tmp_path / "captions.en.vtt").write_text("WEBVTT\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    assert transcribe.fetch_captions("https://x/video", tmp_path, runner=runner) is None


def test_fetch_captions_passes_cookies_from_browser_and_never_a_cookie_file(tmp_path):
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    transcribe.fetch_captions(
        "https://x/video", tmp_path, browser="chrome", runner=runner
    )
    flat = " ".join(" ".join(c) for c in calls)
    assert "--cookies-from-browser chrome" in flat
    assert "--cookies " not in flat
    assert "cookies.txt" not in flat


def test_fetch_captions_ignores_stale_vtt_left_by_an_earlier_run(tmp_path):
    # Regression: yt-dlp can exit 0 while writing nothing this invocation
    # (expired cookies, a re-upload that lost its captions, a subtitle-only
    # network failure). A caption file left over from a PRIOR run must never
    # be reported as this run's result -- that would be a false provenance
    # claim about which captions the transcript came from.
    (tmp_path / "captions.en.vtt").write_text(VTT)

    def runner(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    assert transcribe.fetch_captions("https://x/video", tmp_path, runner=runner) is None


@pytest.fixture
def packet(tmp_path):
    p = ProjectPaths(tmp_path)
    m = Manifest(
        video_id="yt_abc",
        source=SourceInfo(type="youtube", url="u", title="t", duration_s=100.0),
        rubric_version_at_ingest="0.1.0",
    )
    mf.save(p, m)
    mf.finish(p, m, "acquire")
    (p.source_dir("yt_abc")).mkdir(parents=True, exist_ok=True)
    (p.source_dir("yt_abc") / "video.mp4").write_bytes(b"x")
    m.file = None
    return p, m


def test_run_prefers_captions_when_available(packet, monkeypatch):
    p, _ = packet
    monkeypatch.setattr(
        transcribe, "fetch_captions", lambda *a, **k: transcribe.parse_vtt(VTT)
    )

    def explode(*a, **k):
        raise AssertionError("whisper must not run when captions exist")

    t = transcribe.run(p, "yt_abc", transcriber=explode)
    assert t.method == "captions"
    assert len(t.cues) == 2
    assert p.transcript("yt_abc").exists()


def test_run_falls_back_to_whisper_when_no_captions(packet, monkeypatch):
    p, _ = packet
    monkeypatch.setattr(transcribe, "fetch_captions", lambda *a, **k: None)

    def fake_transcriber(path, **kwargs):
        return {
            "language": "en",
            "segments": [{"start": 0.0, "end": 2.0, "text": " Whisper output."}],
        }

    t = transcribe.run(p, "yt_abc", transcriber=fake_transcriber)
    assert t.method == "mlx-whisper"
    assert t.cues[0].text == "Whisper output."


def test_run_blocks_when_acquire_not_done(tmp_path):
    p = ProjectPaths(tmp_path)
    m = Manifest(
        video_id="yt_zzz",
        source=SourceInfo(type="youtube", url="u", title="t", duration_s=1.0),
        rubric_version_at_ingest="0.1.0",
    )
    mf.save(p, m)
    with pytest.raises(mf.StageNotReady):
        transcribe.run(p, "yt_zzz")
