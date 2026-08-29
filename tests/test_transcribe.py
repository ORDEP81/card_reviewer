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
