from card_reviewer.knowledge.paths import ProjectPaths


def test_packet_paths_are_derived_from_root(tmp_path):
    p = ProjectPaths(tmp_path)
    assert p.packet("yt_abc") == tmp_path / "training" / "work" / "yt_abc"
    assert p.manifest("yt_abc") == p.packet("yt_abc") / "manifest.json"
    assert p.transcript("yt_abc") == p.packet("yt_abc") / "transcript.json"
    assert p.segments("yt_abc") == p.packet("yt_abc") / "segments.json"
    assert p.frames("yt_abc") == p.packet("yt_abc") / "frames"
    assert p.source_dir("yt_abc") == p.packet("yt_abc") / "source"


def test_knowledge_paths(tmp_path):
    p = ProjectPaths(tmp_path)
    assert p.pending_rules == tmp_path / "knowledge" / "pending_rules"
    assert p.rules == tmp_path / "knowledge" / "rules"
    assert p.rubric_file == tmp_path / "knowledge" / "ACTIVE_RUBRIC.md"
    assert p.version_file == tmp_path / "knowledge" / "RUBRIC_VERSION"
    assert p.lexicon_file == tmp_path / "knowledge" / "segmentation_lexicon.yaml"
    assert p.lesson("lesson_001") == tmp_path / "training" / "lessons" / "lesson_001.md"
