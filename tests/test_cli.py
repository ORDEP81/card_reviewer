from typer.testing import CliRunner

from card_reviewer.knowledge import cli
from card_reviewer.knowledge.cli import app
from card_reviewer.knowledge.manifest import PacketNotFound

runner = CliRunner()


def test_help_lists_the_doctor_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout


def test_doctor_runs():
    result = runner.invoke(app, ["doctor"])
    # Exit code 1 is correct when tools are genuinely missing on this machine.
    assert result.exit_code in (0, 1)
    assert "yt-dlp" in result.stdout


def test_bare_invocation_prints_help():
    result = runner.invoke(app, [])
    # Bare invocation with no arguments should print help and exit 0.
    # This guards against the Typer single-command collapse issue where
    # the help callback becomes unreachable (when no_args_is_help=True is present).
    assert result.exit_code == 0
    assert "doctor" in result.stdout
    assert "COMMAND" in result.stdout


def test_status_for_unknown_video_id_exits_cleanly():
    result = runner.invoke(app, ["status", "yt_typo"])
    assert result.exit_code == 1
    # A clean `typer.Exit` surfaces as SystemExit through CliRunner. If
    # PacketNotFound itself came out (i.e. status_cmd failed to catch it),
    # it would appear here instead, uncaught, with nothing printed below.
    assert not isinstance(result.exception, PacketNotFound)
    assert "yt_typo" in result.stdout


def test_status_renders_analyze_and_validate_as_not_pending(tmp_path, monkeypatch):
    """Important 3: nothing automated ever advances `analyze` or `validate`,
    so `status` must not render their untouched state as `pending` — that
    would misreport a fully analyzed, promoted video as stalled forever."""
    from card_reviewer.knowledge import manifest as mf
    from card_reviewer.knowledge.models import Manifest, SourceInfo
    from card_reviewer.knowledge.paths import ProjectPaths

    p = ProjectPaths(tmp_path)
    m = Manifest(
        video_id="yt_done",
        source=SourceInfo(type="youtube", url="u", title="Grading 101", duration_s=10.0),
        rubric_version_at_ingest="0.1.0",
    )
    mf.save(p, m)
    for stage in ("acquire", "transcribe", "segment", "extract_frames"):
        mf.finish(p, m, stage)

    monkeypatch.setattr(cli, "paths", lambda: p)
    result = runner.invoke(app, ["status", "yt_done"])

    assert result.exit_code == 0
    assert "pending" not in result.stdout
    assert "n/a" in result.stdout


def test_transcribe_reports_stage_not_ready_cleanly(tmp_path, monkeypatch):
    """Important 7: StageNotReady is a normal condition ('you haven't
    acquired yet'), not a bug — it must not surface as a raw traceback."""
    from card_reviewer.knowledge import manifest as mf
    from card_reviewer.knowledge.manifest import StageNotReady
    from card_reviewer.knowledge.models import Manifest, SourceInfo
    from card_reviewer.knowledge.paths import ProjectPaths

    p = ProjectPaths(tmp_path)
    m = Manifest(
        video_id="yt_fresh",
        source=SourceInfo(type="youtube", url="u", title="t", duration_s=10.0),
        rubric_version_at_ingest="0.1.0",
    )
    mf.save(p, m)  # acquire never ran

    monkeypatch.setattr(cli, "paths", lambda: p)
    result = runner.invoke(app, ["transcribe", "yt_fresh"])

    assert result.exit_code == 1
    assert not isinstance(result.exception, StageNotReady)
    assert "acquire" in result.stdout


def test_acquire_without_url_or_file_exits_cleanly():
    """Important 7: previously a bare ValueError traceback."""
    result = runner.invoke(app, ["acquire"])
    assert result.exit_code == 1
    assert not isinstance(result.exception, ValueError)
    assert "url" in result.stdout.lower() or "file" in result.stdout.lower()


def _rule_yaml(rule_id="SURFACE_PRINT_LINE_001", **over):
    base = {
        "id": rule_id,
        "category": "surface",
        "statement": "Vertical print lines commonly prevent a PSA 10.",
        "evidence_type": "experience_based",
        "confidence": "high",
        "applies_to": {"card_types": [], "sets": []},
        "sources": [
            {
                "lesson": "lesson_001",
                "video_id": "yt_abc",
                "timestamps": ["01:00"],
                "quote": "look at that line",
            }
        ],
        "status": "pending",
        "supersedes": None,
        "created": "2026-08-28",
        "rubric_version_added": None,
    }
    return base | over


def _review_project(tmp_path):
    from card_reviewer.knowledge import manifest as mf
    from card_reviewer.knowledge import version as ver
    from card_reviewer.knowledge.models import Manifest, SourceInfo
    from card_reviewer.knowledge.paths import ProjectPaths

    p = ProjectPaths(tmp_path)
    p.pending_rules.mkdir(parents=True)
    p.rules.mkdir(parents=True)
    p.lessons.mkdir(parents=True)
    p.lesson("lesson_001").write_text("# Lesson 1\n")
    m = Manifest(
        video_id="yt_abc",
        source=SourceInfo(type="youtube", url="u", title="t", duration_s=600.0),
        rubric_version_at_ingest="0.1.0",
    )
    mf.save(p, m)
    ver.write(p, "0.1.0")
    return p


def _write_pending(p, data):
    import yaml

    path = p.pending_rules / f"{data['id']}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_review_accept_only_session_bumps_minor_and_activates(tmp_path, monkeypatch):
    import yaml

    from card_reviewer.knowledge import validate
    from card_reviewer.knowledge import version as ver

    p = _review_project(tmp_path)
    _write_pending(p, _rule_yaml())
    monkeypatch.setattr(cli, "paths", lambda: p)

    result = runner.invoke(app, ["review"], input="accept\n")

    assert result.exit_code == 0, result.output
    assert "1 accepted, 0 superseded" in result.output
    assert ver.read(p) == "0.2.0"
    assert [r.id for r in validate.load_active(p)] == ["SURFACE_PRINT_LINE_001"]
    assert not (p.pending_rules / "SURFACE_PRINT_LINE_001.yaml").exists()


def test_review_reject_session_leaves_version_untouched(tmp_path, monkeypatch):
    import yaml

    from card_reviewer.knowledge import version as ver

    p = _review_project(tmp_path)
    _write_pending(p, _rule_yaml())
    monkeypatch.setattr(cli, "paths", lambda: p)

    result = runner.invoke(app, ["review"], input="reject\ninstructor opinion\n")

    assert result.exit_code == 0, result.output
    assert "0 accepted, 0 superseded" in result.output
    # Rejecting alone adds nothing to what the grader believes -- no bump.
    assert ver.read(p) == "0.1.0"
    stored = yaml.safe_load(
        (p.rules / "surface" / "SURFACE_PRINT_LINE_001.yaml").read_text()
    )
    assert stored["status"] == "rejected"
    assert "instructor opinion" in stored["notes"]


def test_review_supersede_session_retires_old_and_bumps_major(tmp_path, monkeypatch):
    import yaml

    from card_reviewer.knowledge import version as ver

    p = _review_project(tmp_path)
    old = _rule_yaml(status="active", rubric_version_added="0.1.0")
    (p.rules / "surface").mkdir(parents=True)
    (p.rules / "surface" / "SURFACE_PRINT_LINE_001.yaml").write_text(
        yaml.safe_dump(old, sort_keys=False)
    )
    _write_pending(
        p,
        _rule_yaml(
            "SURFACE_PRINT_LINE_002", statement="Print lines under 1cm may still gem."
        ),
    )
    monkeypatch.setattr(cli, "paths", lambda: p)

    result = runner.invoke(app, ["review"], input="supersede SURFACE_PRINT_LINE_001\n")

    assert result.exit_code == 0, result.output
    assert "0 accepted, 1 superseded" in result.output
    assert ver.read(p) == "1.0.0"  # a supersede is a major bump
    old_stored = yaml.safe_load(
        (p.rules / "surface" / "SURFACE_PRINT_LINE_001.yaml").read_text()
    )
    assert old_stored["status"] == "superseded"
    new_stored = yaml.safe_load(
        (p.rules / "surface" / "SURFACE_PRINT_LINE_002.yaml").read_text()
    )
    assert new_stored["status"] == "active"
    assert new_stored["supersedes"] == "SURFACE_PRINT_LINE_001"


def test_review_bad_supersede_id_defers_without_corrupting_version(tmp_path, monkeypatch):
    """Important 6: a mistyped supersede id must be caught at prompt time,
    before anything in the session is applied — not after the accept loop
    has already stamped rules with a version that then never gets written,
    permanently stranding RUBRIC_VERSION behind what's on those rules."""
    from card_reviewer.knowledge import version as ver

    p = _review_project(tmp_path)
    _write_pending(p, _rule_yaml())
    monkeypatch.setattr(cli, "paths", lambda: p)

    result = runner.invoke(app, ["review"], input="supersede SURFACE_TYPO_999\n")

    assert result.exit_code == 0, result.output
    assert "deferring" in result.output
    assert "0 accepted, 0 superseded" in result.output
    assert ver.read(p) == "0.1.0"
    assert (p.pending_rules / "SURFACE_PRINT_LINE_001.yaml").exists()


# --- fix round 2, finding 3: review_cmd matched decisions by prefix
# (`choice.startswith("accept"|"reject"|"supersede")`), so a typo like
# "acept" silently deferred with no message, "rejectionable" would reject,
# and "accepted" would accept. The first whitespace-separated token must
# now match exactly.


def test_review_typo_choice_defers_with_a_message(tmp_path, monkeypatch):
    from card_reviewer.knowledge import validate
    from card_reviewer.knowledge import version as ver

    p = _review_project(tmp_path)
    _write_pending(p, _rule_yaml())
    monkeypatch.setattr(cli, "paths", lambda: p)

    result = runner.invoke(app, ["review"], input="acept\n")

    assert result.exit_code == 0, result.output
    assert "0 accepted, 0 superseded" in result.output
    assert "acept" in result.output  # names what was typed
    assert ver.read(p) == "0.1.0"
    assert validate.load_active(p) == []
    assert (p.pending_rules / "SURFACE_PRINT_LINE_001.yaml").exists()


def test_review_choice_that_merely_starts_with_reject_does_not_reject(tmp_path, monkeypatch):
    """'rejectionable' must not match 'reject' by prefix."""
    from card_reviewer.knowledge import validate

    p = _review_project(tmp_path)
    _write_pending(p, _rule_yaml())
    monkeypatch.setattr(cli, "paths", lambda: p)

    result = runner.invoke(app, ["review"], input="rejectionable\n")

    assert result.exit_code == 0, result.output
    assert "0 accepted, 0 superseded" in result.output
    assert "rejectionable" in result.output
    assert validate.load_active(p) == []
    assert (p.pending_rules / "SURFACE_PRINT_LINE_001.yaml").exists()


def test_review_choice_that_merely_starts_with_accept_does_not_accept(tmp_path, monkeypatch):
    """'accepted' must not match 'accept' by prefix."""
    from card_reviewer.knowledge import validate

    p = _review_project(tmp_path)
    _write_pending(p, _rule_yaml())
    monkeypatch.setattr(cli, "paths", lambda: p)

    result = runner.invoke(app, ["review"], input="accepted\n")

    assert result.exit_code == 0, result.output
    assert "0 accepted, 0 superseded" in result.output
    assert validate.load_active(p) == []
    assert (p.pending_rules / "SURFACE_PRINT_LINE_001.yaml").exists()


def test_review_exact_accept_still_accepts(tmp_path, monkeypatch):
    from card_reviewer.knowledge import validate

    p = _review_project(tmp_path)
    _write_pending(p, _rule_yaml())
    monkeypatch.setattr(cli, "paths", lambda: p)

    result = runner.invoke(app, ["review"], input="accept\n")

    assert result.exit_code == 0, result.output
    assert "1 accepted, 0 superseded" in result.output
    assert [r.id for r in validate.load_active(p)] == ["SURFACE_PRINT_LINE_001"]


def test_review_bare_enter_still_defers_silently_with_no_unrecognised_message(
    tmp_path, monkeypatch
):
    p = _review_project(tmp_path)
    _write_pending(p, _rule_yaml())
    monkeypatch.setattr(cli, "paths", lambda: p)

    result = runner.invoke(app, ["review"], input="\n")

    assert result.exit_code == 0, result.output
    assert "0 accepted, 0 superseded" in result.output
    assert "unrecognised" not in result.output.lower()
    assert (p.pending_rules / "SURFACE_PRINT_LINE_001.yaml").exists()


# --- A10: two pending rules superseding the same active id in one review
# session both pass the prompt-time `find_active` check (the target is still
# active when each is prompted), but the second's application-phase
# `promote.supersede` would find it already retired. The second target must
# be rejected at prompt time instead.


def test_review_second_supersede_of_the_same_target_is_rejected_at_prompt_time(
    tmp_path, monkeypatch
):
    import yaml

    from card_reviewer.knowledge import version as ver

    p = _review_project(tmp_path)
    old = _rule_yaml(status="active", rubric_version_added="0.1.0")
    (p.rules / "surface").mkdir(parents=True)
    (p.rules / "surface" / "SURFACE_PRINT_LINE_001.yaml").write_text(
        yaml.safe_dump(old, sort_keys=False)
    )
    _write_pending(
        p, _rule_yaml("SURFACE_PRINT_LINE_002", statement="A refinement, worded one way.")
    )
    _write_pending(
        p,
        _rule_yaml(
            "SURFACE_PRINT_LINE_003", statement="A refinement, worded a second, distinct way."
        ),
    )
    monkeypatch.setattr(cli, "paths", lambda: p)

    result = runner.invoke(
        app,
        ["review"],
        input=(
            "supersede SURFACE_PRINT_LINE_001\n"
            "supersede SURFACE_PRINT_LINE_001\n"
        ),
    )

    assert result.exit_code == 0, result.output
    assert "already targeted" in result.output.lower()
    assert "0 accepted, 1 superseded" in result.output
    assert ver.read(p) == "1.0.0"
    old_stored = yaml.safe_load(
        (p.rules / "surface" / "SURFACE_PRINT_LINE_001.yaml").read_text()
    )
    assert old_stored["status"] == "superseded"
    # The second pending rule was deferred at prompt time -- it must still be
    # sitting in pending_rules/ for a later session, not lost or half-applied.
    assert (p.pending_rules / "SURFACE_PRINT_LINE_003.yaml").exists()


# --- A1: `validate` and `build-rubric` were not wrapped in
# `_friendly_errors` -- every other command is.


def test_validate_cmd_reports_domain_errors_cleanly_instead_of_a_traceback(
    tmp_path, monkeypatch
):
    from card_reviewer.knowledge import validate as val

    def boom(paths):
        raise RuntimeError("some_broken_file.yaml: does not parse as a Rule")

    monkeypatch.setattr(val, "run", boom)
    result = runner.invoke(app, ["validate"])

    assert result.exit_code == 1
    assert not isinstance(result.exception, RuntimeError)
    assert "some_broken_file.yaml" in result.output


def test_build_rubric_cmd_reports_corrupt_active_rule_cleanly(tmp_path, monkeypatch):
    from card_reviewer.knowledge import version as ver
    from card_reviewer.knowledge.paths import ProjectPaths

    p = ProjectPaths(tmp_path)
    bad_dir = p.rules / "surface"
    bad_dir.mkdir(parents=True)
    bad_file = bad_dir / "BROKEN_001.yaml"
    bad_file.write_text("sources: [not, closed\n")
    ver.write(p, "0.1.0")
    monkeypatch.setattr(cli, "paths", lambda: p)

    result = runner.invoke(app, ["build-rubric"])

    assert result.exit_code == 1
    assert not isinstance(result.exception, RuntimeError)
    assert "BROKEN_001" in result.output


def test_build_rubric_cmd_reports_rubric_error_cleanly(tmp_path, monkeypatch):
    """Even though build_rubric_cmd no longer calls load_active_rubric
    directly (A13), RubricError is still part of the public contract
    _friendly_errors documents handling -- guard it directly against the
    wrapper rather than relying on it never being reachable."""
    from card_reviewer.knowledge import rubric as rb

    def boom(paths):
        raise rb.RubricError("knowledge/rules/surface/BROKEN_001.yaml: does not parse")

    monkeypatch.setattr(rb, "build", boom)
    result = runner.invoke(app, ["build-rubric"])

    assert result.exit_code == 1
    assert not isinstance(result.exception, rb.RubricError)
    assert "BROKEN_001" in result.output
