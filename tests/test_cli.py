from typer.testing import CliRunner

from card_reviewer.knowledge.cli import app

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
