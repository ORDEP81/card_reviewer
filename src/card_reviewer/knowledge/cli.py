"""Typer wiring. No business logic lives here."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import doctor
from .paths import ProjectPaths

app = typer.Typer(
    help="Turn grading training videos into a versioned knowledge base.",
    invoke_without_command=True,
)
console = Console()


# When a Typer app has only one command registered, Typer collapses the app
# into a single-command interface (Usage: doctor [OPTIONS]) rather than showing
# subcommands. The callback with invoke_without_command=True forces Typer to
# treat the registered commands as true subcommands, enabling proper multi-command
# routing. Without this callback, invoking ["doctor"] fails with exit code 2.
@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


def project_root() -> Path:
    """The repository root: three parents up from this file."""
    return Path(__file__).resolve().parents[3]


def paths() -> ProjectPaths:
    return ProjectPaths(project_root())


def doctor_cmd() -> None:
    """Check that yt-dlp, ffmpeg, and mlx-whisper are available."""
    checks = doctor.check_all()
    table = Table("tool", "status", "version", "install")
    for c in checks:
        table.add_row(
            c.name,
            "[green]ok[/green]" if c.found else "[red]missing[/red]",
            c.version or "",
            "" if c.found else c.install_hint,
        )
    console.print(table)
    if not all(c.found for c in checks):
        raise typer.Exit(code=1)


# Typer would derive "doctor-cmd" from the function name, so register the
# command explicitly. Every later command uses the @app.command(name="...")
# decorator form instead; this one is written out to show the equivalence.
app.command(name="doctor")(doctor_cmd)


@app.command(name="acquire")
def acquire_cmd(
    url: str | None = typer.Argument(None, help="Video URL"),
    file: Path | None = typer.Option(None, "--file", help="Local video file"),
    browser: str | None = typer.Option(
        None,
        "--browser",
        help="Read cookies from this browser: chrome, brave, edge, firefox, safari",
    ),
) -> None:
    """Download a video (or adopt a local file) and open its work packet."""
    from . import acquire as acq
    from . import version as ver

    p = paths()
    rubric_version = ver.read(p)
    try:
        m = acq.from_file(p, file, rubric_version) if file else acq.from_url(
            p, url, rubric_version, browser=browser
        )
    except acq.AcquisitionFailed as exc:
        console.print(f"[red]Acquisition failed:[/red] {exc}")
        console.print(exc.guidance)
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Packet ready:[/green] {m.video_id} — {m.source.title}")
