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
    no_args_is_help=True,
)
console = Console()


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print(app.get_help(ctx))


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
