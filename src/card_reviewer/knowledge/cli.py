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


@app.command(name="transcribe")
def transcribe_cmd(
    video_id: str,
    browser: str | None = typer.Option(None, "--browser"),
) -> None:
    """Produce a timestamped transcript for a work packet."""
    from . import transcribe as tr

    t = tr.run(paths(), video_id, browser=browser)
    console.print(
        f"[green]Transcript:[/green] {len(t.cues)} cues via {t.method}"
    )


@app.command(name="segment")
def segment_cmd(video_id: str) -> None:
    """Rank the transcript into candidate windows worth inspecting."""
    from . import segment as seg

    segments = seg.run(paths(), video_id)
    console.print(f"[green]{len(segments)} segments[/green] written")
    for s in segments[:12]:
        console.print(
            f"  {s.id}  {s.start_s:8.1f}s-{s.end_s:8.1f}s  "
            f"score {s.score:6.1f}  {','.join(s.categories) or '-'}"
        )


@app.command(name="extract-frames")
def extract_frames_cmd(
    video_id: str,
    top_n: int = typer.Option(12, "--top-n"),
    uniform: bool = typer.Option(False, "--uniform", help="Ignore ranking; sample the whole video"),
    at: float | None = typer.Option(None, "--at", help="Ad-hoc window start, in seconds"),
    window: float = typer.Option(30.0, "--window", help="Ad-hoc window length, in seconds"),
) -> None:
    """Pull frames for the top-ranked segments, or for an ad-hoc window."""
    from . import frames as fr

    count = fr.run(paths(), video_id, top_n=top_n, uniform=uniform, at=at, window_s=window)
    console.print(f"[green]{count} frames[/green] kept after deduplication")


@app.command(name="validate")
def validate_cmd() -> None:
    """Check every pending rule for schema, citation, and status errors."""
    from . import validate as val

    report = val.run(paths())
    if report.ok:
        console.print(f"[green]{report.checked} pending rules valid[/green]")
        return
    for rule_id, errors in report.errors.items():
        console.print(f"[red]{rule_id}[/red]")
        for error in errors:
            console.print(f"  - {error}")
    raise typer.Exit(code=1)
