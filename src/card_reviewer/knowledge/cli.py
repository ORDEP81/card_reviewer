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


@app.command(name="review")
def review_cmd() -> None:
    """Walk pending rules one at a time and decide each one."""
    from . import dedup, promote as pr, validate as val, version as ver

    p = paths()
    report = val.run(p)
    if not report.ok:
        console.print("[red]Fix validation errors before reviewing:[/red]")
        for rule_id, errors in report.errors.items():
            console.print(f"  {rule_id}: {'; '.join(errors)}")
        raise typer.Exit(code=1)

    pending = val.load_pending(p)
    if not pending:
        console.print("No pending rules.")
        return

    active = val.load_active(p)
    to_accept = []
    to_supersede = []

    for _, rule in pending:
        console.rule(f"[bold]{rule.id}[/bold]  ({rule.category.value})")
        console.print(f"[bold]{rule.statement}[/bold]")
        console.print(
            f"evidence: {rule.evidence_type.value}   confidence: {rule.confidence.value}"
        )
        if rule.applies_to.card_types or rule.applies_to.sets:
            console.print(
                f"applies to: {rule.applies_to.card_types or '-'} / {rule.applies_to.sets or '-'}"
            )
        for source in rule.sources:
            console.print(
                f"  [dim]{source.lesson} {source.video_id} {','.join(source.timestamps)}[/dim]"
            )
            if source.quote:
                console.print(f'    "{source.quote}"')

        for flag in dedup.flags_for(rule, active):
            colour = "red" if flag.kind == "contradiction" else "yellow"
            console.print(
                f"  [{colour}]{flag.kind}[/{colour}] vs {flag.other_id} "
                f"({flag.score}): {flag.other_statement}"
            )

        choice = typer.prompt(
            "accept / reject / supersede <ID> / defer", default="defer"
        ).strip()

        if choice.startswith("accept"):
            to_accept.append(rule)
        elif choice.startswith("reject"):
            reason = typer.prompt("reason")
            pr.reject(p, rule, reason)
        elif choice.startswith("supersede"):
            parts = choice.split(maxsplit=1)
            if len(parts) != 2:
                console.print("[red]supersede needs a rule id; deferring[/red]")
                continue
            to_supersede.append((rule, parts[1].strip()))

    # One version for the whole session: every accepted or superseded rule is
    # stamped with the same value, and the version file is written at most
    # once — never a mid-session read-bump-write per decision, which would
    # let two decisions in one session stamp two different "current" versions
    # that never both existed on disk.
    level = pr.session_bump_level(bool(to_accept), bool(to_supersede))
    new_version = ver.bump(ver.read(p), level) if level else None

    for rule in to_accept:
        pr.accept(p, rule, new_version)
    for rule, old_id in to_supersede:
        pr.supersede(p, rule, old_id, new_version)

    if new_version:
        ver.write(p, new_version)

    console.print(
        f"[green]{len(to_accept)} accepted, {len(to_supersede)} superseded.[/green] "
        f"Rubric now {ver.read(p)}. Run `card-knowledge build-rubric`."
    )


@app.command(name="build-rubric")
def build_rubric_cmd() -> None:
    """Render ACTIVE_RUBRIC.md from the active rule files."""
    from . import rubric as rb

    path = rb.build(paths())
    r = rb.load_active_rubric(project_root())
    console.print(f"[green]Wrote[/green] {path} — v{r.version}, {len(r.rules)} rules")


@app.command(name="run")
def run_cmd(
    url: str | None = typer.Argument(None),
    file: Path | None = typer.Option(None, "--file"),
    browser: str | None = typer.Option(None, "--browser"),
    top_n: int = typer.Option(12, "--top-n"),
    force: bool = typer.Option(False, "--force", help="Re-run stages already done"),
) -> None:
    """Advance a video through every deterministic stage, stopping at analyze."""
    from . import acquire as acq
    from . import pipeline as pl

    try:
        m = pl.run_all(paths(), url=url, file=file, browser=browser, top_n=top_n, force=force)
    except acq.AcquisitionFailed as exc:
        console.print(f"[red]Acquisition failed:[/red] {exc}")
        console.print(exc.guidance)
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Packet ready for analysis:[/green] {m.video_id}")
    console.print(
        "Next: start an interactive Claude Code session and invoke the "
        f"learn-video skill on {m.video_id}."
    )


@app.command(name="status")
def status_cmd(video_id: str | None = typer.Argument(None)) -> None:
    """Show the stage state of one packet, or of every packet."""
    from . import pipeline as pl
    from .models import STAGES

    manifests = pl.status(paths(), video_id)
    if not manifests:
        console.print("No work packets.")
        return

    table = Table("video_id", "title", *STAGES)
    for m in manifests:
        marks = []
        for stage in STAGES:
            state = m.stages[stage].status.value
            marks.append(
                {"done": "[green]done[/green]", "failed": "[red]failed[/red]"}.get(
                    state, state
                )
            )
        table.add_row(m.video_id, m.source.title[:30], *marks)
    console.print(table)
