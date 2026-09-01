"""Card review CLI.

A surface only. No thresholds, no verdict logic, no scoring — those live in
the versioned policies, and a test asserts this module contains none of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .enums import Mode
from .models import CandidateInput
from .report import render
from .service import build_provider, open_context, review_card

app = typer.Typer(help="Card review engine — screening, not final judgement.",
                  no_args_is_help=True)

DATA_DIR = typer.Option("data", "--data-dir", help="Where reviews are stored.")


@app.command()
def screen(
    paths: list[Path] = typer.Argument(..., help="Card photographs."),
    mode: Mode = typer.Option(Mode.SMART, "--mode", case_sensitive=False,
                              help="off / smart / deep."),
    title: str = typer.Option("", "--title", help="Listing title, if any."),
    data_dir: Path = DATA_DIR,
    require_provider: bool = typer.Option(
        False, "--require-provider",
        help="Fail rather than degrade when vision is wanted but "
             "unconfigured."),
) -> None:
    """Screen one card and print its review."""
    with open_context(data_dir) as context:
        provider = None if mode is Mode.OFF else build_provider(context.store)
        if require_provider and mode is not Mode.OFF and provider is None:
            raise typer.BadParameter(
                "vision was required but ANTHROPIC_API_KEY is not set")
        review = review_card(
            CandidateInput(source="manual", title=title,
                           image_paths=list(paths)),
            mode=mode, provider=provider, context=context)
    typer.echo(render(review))


@app.command()
def show(review_id: int, data_dir: Path = DATA_DIR) -> None:
    """Render a stored review."""
    with open_context(data_dir) as context:
        typer.echo(render(_load(context, review_id)))


@app.command()
def export(review_id: int, data_dir: Path = DATA_DIR) -> None:
    """Emit a stored review as JSON, for interchange rather than storage."""
    with open_context(data_dir) as context:
        typer.echo(_load(context, review_id).model_dump_json(indent=2))


@app.command()
def outcome(
    review_id: int,
    grade: str = typer.Option(..., "--grade", help="The grade PSA returned."),
    cert: str = typer.Option("", "--cert", help="Certification number."),
    grader: str = typer.Option("PSA", "--grader"),
    data_dir: Path = DATA_DIR,
) -> None:
    """Record a grading result against the review that predicted it.

    Append-only: a card can be cracked and resubmitted, so a second outcome
    never overwrites the first.
    """
    with open_context(data_dir) as context:
        review = _load(context, review_id)
        context.repo.record_grading_outcome(
            review.candidate_id, grade, grader=grader, cert_number=cert)
    typer.echo(f"recorded {grader} {grade} against review {review_id}")


@app.command(name="provider-smoke")
def provider_smoke(
    image: Path = typer.Argument(..., help="One card photograph."),
    data_dir: Path = DATA_DIR,
) -> None:
    """Make ONE real provider call. Manual only — never run in CI."""
    with open_context(data_dir) as context:
        provider = build_provider(context.store)
        if provider is None:
            raise typer.BadParameter("ANTHROPIC_API_KEY is not set")
        review = review_card(
            CandidateInput(source="manual", title="smoke test",
                           image_paths=[image]),
            mode=Mode.DEEP, provider=provider, context=context)
    typer.echo(render(review))


def _load(context, review_id: int):
    from .models import CardReview

    row = context.repo.get_review(review_id)
    if row is None:
        raise typer.BadParameter(f"no review with id {review_id}")
    return CardReview.model_validate(json.loads(row["output_json"]))
