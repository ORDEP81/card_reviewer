"""Stage 1: get the media onto disk and open a work packet.

Three sources, one stage. Authenticated failure is terminal by design: see
spec §9 and CARD_REVIEWER_BUILD_PLAN §4 and §30 rule 13. There is deliberately
no retry, no alternate extractor, and no player workaround in this module.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import urllib.parse
from collections.abc import Callable
from pathlib import Path

from . import manifest as mf
from .models import FileInfo, Manifest, SourceInfo
from .paths import ProjectPaths

Runner = Callable[..., subprocess.CompletedProcess]

MANUAL_FALLBACK = (
    "If this is protected course material you have access to, play the lesson "
    "in your browser, save the video yourself, then run:\n"
    "  card-knowledge acquire --file /path/to/lesson.mp4"
)


class AcquisitionFailed(Exception):
    def __init__(self, message: str, guidance: str = MANUAL_FALLBACK) -> None:
        super().__init__(message)
        self.guidance = guidance


# Real YouTube video ids are exactly 11 chars from this alphabet. Anything
# extracted from the URL that doesn't match is untrusted input (e.g. a
# path-traversal-shaped `v=` parameter) and must not be used to build a
# filesystem path — fall back to hashing the whole URL instead.
_SAFE_YT_ID = re.compile(r"[A-Za-z0-9_-]{1,32}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_video_id(url: str | None = None, file: Path | None = None) -> str:
    """A stable, filesystem-safe id so re-running never duplicates work."""
    if file is not None:
        return f"local_{_sha256_file(Path(file))[:12]}"
    if url is None:
        raise ValueError("derive_video_id requires either url or file")

    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    if "youtube.com" in host:
        qs = urllib.parse.parse_qs(parsed.query)
        if "v" in qs and _SAFE_YT_ID.fullmatch(qs["v"][0]):
            return f"yt_{qs['v'][0]}"
    if "youtu.be" in host:
        candidate = parsed.path.lstrip("/")
        if _SAFE_YT_ID.fullmatch(candidate):
            return f"yt_{candidate}"

    prefix = "skool" if "skool.com" in host else "web"
    return f"{prefix}_{hashlib.sha256(url.encode()).hexdigest()[:12]}"


def _probe_duration(path: Path, runner: Runner) -> float:
    proc = runner(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float((proc.stdout or "0").strip())
    except ValueError:
        return 0.0


def _cookie_args(browser: str | None) -> list[str]:
    return ["--cookies-from-browser", browser] if browser else []


def _open_manifest(
    paths: ProjectPaths, video_id: str, source: SourceInfo, rubric_version: str
) -> Manifest:
    """Start a fresh manifest, or refresh an existing one in place.

    A packet is resumable work: re-running `acquire` for a `video_id` that
    already has a manifest must not reset `stages` or `lesson_id` for
    stages that already ran. Only `source` (and, by the caller, `file`) may
    legitimately change on a re-acquire — everything else about the packet
    carries forward untouched.
    """
    if paths.manifest(video_id).exists():
        m = mf.load(paths, video_id)
        m.source = source
        return m
    return Manifest(video_id=video_id, source=source, rubric_version_at_ingest=rubric_version)


def from_url(
    paths: ProjectPaths,
    url: str,
    rubric_version: str,
    browser: str | None = None,
    runner: Runner = subprocess.run,
) -> Manifest:
    video_id = derive_video_id(url=url)
    dest = paths.source_dir(video_id)
    dest.mkdir(parents=True, exist_ok=True)

    meta_proc = runner(
        ["yt-dlp", "--dump-json", "--no-warnings", *_cookie_args(browser), url],
        capture_output=True,
        text=True,
        check=False,
    )
    if meta_proc.returncode != 0:
        error = (meta_proc.stderr or "").strip()
        # We don't know the title or duration yet — this is a placeholder record
        # of the attempt and its failure reason, not a completed packet. A later
        # successful run overwrites it cleanly (same video_id, same manifest path).
        placeholder = SourceInfo(
            type="skool" if video_id.startswith("skool_") else "youtube",
            url=url,
            title=video_id,
            uploader=None,
            duration_s=0.0,
        )
        m = _open_manifest(paths, video_id, placeholder, rubric_version)
        mf.save(paths, m)
        mf.start(paths, m, "acquire")
        mf.fail(paths, m, "acquire", error)
        raise AcquisitionFailed(f"yt-dlp could not read {url}: {error}")

    meta = json.loads(meta_proc.stdout)
    source = SourceInfo(
        type="skool" if video_id.startswith("skool_") else "youtube",
        url=url,
        title=meta.get("title", video_id),
        uploader=meta.get("uploader"),
        duration_s=float(meta.get("duration") or 0),
    )

    m = _open_manifest(paths, video_id, source, rubric_version)
    mf.save(paths, m)
    mf.start(paths, m, "acquire")

    out_template = str(dest / "video.%(ext)s")
    dl = runner(
        [
            "yt-dlp",
            "-f", "bv*+ba/b",
            *_cookie_args(browser),
            "-o", out_template,
            url,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    downloaded = sorted(dest.glob("video.*"))
    if dl.returncode != 0 or not downloaded:
        error = (dl.stderr or "yt-dlp produced no file").strip()
        mf.fail(paths, m, "acquire", error)
        raise AcquisitionFailed(f"download failed for {url}: {error}")

    path = downloaded[0]
    m.file = FileInfo(
        path=str(path.relative_to(paths.packet(video_id))),
        sha256=_sha256_file(path),
        bytes=path.stat().st_size,
    )
    return mf.finish(paths, m, "acquire", tool="yt-dlp", browser=browser)


def from_file(
    paths: ProjectPaths,
    file: Path | str,
    rubric_version: str,
    runner: Runner = subprocess.run,
) -> Manifest:
    src = Path(file)
    if not src.exists():
        raise AcquisitionFailed(f"no such file: {src}", guidance="Check the path.")

    video_id = derive_video_id(file=src)
    dest_dir = paths.source_dir(video_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"video{src.suffix}"
    if not dest.exists():
        shutil.copy2(src, dest)

    source = SourceInfo(
        type="local",
        url=None,
        title=src.stem,
        uploader=None,
        duration_s=_probe_duration(dest, runner),
    )
    m = _open_manifest(paths, video_id, source, rubric_version)
    m.file = FileInfo(
        path=str(dest.relative_to(paths.packet(video_id))),
        sha256=_sha256_file(dest),
        bytes=dest.stat().st_size,
    )
    mf.save(paths, m)
    return mf.finish(paths, m, "acquire", tool="local-copy", original=str(src))
