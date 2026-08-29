"""Stage 4: pull frames for the top-ranked segments.

Deduplication matters more than it sounds: a static talking head at 1 fps
produces twenty copies of one image and buries the frames that show a card.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from . import manifest as mf
from .models import Segment
from .paths import ProjectPaths

Runner = Callable[..., subprocess.CompletedProcess]

TOP_N = 12
FPS = 1.0
CAP_PER_SEGMENT = 20
PHASH_THRESHOLD = 5


def _default_hasher(path: Path):  # pragma: no cover - needs a real image
    import imagehash
    from PIL import Image

    return imagehash.phash(Image.open(path))


def sample(
    video: Path,
    out_dir: Path,
    start_s: float,
    end_s: float,
    fps: float = FPS,
    cap: int = CAP_PER_SEGMENT,
    runner: Runner = subprocess.run,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end_s - start_s)
    proc = runner(
        [
            "ffmpeg", "-v", "error", "-y",
            "-ss", str(start_s),
            "-t", str(duration),
            "-i", str(video),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            str(out_dir / "frame_%04d.jpg"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {(proc.stderr or '').strip()}")

    produced = sorted(out_dir.glob("frame_*.jpg"))
    for extra in produced[cap:]:
        extra.unlink()
    return produced[:cap]


def dedupe(
    image_paths: list[Path],
    threshold: int = PHASH_THRESHOLD,
    hasher: Callable[[Path], object] | None = None,
) -> list[Path]:
    """Delete frames within `threshold` perceptual distance of a kept frame."""
    hasher = hasher or _default_hasher
    survivors: list[Path] = []
    kept_hashes: list[object] = []

    for path in image_paths:
        digest = hasher(path)
        if any(abs(digest - kept) <= threshold for kept in kept_hashes):
            path.unlink(missing_ok=True)
            continue
        survivors.append(path)
        kept_hashes.append(digest)

    return survivors


def run(
    paths: ProjectPaths,
    video_id: str,
    top_n: int = TOP_N,
    uniform: bool = False,
    at: float | None = None,
    window_s: float = 30.0,
    fps: float = FPS,
    runner: Runner = subprocess.run,
) -> int:
    m = mf.load(paths, video_id)
    mf.require_ready(m, "extract_frames")

    videos = sorted(paths.source_dir(video_id).glob("video.*"))
    if not videos:
        raise FileNotFoundError(f"no media in {paths.source_dir(video_id)}")
    video = videos[0]

    if at is not None:
        targets = [Segment(id="seg_adhoc", start_s=at, end_s=at + window_s, score=0.0)]
    elif uniform:
        step = max(30.0, m.source.duration_s / max(top_n, 1))
        targets = [
            Segment(id=f"seg_u{i:03d}", start_s=i * step, end_s=i * step + window_s, score=0.0)
            for i in range(top_n)
        ]
    else:
        data = json.loads(paths.segments(video_id).read_text())
        targets = [Segment(**s) for s in data["segments"]][:top_n]

    total = 0
    for seg in targets:
        out_dir = paths.frames(video_id) / seg.id
        produced = sample(video, out_dir, seg.start_s, seg.end_s, fps=fps, runner=runner)
        total += len(dedupe(produced))

    if at is None:
        mf.finish(paths, m, "extract_frames", n_frames=total, uniform=uniform, fps=fps)
    return total
