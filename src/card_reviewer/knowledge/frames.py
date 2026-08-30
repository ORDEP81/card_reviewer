"""Stage 4: pull frames for the top-ranked segments.

Deduplication matters more than it sounds: a static talking head at 1 fps
produces twenty copies of one image and buries the frames that show a card.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from . import manifest as mf
from .acquire import select_media_file
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
    # A crashed prior attempt or a shorter re-run can leave frames behind that
    # ffmpeg's -y would not overwrite (it only touches indices it regenerates).
    # Clear them first so anything found afterward is from this invocation only.
    for stale in out_dir.glob("frame_*.jpg"):
        stale.unlink()
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

    video = select_media_file(paths.source_dir(video_id))
    if video is None:
        raise FileNotFoundError(f"no media in {paths.source_dir(video_id)}")

    if at is not None:
        # Named by timestamp (e.g. seg_at_754), not a single shared
        # "seg_adhoc": every --at run used to write to the same directory,
        # which `sample()` clears first, so a second ad-hoc pull silently
        # destroyed the first with no record of which timestamp it held.
        label = f"{at:g}".replace(".", "_").replace("-", "neg")
        targets = [Segment(id=f"seg_at_{label}", start_s=at, end_s=at + window_s, score=0.0)]
    elif uniform:
        # Stop once a window would start at or past the end of the video, and
        # clamp each window's end to the duration: a short video should yield
        # fewer than top_n windows rather than seeking past EOF into nothing.
        step = max(30.0, m.source.duration_s / max(top_n, 1))
        targets = []
        for i in range(top_n):
            start = i * step
            if start >= m.source.duration_s:
                break
            end = min(start + window_s, m.source.duration_s)
            targets.append(Segment(id=f"seg_u{i:03d}", start_s=start, end_s=end, score=0.0))
    else:
        data = json.loads(paths.segments(video_id).read_text())
        targets = [Segment(**s) for s in data["segments"]][:top_n]

    total = 0
    for seg in targets:
        out_dir = paths.frames(video_id) / seg.id
        produced = sample(video, out_dir, seg.start_s, seg.end_s, fps=fps, runner=runner)
        total += len(dedupe(produced))

    if at is None and targets:
        # Remove segment directories from a prior ranked/uniform run that
        # aren't part of *this* run (a smaller --top-n, or a re-segment that
        # produced fewer segments) -- otherwise they linger and look like
        # current output.
        #
        # Scoped strictly to `seg_*` names: anything else under frames/ (a
        # user's own notes, screenshots, whatever) is never a candidate for
        # removal, even if it isn't in `keep_ids`. Ad-hoc (`seg_adhoc`/
        # `seg_at_*`) directories are a separate, deliberately-persistent
        # namespace and are excluded too. A symlinked child is skipped
        # rather than removed: `shutil.rmtree` refuses a symlink to a
        # directory outright, and letting that raise would abort cleanup
        # partway through.
        #
        # Guarded on `targets` being non-empty: a run producing zero targets
        # (an empty segments.json, or --uniform against a duration_s of 0.0)
        # must be a no-op here, not a wipe of every existing seg_* directory.
        keep_ids = {seg.id for seg in targets}
        frames_dir = paths.frames(video_id)
        if frames_dir.exists():
            for existing in frames_dir.iterdir():
                if not existing.name.startswith("seg_"):
                    continue
                if existing.is_symlink() or not existing.is_dir():
                    continue
                if existing.name.startswith("seg_adhoc") or existing.name.startswith("seg_at_"):
                    continue
                if existing.name not in keep_ids:
                    shutil.rmtree(existing)
    if at is None:
        mf.finish(paths, m, "extract_frames", n_frames=total, uniform=uniform, fps=fps)
    return total
