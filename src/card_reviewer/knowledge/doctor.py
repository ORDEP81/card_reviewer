"""Preflight checks for the external tools the pipeline shells out to."""

from __future__ import annotations

import importlib.util
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

EXTERNAL_TOOLS = [
    ("yt-dlp", ["--version"], "brew install yt-dlp"),
    ("ffmpeg", ["-version"], "brew install ffmpeg"),
]

PYTHON_MODULES = [("mlx-whisper", "mlx_whisper", "uv sync")]


@dataclass
class ToolCheck:
    name: str
    found: bool
    version: str | None
    install_hint: str


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_all(
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    has_module: Callable[[str], bool] = _has_module,
) -> list[ToolCheck]:
    checks: list[ToolCheck] = []

    for name, args, hint in EXTERNAL_TOOLS:
        try:
            proc = runner([name, *args], capture_output=True, text=True, check=False)
        except (FileNotFoundError, OSError):
            checks.append(ToolCheck(name, False, None, hint))
            continue
        first_line = (proc.stdout or "").strip().splitlines()
        checks.append(
            ToolCheck(name, True, first_line[0] if first_line else None, hint)
        )

    for display, module, hint in PYTHON_MODULES:
        checks.append(ToolCheck(display, has_module(module), None, hint))

    return checks
