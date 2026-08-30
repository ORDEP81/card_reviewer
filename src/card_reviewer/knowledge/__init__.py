"""Video learning pipeline: videos to versioned grading rules.

`load_active_rubric` (plus the `Rubric` and `RubricError` types around it) is
the one named contract subsystem A imports. It is re-exported here so a
caller can write `from card_reviewer.knowledge import load_active_rubric`
instead of needing to know it actually lives in `.rubric`.

Importing this package must stay cheap: `.rubric` only pulls in `.validate`,
`.version`, `.models`, and `.paths`, none of which import `mlx_whisper`,
`PIL`, or `imagehash` at module level — those stay behind lazy imports inside
`transcribe.py` and `frames.py`. Keep it that way; do not add a top-level
import here of any sibling module that drags in a heavy dependency.
"""

from __future__ import annotations

from .rubric import Rubric, RubricError, load_active_rubric

__all__ = ["Rubric", "RubricError", "load_active_rubric"]
