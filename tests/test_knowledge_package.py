"""fix round 2, finding 4: `load_active_rubric` is the one named contract
subsystem A imports, and it did not exist at the package root -- a caller
had to already know it actually lives in `.rubric`. It must be re-exported,
and re-exporting it must not make importing the package expensive: several
sibling modules deliberately keep `mlx_whisper`/`PIL`/`imagehash` behind
lazy imports, and pulling those in at `import card_reviewer.knowledge` time
would defeat that.
"""

from __future__ import annotations

import subprocess
import sys


def test_load_active_rubric_is_importable_from_the_package_root():
    from card_reviewer.knowledge import Rubric, RubricError, load_active_rubric

    assert callable(load_active_rubric)
    assert isinstance(RubricError, type) and issubclass(RubricError, Exception)
    assert Rubric is not None


def test_package_all_names_the_public_contract():
    import card_reviewer.knowledge as pkg

    assert set(pkg.__all__) == {"Rubric", "RubricError", "load_active_rubric"}


def test_importing_knowledge_package_does_not_load_heavy_dependencies():
    """Run in a fresh interpreter: importing the package must not, itself,
    pull mlx_whisper/PIL/imagehash into sys.modules. A same-process check
    would be contaminated by whatever other tests already imported lazily.
    """
    script = (
        "import sys\n"
        "import card_reviewer.knowledge\n"
        "heavy = {'mlx_whisper', 'PIL', 'imagehash'}\n"
        "loaded = heavy & set(sys.modules)\n"
        "assert not loaded, f'unexpectedly loaded: {loaded}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
