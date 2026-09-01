"""Offline contract tests for the Anthropic provider.

No test here makes a network call. The request is constructed and inspected;
responses come from saved fixtures.
"""

import json
from pathlib import Path

import pytest

from card_reviewer.review.imaging.synthetic import CardSpec, render_png
from card_reviewer.review.storage.artifacts import ArtifactStore
from card_reviewer.review.vision.anthropic import AnthropicVisionProvider, build_request
from card_reviewer.review.vision.prompt import PROMPT_VERSION, build_prompt
from card_reviewer.review.vision.provider import ProviderContractError

FIXTURES = Path(__file__).parent / "fixtures" / "vision"


def _fixture(name, artifact_id="a1"):
    return json.loads(FIXTURES.joinpath(name).read_text()
                      .replace("__ARTIFACT__", artifact_id))


@pytest.fixture
def rig(tmp_path):
    """A provider whose store actually holds the artifact the manifest cites."""
    store = ArtifactStore(tmp_path)
    image_hash = store.put_image(render_png(CardSpec()))
    artifact_id = store.put_derived(image_hash, "surface", "original.png",
                                    render_png(CardSpec()))
    provider = AnthropicVisionProvider(model="m", store=store, api_key="unused")
    manifest = {
        "artifacts": [{"artifact_id": artifact_id, "view": "surface_original",
                       "origin": "normalized", "enhancement": None,
                       "region": None}],
        "rubric_rules": [], "measurements": {},
    }
    return provider, manifest, artifact_id


# --- the prompt ------------------------------------------------------------

def test_no_test_in_this_suite_constructs_a_live_client():
    """Delegates to the real guard in test_no_live_api.py.

    This copy had the same two holes as the one in test_definition_of_done:
    a relative path whose absence read as success, and a pattern that missed
    the plain construction.
    """
    from test_no_live_api import CONSTRUCTORS

    # Assembled at runtime, so this line is not itself an offender.
    assert CONSTRUCTORS.search("client = AsyncAnthro" + "pic()")
