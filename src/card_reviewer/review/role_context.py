"""Role resolution and card-context normalization, as one cached stage.

They are resolved from the same inputs and consumed together, so they cache
together. This lives in its own module rather than in `roles.py`: it imports
`imaging.role_features`, and `policies/coverage_v1` imports `ImageRole` from
`roles`, so putting it there would drag NumPy into the policy layer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .context import CardContext
from .imaging.role_features import RoleFeatures
from .models import ResolvedCandidate
from .normalize import CardContextNormalizer
from .roles import ResolvedRole, RoleInput, resolve_roles
from .versions import RESOLVER_VERSION

__all__ = ["RESOLVER_VERSION", "RoleContext", "resolve_role_context"]


class RoleContext(BaseModel):
    roles: dict[str, ResolvedRole] = Field(default_factory=dict)
    card_context: CardContext = Field(default_factory=CardContext)
    version: str = RESOLVER_VERSION


def resolve_role_context(
    candidate: ResolvedCandidate, image_outputs: list
) -> RoleContext:
    """The cached `role_context` stage.

    Role features come from the `role_features` stage, never from defect
    measurements — an image whose geometry failed has no features, and is
    left `unknown` rather than assigned a face.
    """
    supplied = {i.image_hash: i.supplied_role for i in candidate.images}
    inputs = []
    for output in image_outputs:
        features = RoleFeatures.model_validate(output.role_features or {})
        inputs.append(
            RoleInput(
                image_hash=output.image_hash,
                supplied_role=supplied.get(output.image_hash),
                text_density=features.text_density,
                has_central_image_region=features.has_central_image_region,
            )
        )

    return RoleContext(
        roles=resolve_roles(inputs),
        card_context=CardContextNormalizer().normalize(
            raw_title=candidate.title,
            supplied_card_type=candidate.card_type,
            supplied_set=candidate.set_name,
        ),
    )
