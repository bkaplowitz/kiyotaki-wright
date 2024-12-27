from __future__ import annotations

from typing import Final, TypeVar, final

import torch

import vmas.simulator.core
import vmas.simulator.utils
from vmas.simulator.rendering import Geom

UNCOLLIDABLE_JOINT_RENDERING_WIDTH: Final[int] = 1

Entity = TypeVar("Entity", bound=vmas.simulator.core.Entity)

class Joint(vmas.simulator.utils.Observer):
    entity_a: vmas.simulator.core.Entity
    entity_b: vmas.simulator.core.Entity
    rotate_a: bool
    rotate_b: bool
    fixed_rotation_a: float | None
    fixed_rotation_b: float | None
    landmark: vmas.simulator.core.Landmark | None
    joint_constraints: list[JointConstraint]

    def __init__(
        self,
        entity_a: vmas.simulator.core.Entity,
        entity_b: vmas.simulator.core.Entity,
        anchor_a: tuple[float, float] = (0.0, 0.0),
        anchor_b: tuple[float, float] = (0.0, 0.0),
        rotate_a: bool = True,
        rotate_b: bool = True,
        dist: float = 0.0,
        collidable: bool = False,
        width: float = 0.0,
        mass: float = 1.0,
        fixed_rotation_a: float | None = None,
        fixed_rotation_b: float | None = None,
    ) -> None: ...
    def notify(self, observable: object, *args: object, **kwargs: object) -> None: ...

@final
class JointConstraint:
    entity_a: vmas.simulator.core.Entity
    entity_b: vmas.simulator.core.Entity
    anchor_a: tuple[float, float]
    anchor_b: tuple[float, float]
    dist: float
    fixed_rotation: float | None
    rotate: bool
    _delta_anchor_tensor_map: dict[vmas.simulator.core.Entity, torch.Tensor]

    def __init__(
        self,
        entity_a: vmas.simulator.core.Entity,
        entity_b: vmas.simulator.core.Entity,
        anchor_a: tuple[float, float] = (0.0, 0.0),
        anchor_b: tuple[float, float] = (0.0, 0.0),
        dist: float = 0.0,
        rotate: bool = True,
        fixed_rotation: float | None = None,
    ) -> None: ...
    def _delta_anchor_tensor(
        self, entity: vmas.simulator.core.Entity
    ) -> torch.Tensor: ...
    def get_delta_anchor(self, entity: vmas.simulator.core.Entity) -> torch.Tensor: ...
    def pos_point(self, entity: vmas.simulator.core.Entity) -> torch.Tensor: ...
    def render(self, env_index: int = 0) -> list[Geom]: ...
