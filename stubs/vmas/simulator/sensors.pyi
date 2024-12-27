from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

import vmas.simulator.core
from vmas.simulator.rendering import Geom
from vmas.simulator.utils import Color

class Sensor(ABC):
    _world: vmas.simulator.core.World
    _agent: vmas.simulator.core.Agent | None

    def __init__(self, world: vmas.simulator.core.World) -> None: ...
    @property
    def agent(self) -> vmas.simulator.core.Agent | None: ...
    @agent.setter
    def agent(self, agent: vmas.simulator.core.Agent) -> None: ...
    @abstractmethod
    def measure(self, vectorized: bool = True) -> torch.Tensor: ...
    @abstractmethod
    def render(self, env_index: int = 0) -> list[Geom]: ...
    def to(self, device: torch.device) -> None: ...

class Lidar(Sensor):
    _angles: torch.Tensor
    _max_range: float
    _last_measurement: torch.Tensor | None
    _render: bool
    _entity_filter: Callable[[vmas.simulator.core.Entity], bool]
    _render_color: Color | tuple[float, float, float]
    _alpha: float

    def __init__(
        self,
        world: vmas.simulator.core.World,
        angle_start: float = 0.0,
        angle_end: float = 2 * torch.pi,
        n_rays: int = 8,
        max_range: float = 1.0,
        entity_filter: Callable[[vmas.simulator.core.Entity], bool] = ...,
        render_color: Color | tuple[float, float, float] = Color.GRAY,
        alpha: float = 1.0,
        render: bool = True,
    ) -> None: ...
    def to(self, device: torch.device) -> None: ...
    @property
    def entity_filter(self) -> Callable[[vmas.simulator.core.Entity], bool]: ...
    @entity_filter.setter
    def entity_filter(
        self, entity_filter: Callable[[vmas.simulator.core.Entity], bool]
    ) -> None: ...
    @property
    def render_color(self) -> tuple[float, float, float] | Color: ...
    @property
    def alpha(self) -> float: ...
    def measure(self, vectorized: bool = True) -> torch.Tensor: ...
    def set_render(self, render: bool) -> None: ...
    def render(self, env_index: int = 0) -> list[Geom]: ...
