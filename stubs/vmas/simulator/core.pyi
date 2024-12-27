from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeVar, TypeVarTuple, override

import torch
from torch import Tensor

from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.joints import Joint
from vmas.simulator.rendering import Geom
from vmas.simulator.sensors import Sensor
from vmas.simulator.utils import (
    ANGULAR_FRICTION,
    COLLISION_FORCE,
    DRAG,
    JOINT_FORCE,
    LINEAR_FRICTION,
    # override,
    TORQUE_CONSTRAINT_FORCE,
    Color,
    Observable,
)

# Constants

T = TypeVar("T")
Ts = TypeVarTuple("Ts")

class TorchVectorizedObject:
    _batch_dim: int | None
    _device: torch.device | None

    def __init__(
        self, batch_dim: int | None = None, device: torch.device | None = None
    ) -> None: ...
    @property
    def batch_dim(self) -> int: ...
    @batch_dim.setter
    def batch_dim(self, batch_dim: int) -> None: ...
    @property
    def device(self) -> torch.device: ...
    @device.setter
    def device(self, device: torch.device) -> None: ...
    def _check_batch_index(self, batch_index: int) -> None: ...
    def to(self, device: torch.device) -> None: ...

class Shape(ABC):
    @abstractmethod
    def moment_of_inertia(self, mass: float) -> float: ...
    @abstractmethod
    def get_delta_from_anchor(
        self, anchor: tuple[float, float]
    ) -> tuple[float, float]: ...
    @abstractmethod
    def get_geometry(self) -> Geom: ...
    @abstractmethod
    def circumscribed_radius(self) -> float: ...

class Box(Shape):
    _length: float
    _width: float
    hollow: bool

    def __init__(
        self, length: float = 0.3, width: float = 0.1, hollow: bool = False
    ) -> None: ...
    @property
    def length(self) -> float: ...
    @property
    def width(self) -> float: ...
    def moment_of_inertia(self, mass: float) -> float: ...
    def get_delta_from_anchor(
        self, anchor: tuple[float, float]
    ) -> tuple[float, float]: ...
    def get_geometry(self) -> Geom: ...
    def circumscribed_radius(self) -> float: ...

class Sphere(Shape):
    _radius: float

    def __init__(self, radius: float = 0.05) -> None: ...
    @property
    def radius(self) -> float: ...
    def moment_of_inertia(self, mass: float) -> float: ...
    def get_delta_from_anchor(
        self, anchor: tuple[float, float]
    ) -> tuple[float, float]: ...
    def get_geometry(self) -> Geom: ...
    def circumscribed_radius(self) -> float: ...

class Line(Shape):
    _length: float
    _width: int

    def __init__(self, length: float = 0.5) -> None: ...
    @property
    def length(self) -> float: ...
    @property
    def width(self) -> int: ...
    def moment_of_inertia(self, mass: float) -> float: ...
    def get_delta_from_anchor(
        self, anchor: tuple[float, float]
    ) -> tuple[float, float]: ...
    def get_geometry(self) -> Geom: ...
    def circumscribed_radius(self) -> float: ...

class EntityState(TorchVectorizedObject):
    _pos: Tensor | None
    _rot: Tensor | None
    _vel: Tensor | None
    _ang_vel: Tensor | None

    def __init__(self) -> None: ...
    @property
    def pos(self) -> Tensor: ...
    @pos.setter
    def pos(self, pos: Tensor) -> None: ...
    @property
    def rot(self) -> Tensor: ...
    @rot.setter
    def rot(self, rot: Tensor) -> None: ...
    @property
    def vel(self) -> Tensor: ...
    @vel.setter
    def vel(self, vel: Tensor) -> None: ...
    @property
    def ang_vel(self) -> Tensor: ...
    @ang_vel.setter
    def ang_vel(self, ang_vel: Tensor) -> None: ...
    def _reset(self, env_index: int | None) -> None: ...
    def zero_grad(self) -> None: ...
    def _spawn(self, dim_c: int, dim_p: int) -> None: ...

class Entity(TorchVectorizedObject, Observable, ABC):
    _name: str
    _movable: bool
    _rotatable: bool
    _collide: bool
    _density: float
    _mass: float
    _max_speed: float | None
    _v_range: float | None
    _color: Color | Tensor
    _shape: Shape | None
    _is_joint: bool
    _collision_filter: Callable[[Entity], bool]
    _state: EntityState
    _drag: float | None
    _linear_friction: float | None
    _angular_friction: float | None
    _gravity: Tensor | None
    _goal: Entity | None
    _render: Tensor | None

    def __init__(
        self,
        name: str,
        movable: bool = False,
        rotatable: bool = False,
        collide: bool = True,
        density: float = 25.0,
        mass: float = 1.0,
        shape: Shape | None = None,
        v_range: float | None = None,
        max_speed: float | None = None,
        color: Color | Tensor = Color.GRAY,
        is_joint: bool = False,
        drag: float | None = None,
        linear_friction: float | None = None,
        angular_friction: float | None = None,
        gravity: float | Tensor | None = None,
        collision_filter: Callable[[Entity], bool] = lambda _: True,
    ) -> None: ...
    @property
    def batch_dim(self) -> int: ...
    @batch_dim.setter
    def batch_dim(self, batch_dim: int) -> None: ...
    @property
    def is_rendering(self) -> Tensor: ...
    def reset_render(self) -> None: ...
    def collides(self, entity: Entity) -> bool: ...
    @property
    def is_joint(self) -> bool: ...
    @property
    def mass(self) -> float: ...
    @mass.setter
    def mass(self, mass: float) -> None: ...
    @property
    def moment_of_inertia(self) -> float: ...
    @property
    def state(self) -> EntityState: ...
    @property
    def movable(self) -> bool: ...
    @property
    def collide(self) -> bool: ...
    @property
    def shape(self) -> Shape: ...
    @property
    def max_speed(self) -> float | None: ...
    @property
    def v_range(self) -> float | None: ...
    @property
    def name(self) -> str: ...
    @property
    def rotatable(self) -> bool: ...
    @property
    def color(self) -> Color | Tensor | tuple[float, ...]: ...
    @color.setter
    def color(self, color: Color | Tensor | tuple[float, ...]) -> None: ...
    @property
    def goal(self) -> Entity | None: ...
    @goal.setter
    def goal(self, goal: Entity) -> None: ...
    @property
    def drag(self) -> float | None: ...
    @property
    def linear_friction(self) -> float | None: ...
    @linear_friction.setter
    def linear_friction(self, value: float | None) -> None: ...
    @property
    def gravity(self) -> Tensor | None: ...
    @gravity.setter
    def gravity(self, value: Tensor | None) -> None: ...
    @property
    def angular_friction(self) -> float | None: ...
    @property
    def collision_filter(self) -> Callable[[Entity], bool]: ...
    @collision_filter.setter
    def collision_filter(self, collision_filter: Callable[[Entity], bool]) -> None: ...
    def _spawn(self, dim_c: int, dim_p: int) -> None: ...
    def _reset(self, env_index: int) -> None: ...
    def zero_grad(self) -> None: ...
    def set_pos(self, pos: Tensor, batch_index: int) -> None: ...
    def set_vel(self, vel: Tensor, batch_index: int) -> None: ...
    def set_rot(self, rot: Tensor, batch_index: int) -> None: ...
    def set_ang_vel(self, ang_vel: Tensor, batch_index: int) -> None: ...
    def _set_state_property(
        self, prop: property, entity: EntityState, new: Tensor, batch_index: int
    ) -> None: ...
    @override
    def to(self, device: torch.device) -> None: ...
    def render(self, env_index: int = 0) -> list[Geom]: ...

class Action(TorchVectorizedObject):
    _u_noise: float | Sequence[float]
    _u_range: float | Sequence[float]
    _u_multiplier: float | Sequence[float]
    action_size: int
    _u: Tensor | None
    _c: Tensor | None
    _u_range_tensor: Tensor | None
    _u_multiplier_tensor: Tensor | None
    _u_noise_tensor: Tensor | None

    def __init__(
        self,
        u_range: float | Sequence[float],
        u_multiplier: float | Sequence[float],
        u_noise: float | Sequence[float],
        action_size: int,
    ) -> None: ...
    def _check_action_init(self) -> None: ...
    @property
    def u(self) -> Tensor | None: ...
    @u.setter
    def u(self, u: Tensor) -> None: ...
    @property
    def c(self) -> Tensor | None: ...
    @c.setter
    def c(self, c: Tensor) -> None: ...
    @property
    def u_range(self) -> float | Sequence[float]: ...
    @property
    def u_multiplier(self) -> float | Sequence[float]: ...
    @property
    def u_noise(self) -> float | Sequence[float]: ...
    @property
    def u_range_tensor(self) -> Tensor: ...
    @property
    def u_multiplier_tensor(self) -> Tensor: ...
    @property
    def u_noise_tensor(self) -> Tensor: ...
    def _to_tensor(self, value: float | Sequence[float]) -> Tensor: ...
    def _reset(self, env_index: int | None) -> None: ...
    def zero_grad(self) -> None: ...
    def to(self, device: torch.device) -> None: ...

class AgentState(EntityState):
    _c: Tensor | None
    _force: Tensor | None
    _torque: Tensor | None

    def __init__(self) -> None: ...
    @property
    def c(self) -> Tensor | None: ...
    @c.setter
    def c(self, c: Tensor) -> None: ...
    @property
    def force(self) -> Tensor | None: ...
    @force.setter
    def force(self, value: Tensor) -> None: ...
    @property
    def torque(self) -> Tensor | None: ...
    @torque.setter
    def torque(self, value: Tensor) -> None: ...
    @override
    def _reset(self, env_index: int | None) -> None: ...
    @override
    def zero_grad(self) -> None: ...
    @override
    def _spawn(self, dim_c: int, dim_p: int) -> None: ...

class Landmark(Entity):
    def __init__(
        self,
        name: str,
        shape: Shape | None = None,
        movable: bool = False,
        rotatable: bool = False,
        collide: bool = True,
        density: float = 25.0,
        mass: float = 1.0,
        v_range: float | None = None,
        max_speed: float | None = None,
        color: Color | Tensor = Color.GRAY,
        is_joint: bool = False,
        drag: float | None = None,
        linear_friction: float | None = None,
        angular_friction: float | None = None,
        gravity: float | None = None,
        collision_filter: Callable[[Entity], bool] = lambda _: True,
    ) -> None: ...

class Agent(Entity):
    _obs_range: float | None
    _obs_noise: float | None
    _f_range: float | None
    _max_f: float | None
    _t_range: float | None
    _max_t: float | None
    _action_script: Callable[[Agent, World], None] | None
    _sensors: list[Sensor]
    _c_noise: float
    _silent: bool
    _render_action: bool
    _adversary: bool
    _alpha: float
    dynamics: Dynamics
    action_size: int
    discrete_action_nvec: list[int]
    _action: Action
    _state: AgentState

    def __init__(
        self,
        name: str,
        shape: Shape | None = None,
        movable: bool = True,
        rotatable: bool = True,
        collide: bool = True,
        density: float = 25.0,
        mass: float = 1.0,
        f_range: float | None = None,
        max_f: float | None = None,
        t_range: float | None = None,
        max_t: float | None = None,
        v_range: float | None = None,
        max_speed: float | None = None,
        color: Color = Color.BLUE,
        alpha: float = 0.5,
        obs_range: float | None = None,
        obs_noise: float | None = None,
        u_noise: float | Sequence[float] = 0.0,
        u_range: float | Sequence[float] = 1.0,
        u_multiplier: float | Sequence[float] = 1.0,
        action_script: Callable[[Agent, World], None] | None = None,
        sensors: list[Sensor] | None = None,
        c_noise: float = 0.0,
        silent: bool = True,
        adversary: bool = False,
        drag: float | None = None,
        linear_friction: float | None = None,
        angular_friction: float | None = None,
        gravity: float | None = None,
        collision_filter: Callable[[Entity], bool] = lambda _: True,
        render_action: bool = False,
        dynamics: Dynamics | None = None,
        action_size: int | None = None,
        discrete_action_nvec: list[int] | None = None,
    ) -> None: ...
    def add_sensor(self, sensor: Sensor) -> None: ...
    @property
    def action_script(self) -> Callable[[Agent, World], None] | None: ...
    def action_callback(self, world: World) -> None: ...
    @property
    def u_range(self) -> float | Sequence[float]: ...
    @property
    def obs_noise(self) -> float: ...
    @property
    def action(self) -> Action: ...
    @property
    def u_multiplier(self) -> float | Sequence[float]: ...
    @property
    def max_f(self) -> float | None: ...
    @property
    def f_range(self) -> float | None: ...
    @property
    def max_t(self) -> float | None: ...
    @property
    def t_range(self) -> float | None: ...
    @property
    def silent(self) -> bool: ...
    @property
    def sensors(self) -> list[Sensor]: ...
    @property
    def u_noise(self) -> float | Sequence[float]: ...
    @property
    def c_noise(self) -> float: ...
    @property
    def batch_dim(self) -> int: ...
    @batch_dim.setter
    def batch_dim(self, batch_dim: int) -> None: ...
    @property
    def adversary(self) -> bool: ...
    @override
    def _spawn(self, dim_c: int, dim_p: int) -> None: ...
    def _reset(self, env_index: int) -> None: ...
    def zero_grad(self) -> None: ...
    def to(self, device: torch.device) -> None: ...
    def render(self, env_index: int = 0) -> list[Geom]: ...

class World(TorchVectorizedObject):
    _agents: Sequence[Agent]
    _landmarks: list[Landmark]
    _x_semidim: float | None
    _y_semidim: float | None
    _dim_p: Literal[2]
    _dim_c: int
    _dt: float
    _substeps: int
    _sub_dt: float
    _drag: float
    _gravity: Tensor
    _linear_friction: float
    _angular_friction: float
    _collision_force: float
    _joint_force: float
    _contact_margin: float
    _torque_constraint_force: float
    _joints: dict[frozenset[str], Any]  # Joint type could be more specific
    _collidable_pairs: list[set[type[Shape]]]
    entity_index_map: dict[Entity, int]
    forces_dict: dict[Entity, Tensor]
    torques_dict: dict[Entity, Tensor]

    def __init__(
        self,
        batch_dim: int,
        device: torch.device,
        dt: float = 0.1,
        substeps: int = 1,
        drag: float = DRAG,
        linear_friction: float = LINEAR_FRICTION,
        angular_friction: float = ANGULAR_FRICTION,
        x_semidim: float | None = None,
        y_semidim: float | None = None,
        dim_c: int = 0,
        collision_force: float = COLLISION_FORCE,
        joint_force: float = JOINT_FORCE,
        torque_constraint_force: float = TORQUE_CONSTRAINT_FORCE,
        contact_margin: float = 1e-3,
        gravity: tuple[float, float] = (0.0, 0.0),
    ) -> None: ...
    def add_agent(self, agent: Agent) -> None: ...
    def add_landmark(self, landmark: Landmark) -> None: ...
    def add_joint(self, joint: Joint) -> None: ...
    def reset(self, env_index: int) -> None: ...
    def zero_grad(self) -> None: ...
    @property
    def agents(self) -> Sequence[Agent]: ...
    @property
    def landmarks(self) -> list[Landmark]: ...
    @property
    def x_semidim(self) -> float | None: ...
    @property
    def dt(self) -> float: ...
    @property
    def y_semidim(self) -> float | None: ...
    @property
    def dim_p(self) -> Literal[2]: ...
    @property
    def dim_c(self) -> int: ...
    @property
    def joints(self) -> list[Joint]: ...
    @property
    def entities(self) -> list[Entity]: ...
    @property
    def policy_agents(self) -> list[Agent]: ...
    @property
    def scripted_agents(self) -> list[Agent]: ...
    def _cast_ray_to_box(
        self,
        box: Entity,
        ray_origin: Tensor,
        ray_direction: Tensor,
        max_range: float,
    ) -> Tensor: ...
    def _cast_rays_to_box(
        self,
        box_pos: Tensor,
        box_rot: Tensor,
        box_length: Tensor,
        box_width: Tensor,
        ray_origin: Tensor,
        ray_direction: Tensor,
        max_range: float,
    ) -> Tensor: ...
    def cast_ray(
        self,
        entity: Entity,
        angles: Tensor,
        max_range: float,
        entity_filter: Callable[[Entity], bool] = lambda _: False,
    ) -> Tensor: ...
    def cast_rays(
        self,
        entity: Entity,
        angles: Tensor,
        max_range: float,
        entity_filter: Callable[[Entity], bool] = lambda _: False,
    ) -> Tensor: ...
    def get_distance_from_point(
        self, entity: Entity, test_point_pos: Tensor, env_index: int | None = None
    ) -> Tensor: ...
    def get_distance(
        self, entity_a: Entity, entity_b: Entity, env_index: int | None = None
    ) -> Tensor: ...
    def is_overlapping(
        self, entity_a: Entity, entity_b: Entity, env_index: int | None = None
    ) -> Tensor: ...
    def step(self) -> None: ...
    def update_env_forces(
        self,
        entity_a: Entity,
        f_a: Tensor,
        t_a: Tensor,
        entity_b: Entity,
        f_b: Tensor,
        t_b: Tensor,
    ) -> None: ...
    def collides(self, a: Entity, b: Entity) -> bool: ...
    def _get_constraint_forces(
        self,
        pos_a: Tensor,
        pos_b: Tensor,
        dist_min: Tensor,
        force_multiplier: float,
        attractive: bool = False,
    ) -> tuple[Tensor, Tensor]: ...
    def _get_constraint_torques(
        self,
        rot_a: Tensor,
        rot_b: Tensor,
        force_multiplier: float = TORQUE_CONSTRAINT_FORCE,
    ) -> tuple[Tensor, Tensor]: ...
    def _integrate_state(self, entity: Entity, substep: int) -> None: ...
    def _update_comm_state(self, agent: Agent) -> None: ...
    def to(self, device: torch.device) -> None: ...
