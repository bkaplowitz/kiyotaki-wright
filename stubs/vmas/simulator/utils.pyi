from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal, overload

import numpy as np
import torch
from torch import Tensor

# Constants with literal types
X: Literal[0]
Y: Literal[1]
Z: Literal[2]
ALPHABET: str
VIEWER_DEFAULT_ZOOM: float
INITIAL_VIEWER_SIZE: tuple[Literal[700], Literal[700]]
LINE_MIN_DIST: float  # 4/6e2
COLLISION_FORCE: Literal[100]
JOINT_FORCE: Literal[130]
TORQUE_CONSTRAINT_FORCE: Literal[1]
DRAG: float
LINEAR_FRICTION: float
ANGULAR_FRICTION: float

# Type aliases using modern syntax
DEVICE_TYPING = torch.device | str | int
AGENT_OBS_TYPE = Tensor | dict[str, Tensor]
AGENT_INFO_TYPE = dict[str, Tensor]
AGENT_REWARD_TYPE = Tensor
OBS_TYPE = list[AGENT_OBS_TYPE] | dict[str, AGENT_OBS_TYPE]
INFO_TYPE = list[AGENT_INFO_TYPE] | dict[str, AGENT_INFO_TYPE]
REWARD_TYPE = list[AGENT_REWARD_TYPE] | dict[str, AGENT_REWARD_TYPE]
DONE_TYPE = Tensor

class Color(Enum):
    RED: tuple[float, float, float]
    GREEN: tuple[float, float, float]
    BLUE: tuple[float, float, float]
    LIGHT_GREEN: tuple[float, float, float]
    WHITE: tuple[float, float, float]
    GRAY: tuple[float, float, float]
    BLACK: tuple[float, float, float]

# Modern decorator type
def override(cls: type) -> Any: ...
def _init_pyglet_device() -> None: ...

class Observable:
    _observers: list[Observer]
    def __init__(self) -> None: ...
    def subscribe(self, observer: Observer) -> None: ...
    def notify_observers(self, *args: Any, **kwargs: Any) -> None: ...
    def unsubscribe(self, observer: Observer) -> None: ...

class Observer(ABC):
    @abstractmethod
    def notify(self, observable: Observable, *args: Any, **kwargs: Any) -> None: ...

# Video and visualization functions
def save_video(name: str, frame_list: list[np.ndarray[Any, Any]], fps: int) -> None: ...
def x_to_rgb_colormap(
    x: np.ndarray[Any, Any],
    low: float | None = None,
    high: float | None = None,
    alpha: float = 1.0,
    cmap_name: str = "viridis",
    cmap_res: int = 10,
) -> np.ndarray[Any, Any]: ...

# Generic type for nested structures

def extract_nested_with_index[T](
    data: Tensor | dict[str, T], index: int
) -> Tensor | dict[str, T]: ...

# Torch utilities with modern typing
class TorchUtils:
    @staticmethod
    def clamp_with_norm(tensor: Tensor, max_norm: float) -> Tensor: ...
    @staticmethod
    def rotate_vector(vector: Tensor, angle: Tensor) -> Tensor: ...
    @staticmethod
    def cross(vector_a: Tensor, vector_b: Tensor) -> Tensor: ...
    @staticmethod
    def compute_torque(f: Tensor, r: Tensor) -> Tensor: ...
    @staticmethod
    @overload
    def to_numpy(data: Tensor) -> np.ndarray[Any, Any]: ...
    @staticmethod
    @overload
    def to_numpy(data: dict[str, Tensor]) -> dict[str, np.ndarray[Any, Any]]: ...
    @staticmethod
    @overload
    def to_numpy(data: list[Tensor]) -> list[np.ndarray[Any, Any]]: ...
    @staticmethod
    @overload
    def recursive_clone(value: Tensor) -> Tensor: ...
    @staticmethod
    @overload
    def recursive_clone(value: dict[str, Tensor]) -> dict[str, Tensor]: ...
    @staticmethod
    def recursive_require_grad_(
        value: Tensor | dict[str, Tensor] | list[Tensor],
    ) -> None: ...
    @staticmethod
    def where_from_index(
        env_index: int, new_value: Tensor, old_value: Tensor
    ) -> Tensor: ...

# Scenario utilities with modern typing
class ScenarioUtils:
    @staticmethod
    def spawn_entities_randomly(
        entities: list[Any],  # Could be more specific with Entity type
        world: Any,  # Could be more specific with World type
        env_index: int | None,
        min_dist_between_entities: float,
        x_bounds: tuple[int, int],
        y_bounds: tuple[int, int],
        occupied_positions: Tensor | None = None,
        disable_warn: bool = False,
    ) -> None: ...
    @staticmethod
    def find_random_pos_for_entity(
        occupied_positions: Tensor,
        env_index: int | None,
        world: Any,  # Could be more specific with World type
        min_dist_between_entities: float,
        x_bounds: tuple[int, int],
        y_bounds: tuple[int, int],
        disable_warn: bool = False,
    ) -> Tensor: ...
    @staticmethod
    def check_kwargs_consumed(
        dictionary_of_kwargs: dict[str, Any], warn: bool = True
    ) -> None: ...
