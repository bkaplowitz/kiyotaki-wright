from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pyglet  # type: ignore
import torch
from numpy.typing import NDArray
from pyglet.window import Window  # type: ignore

# Type aliases
type Color4f = tuple[float, float, float, float]
type Point2D = tuple[float, float]
type Vertices = Sequence[Point2D]
type PlotRange = (
    float | tuple[float, float] | tuple[tuple[float, float], tuple[float, float]]
)

def get_display(spec: str | None) -> pyglet.canvas.Display | None: ...

class Viewer:
    width: int
    height: int
    window: Window
    geoms: list[Geom]
    onetime_geoms: list[Geom]
    transform: Transform
    bounds: torch.Tensor | None

    def __init__(
        self,
        width: int,
        height: int,
        display: str | None = None,
        visible: bool = True,
    ) -> None: ...
    def close(self) -> None: ...
    def window_closed_by_user(self) -> None: ...
    def set_bounds(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        bottom: torch.Tensor,
        top: torch.Tensor,
    ) -> None: ...
    def add_geom(self, geom: Geom) -> None: ...
    def add_onetime(self, geom: Geom) -> None: ...
    def add_onetime_list(self, geoms: Sequence[Geom]) -> None: ...
    def render(self, return_rgb_array: bool = False) -> NDArray[np.uint8] | None: ...
    def get_array(self) -> NDArray[np.uint8]: ...

class Attr:
    def enable(self) -> None: ...
    def disable(self) -> None: ...

class Transform(Attr):
    translation: Point2D
    rotation: float
    scale: Point2D

    def __init__(
        self,
        translation: Point2D = (0.0, 0.0),
        rotation: float = 0.0,
        scale: Point2D = (1, 1),
    ) -> None: ...
    def set_translation(self, newx: float, newy: float) -> None: ...
    def set_rotation(self, new: float) -> None: ...
    def set_scale(self, newx: float, newy: float) -> None: ...

class Color(Attr):
    vec4: Color4f

    def __init__(self, vec4: Color4f) -> None: ...

class Geom:
    attrs: list[Attr]
    _color: Color

    def __init__(self) -> None: ...
    def render(self) -> None: ...
    def render1(self) -> None: ...
    def add_attr(self, attr: Attr) -> None: ...
    def set_color(self, r: float, g: float, b: float, alpha: float = 1) -> None: ...

class TextLine(Geom):
    label: pyglet.text.Label

    def __init__(
        self, text: str = "", font_size: int = 15, x: float = 0.0, y: float = 0.0
    ) -> None: ...
    def set_text(self, text: str, font_size: int | None = None) -> None: ...

class FilledPolygon(Geom):
    v: Vertices
    draw_border: bool

    def __init__(self, v: Vertices, draw_border: bool = True) -> None: ...

class Line(Geom):
    start: Point2D
    end: Point2D

    def __init__(
        self, start: Point2D = (0.0, 0.0), end: Point2D = (0.0, 0.0), width: float = 1
    ) -> None: ...
    def set_linewidth(self, x: float) -> None: ...

class Image(Geom):
    width: int
    height: int
    arr: NDArray[np.uint8]

    def __init__(self, width: int, height: int, arr: NDArray[np.uint8]) -> None: ...
    def set_arr(self, arr: NDArray[np.uint8]) -> None: ...

class PolyLine(Geom):
    v: Vertices
    width: float

    def __init__(self, v: Vertices, width: float = 1) -> None: ...
    def set_linewidth(self, x: float) -> None: ...

class Compound(Geom):
    gs: list[Geom]

    def __init__(self, gs: list[Geom]) -> None: ...
    def add_attr(self, attr: Attr) -> None: ...

def render_function_util(
    f: Callable[
        [np.ndarray[tuple[int, int], np.dtype[Any]]],
        np.ndarray[tuple[int, int], np.dtype[Any]] | torch.Tensor,
    ],
    plot_range: PlotRange,
    precision: float = 0.01,
    cmap_range: tuple[float, float] | None = None,
    cmap_alpha: float = 1.0,
    cmap_name: str = "viridis",
) -> Image: ...
def make_circle(
    radius: float = 10,
    res: int = 30,
    filled: bool = True,
    angle: float = 2 * 3.141592653589793,
) -> FilledPolygon | PolyLine: ...
def make_ellipse(
    radius_x: float = 10,
    radius_y: float = 5,
    res: int = 30,
    filled: bool = True,
    angle: float = 2 * 3.141592653589793,
) -> FilledPolygon | PolyLine: ...
def make_polygon(
    v: Vertices, filled: bool = True, draw_border: bool = True
) -> FilledPolygon | PolyLine: ...
def make_polyline(v: Vertices) -> PolyLine: ...
def make_capsule(length: float, width: float) -> Compound: ...
