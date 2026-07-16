"""Economy primitives: the five Kiyotaki-Wright economies (paper §5).

All goods/types use 0-based positional indices. `consume_idx[i]` maps type i
to the index of the good it consumes; the paper's 1-indexed labels are used
only in comments below. In Economy C, good index 0 is fiat money (not in any
type's consumption or production set).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Economy:
    name: str
    n_goods: int
    n_types: int
    nu: np.ndarray
    f: np.ndarray
    consume_idx: np.ndarray
    s: np.ndarray
    ubar: np.ndarray
    gamma_feasible: np.ndarray
    lambda_feasible: np.ndarray
    fiat_idx: int | None
    M0: float | None


def _default_lambda_feasible(n_types: int, n_goods: int) -> np.ndarray:
    k_ne_j = ~np.eye(n_goods, dtype=bool)
    return np.broadcast_to(k_ne_j, (n_types, n_goods, n_goods)).copy()


def _default_gamma_feasible(n_types: int, n_goods: int, fiat_idx: int | None) -> np.ndarray:
    gamma = np.ones((n_types, n_goods), dtype=bool)
    if fiat_idx is not None:
        gamma[:, fiat_idx] = False
    return gamma


def _make_economy(
    name: str,
    n_goods: int,
    n_types: int,
    f: list[int],
    consume_idx: list[int],
    s: list[float],
    ubar: list[float],
    fiat_idx: int | None = None,
    M0: float | None = None,
) -> Economy:
    nu = np.full(n_types, 1.0 / n_types, dtype=np.float64)
    f_arr = np.asarray(f, dtype=np.int64)
    consume_idx_arr = np.asarray(consume_idx, dtype=np.int64)
    s_arr = np.asarray(s, dtype=np.float64)
    ubar_arr = np.asarray(ubar, dtype=np.float64)
    gamma_feasible = _default_gamma_feasible(n_types, n_goods, fiat_idx)
    lambda_feasible = _default_lambda_feasible(n_types, n_goods)

    for arr in (nu, f_arr, consume_idx_arr, s_arr, ubar_arr, gamma_feasible, lambda_feasible):
        arr.setflags(write=False)

    return Economy(
        name=name,
        n_goods=n_goods,
        n_types=n_types,
        nu=nu,
        f=f_arr,
        consume_idx=consume_idx_arr,
        s=s_arr,
        ubar=ubar_arr,
        gamma_feasible=gamma_feasible,
        lambda_feasible=lambda_feasible,
        fiat_idx=fiat_idx,
        M0=M0,
    )


def _build_registry() -> dict[str, Economy]:
    a1 = _make_economy(
        name="A1",
        n_goods=3,
        n_types=3,
        f=[1, 2, 0],  # paper f(1)=2, f(2)=3, f(3)=1 → 0-indexed
        consume_idx=[0, 1, 2],
        s=[0.1, 1.0, 20.0],
        ubar=[100.0, 100.0, 100.0],
    )
    a2 = _make_economy(
        name="A2",
        n_goods=3,
        n_types=3,
        f=[1, 2, 0],
        consume_idx=[0, 1, 2],
        s=[0.1, 1.0, 20.0],
        ubar=[500.0, 500.0, 500.0],
    )
    b = _make_economy(
        name="B",
        n_goods=3,
        n_types=3,
        f=[2, 0, 1],  # paper f(1)=3, f(2)=1, f(3)=2 → 0-indexed
        consume_idx=[0, 1, 2],
        s=[1.0, 4.0, 9.0],
        ubar=[100.0, 100.0, 100.0],
    )
    c = _make_economy(
        name="C",
        n_goods=4,
        n_types=3,
        f=[2, 3, 1],  # paper f(1)=2, f(2)=3, f(3)=1 over goods {1,2,3} → indices {1,2,3}
        consume_idx=[1, 2, 3],
        s=[0.0, 9.0, 14.0, 29.0],
        ubar=[100.0, 100.0, 100.0],
        fiat_idx=0,
        M0=0.32,
    )
    d = _make_economy(
        name="D",
        n_goods=5,
        n_types=5,
        f=[2, 3, 4, 0, 1],  # paper f(1)=3, f(2)=4, f(3)=5, f(4)=1, f(5)=2 → 0-indexed
        consume_idx=[0, 1, 2, 3, 4],
        s=[1.0, 4.0, 9.0, 20.0, 30.0],
        ubar=[200.0, 200.0, 200.0, 200.0, 200.0],
    )
    return {e.name: e for e in (a1, a2, b, c, d)}


ECONOMIES: dict[str, Economy] = _build_registry()


def get_economy(name: str) -> Economy:
    try:
        return ECONOMIES[name]
    except KeyError:
        raise KeyError(f"Unknown economy {name!r}; valid names: {sorted(ECONOMIES)}") from None
