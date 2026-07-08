"""Physical environment kernels: reward, consumption transition, trade resolution.

Pure functions of the period structure (architecture.md §5.2). These do not
depend on population state or policy; they only encode the physics fixed by
an `Economy`.
"""

import numpy as np

from kw.economies import Economy


def reward_cons_table(econ: Economy) -> np.ndarray:
    """Per-(type, holding, consume-decision) reward, eq. (R).

    r[i,h,1] = ubar[i]*(h==consume_idx[i]) - s[f[i]], masked to -inf where
    gamma_feasible[i,h] is False (e.g. holding fiat in Economy C).
    r[i,h,0] = -s[h].
    """
    n_types, n_goods = econ.n_types, econ.n_goods
    holdings = np.arange(n_goods)
    consume_match = holdings[None, :] == econ.consume_idx[:, None]
    r_consume = econ.ubar[:, None] * consume_match - econ.s[econ.f][:, None]
    r_consume = np.where(econ.gamma_feasible, r_consume, -np.inf)
    r_hold = -np.broadcast_to(econ.s, (n_types, n_goods))

    r = np.empty((n_types, n_goods, 2), dtype=np.float64)
    r[:, :, 0] = r_hold
    r[:, :, 1] = r_consume
    return r


def next_holding_cons(econ: Economy) -> np.ndarray:
    """Post-decision holding: c[i,h,1] = f[i], c[i,h,0] = h."""
    n_types, n_goods = econ.n_types, econ.n_goods
    nh = np.empty((n_types, n_goods, 2), dtype=np.int64)
    nh[:, :, 0] = np.arange(n_goods)[None, :]
    nh[:, :, 1] = econ.f[:, None]
    return nh


def trade_outcome(k: int, j: int, offer_k, offer_j) -> tuple[int, int]:
    """Resolve a bilateral meeting: swap holdings iff k != j and both offer."""
    if k != j and offer_k and offer_j:
        return (j, k)
    return (k, j)
