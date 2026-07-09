"""Directional policy seeds ("prompts") for the Nash solver (architecture.md §6.1).

Each named prompt is a data object describing an initial (alpha0, beta0)
policy profile that biases damped best-response iteration
(`nash_solver.solve_nash`) toward a specific equilibrium branch. Prompts are
stored declaratively as per-economy arc tables, not as bespoke code paths per
equilibrium, so the same registry can be reused by later solvers (e.g. SCE,
whose own misdirectional prompts are architecture.md §12 item 6).
"""

import numpy as np

from kw.economies import Economy

# Per-economy (speculating type, [(k, j), ...]) arcs for the "speculative"
# prompt: the listed type accepts good j when holding k even though j is not
# its consumption good and is costlier to store than k (architecture.md
# §6.1). All other types, and all other (k, j) pairs of the speculating type,
# use the fundamental-pure rule.
_SPECULATIVE_ARCS: dict[str, tuple[int, list[tuple[int, int]]]] = {
    "A1": (0, [(1, 2), (2, 0)]),
    "A2": (0, [(1, 2), (2, 0)]),
}

# Economies whose speculative prompt is the paper's full pure accept pattern
# ("a full pure policy pattern, not just a change at those two states",
# p. 24), listed exhaustively as (i, k, j) accept arcs; every unlisted entry
# is 0. B's pattern is the speculative Nash of Figure 6 (p. 49): type 1
# (0-indexed) speculates on the costliest good 2, and type 2 holds good 1
# exclusively, refusing good 0.
_SPECULATIVE_FULL_ARCS: dict[str, list[tuple[int, int, int]]] = {
    "B": [
        (0, 2, 0),
        (0, 2, 1),
        (0, 1, 0),
        (1, 0, 1),
        (1, 0, 2),
        (1, 2, 1),
        (2, 1, 2),
    ],
}


def _default_beta(econ: Economy) -> np.ndarray:
    beta = np.zeros((econ.n_types, econ.n_goods), dtype=np.float64)
    beta[np.arange(econ.n_types), econ.consume_idx] = 1.0
    return beta


def _fundamental_alpha(econ: Economy) -> np.ndarray:
    """Accept j over held k iff j is i's consumption good or strictly cheaper than k."""
    cheaper_kj = econ.s[None, :] < econ.s[:, None]  # [k, j] = s[j] < s[k]
    cheaper = np.broadcast_to(cheaper_kj, (econ.n_types, econ.n_goods, econ.n_goods))
    own_consume = np.zeros((econ.n_types, econ.n_goods, econ.n_goods), dtype=bool)
    own_consume[np.arange(econ.n_types), :, econ.consume_idx] = True
    alpha = (cheaper | own_consume) & econ.lambda_feasible
    return alpha.astype(np.float64)


def _default_alpha(econ: Economy) -> np.ndarray:
    return (0.5 * econ.lambda_feasible).astype(np.float64)


def _speculative_alpha(econ: Economy) -> np.ndarray:
    if econ.name in _SPECULATIVE_FULL_ARCS:
        alpha = np.zeros((econ.n_types, econ.n_goods, econ.n_goods), dtype=np.float64)
        for i, k, j in _SPECULATIVE_FULL_ARCS[econ.name]:
            alpha[i, k, j] = 1.0
        alpha *= econ.lambda_feasible
        return alpha
    try:
        speculator, arcs = _SPECULATIVE_ARCS[econ.name]
    except KeyError:
        raise ValueError(f"No speculative prompt defined for economy {econ.name!r}") from None
    alpha = _fundamental_alpha(econ)
    for k, j in arcs:
        alpha[speculator, k, j] = 1.0
    return alpha


def _money_alpha(econ: Economy) -> np.ndarray:
    if econ.fiat_idx is None:
        raise ValueError(f"'money' prompt requires a fiat good; economy {econ.name!r} has none")
    alpha = np.zeros((econ.n_types, econ.n_goods, econ.n_goods), dtype=np.float64)
    for i in range(econ.n_types):
        alpha[i, :, econ.fiat_idx] = 1.0
        alpha[i, :, econ.consume_idx[i]] = 1.0
    alpha *= econ.lambda_feasible
    return alpha


_ALPHA_BUILDERS = {
    "default": _default_alpha,
    "speculative": _speculative_alpha,
    "money": _money_alpha,
}


def nash_prompt(econ: Economy, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Named initial (alpha0 (I,G,G), beta0 (I,G)) policy seed for `solve_nash`."""
    try:
        builder = _ALPHA_BUILDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown prompt {name!r}; valid names: {sorted(_ALPHA_BUILDERS)}"
        ) from None
    return builder(econ), _default_beta(econ)
