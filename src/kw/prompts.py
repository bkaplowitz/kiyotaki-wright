"""Directional and misdirectional policy/belief seeds ("prompts") for the two
solvers (architecture.md §6.1, §6.2).

`nash_prompt` describes an initial (alpha0, beta0) policy profile that biases
damped best-response iteration (`nash_solver.solve_nash`) toward a specific
equilibrium branch. `sce_prompt` describes an initial (alpha0, beta0, m0, q0)
policy-and-belief profile that biases the restricted-information SCE solver
(`sce_solver.solve_sce`) toward a specific *misdirectional* self-confirming
equilibrium — "skeptical off-path beliefs ... that close off paths to
improvement through trade" (paper p. 26). The paper publishes no per-economy
prompt recipes (no initial m0/q0/mu0 values or prompt names appear anywhere;
see .superpowers/sdd/paper-check-sce.md). Each SCE construction below is
therefore OUR reconstruction, engineered so that `solve_sce` reaches the
non-Nash SCE endpoint the paper reports for that economy in §7; docstrings
quote and cite the reported ENDPOINT (holdings, policy links, deviation
mechanism), not a recipe. Prompts are stored declaratively, not as bespoke
code paths per equilibrium, so both solvers share one registry.
"""

import numpy as np

from kw.distributions import aggregate, reciprocation
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


def _f_endowment_mu0(econ: Economy) -> np.ndarray:
    """Everyone holds f[i]; economy C splits fiat share M0 on idx 0, remainder on f[i]."""
    mu0 = np.zeros((econ.n_types, econ.n_goods), dtype=np.float64)
    if econ.fiat_idx is not None:
        mu0[:, econ.fiat_idx] = econ.M0
        mu0[np.arange(econ.n_types), econ.f] = 1.0 - econ.M0
    else:
        mu0[np.arange(econ.n_types), econ.f] = 1.0
    return mu0


def _f_endowment_agg(econ: Economy) -> np.ndarray:
    return aggregate(econ, _f_endowment_mu0(econ))


_AUTARKY_ACCEPT_RESIDUE = 0.005


def _skeptical_autarky_sce(econ: Economy) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """"skeptical-autarky": autarky under q0 == 0, our reconstruction of the
    Economy A1 endpoint reported at p. 45 (Figure 2, "Non-Nash SCE" panel).

    Reported endpoint: "The non-Nash SCE is autarkic. Every type holds only
    its production good. ... No exchange takes place. ... Off-path beliefs
    about whether a partner would reciprocate a trade offer are never tested
    and so never revised." (p. 45). The paper gives no recipe for reaching
    it; this construction is ours.

    q0 identically 0 (skeptical: no reciprocation believed possible anywhere)
    makes every offer margin an exact tie, so `bellman.greedy_policy`'s
    minimal-movement tie-break (its `current=` argument) keeps alpha0
    unchanged forever, reproducing "never tested and so never revised"
    mechanically. alpha0 is 0 at every spending arc but keeps a small mixed
    residue (0.005) at each type's consumption-acceptance arc
    alpha_i(f(i), c_i) — subjectively optimal because under q0 == 0 the
    policy "is not pinned down at off-path or indifferent trade states"
    (p. 57). The residue never produces an exchange (a KW production cycle
    has no double coincidence, so the counterpart arc alpha_i'(c_i, f(i)) is
    0 and every rare offer is refused), leaving the f-endowment holdings
    stationary, but it is exactly the latent acceptance that makes the
    endpoint non-Nash: "an agent who experimented with one of the refused
    offers would discover that some are in fact accepted" (p. 4). With
    alpha0 strictly 0 instead, (B2) gives Q^sigma == 0, no unilateral
    deviation can ever trade, and autarky would be (degenerately) Nash —
    contradicting the reported endpoint. The residue sits only on the held
    row k = f(i): both sides of every seeded arc pair then refuse with
    probability 1 - 0.005 and 1 respectively, so no policy chain acquires
    O(residue^2) transitions that would stall average-reward evaluation.
    m0 is the population's f-endowment holdings (the autarky allocation
    itself).
    """
    m0 = np.broadcast_to(_f_endowment_agg(econ), (econ.n_types, econ.n_goods)).copy()
    q0 = np.zeros((econ.n_types, econ.n_goods, econ.n_goods), dtype=np.float64)
    alpha0 = np.zeros((econ.n_types, econ.n_goods, econ.n_goods), dtype=np.float64)
    alpha0[np.arange(econ.n_types), econ.f, econ.consume_idx] = _AUTARKY_ACCEPT_RESIDUE
    alpha0 *= econ.lambda_feasible
    beta0 = _default_beta(econ)
    return alpha0, beta0, m0, q0


def _skeptical_lowtrade_sce(econ: Economy) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """"skeptical-lowtrade": our reconstruction of Economy C's low-trade fiat
    SCE endpoint reported at p. 51 (Figure 8, "Non-Nash SCE" panel).

    Reported endpoint: "type 1 holds good 2 with probability 0.999, type 2
    holds fiat money with probability 0.932, and type 3 holds good 1 with
    probability 0.973. In aggregate terms the SCE is approximately 0.320 for
    fiat money, 0.335 for good 1, 0.333 for good 2, and 0.012 for good 3.
    ... In the constructed SCE, type 1 would accept fiat money for good 2,
    type 2 would accept good 1 for good 3, and type 3 would accept good 2
    for good 1. The corresponding spenders -- type 2 offering good 0 for
    good 2, type 3 offering good 1 for good 3, and type 1 offering good 2
    for good 1 -- do not make these offers in that SCE policy, so the
    missing links are not tested by experience. With no successful trade and
    no consumption, the holding law is stationary." (p. 51). The
    accept/spender link lists are the paper's; the initial policy, beliefs,
    and distribution that reach the endpoint are ours.

    Construction (0-indexed types/goods, fiat = good 0): start from the
    "sell any held good for fiat, or for my own consumption good" baseline
    (the same rule as `nash_prompt`'s "money" prompt), which already sets the
    three "spender" links the paper says are missing (each is exactly a type
    selling its current holding for its own consumption good) -- override
    those three to 0. Then add the two "accept" intermediary links the
    baseline does not cover (type 2 selling good 3 for good 1; type 3 selling
    good 1 for good 2); the third accept link (type 1 selling good 2 for
    fiat) is already covered by the baseline. q0 == 0 throughout, so this
    exact mixed accept/refuse pattern is a subjective fixed point by the same
    all-ties-preserve-`current` mechanism as `_skeptical_autarky_sce`. Note
    the frozen endpoint distribution is transient-dependent (with zero trade
    flow, F_sigma is the identity and every distribution is stationary): the
    exact 0.999/0.932/0.973 figures reflect the paper's unpublished run
    path, while this deterministic construction freezes within ~0.03 of them
    (see findings.md, H2 section).
    """
    if econ.fiat_idx is None:
        raise ValueError(f"'skeptical-lowtrade' prompt requires a fiat good; {econ.name!r} has none")
    alpha0 = np.zeros((econ.n_types, econ.n_goods, econ.n_goods), dtype=np.float64)
    for i in range(econ.n_types):
        alpha0[i, :, econ.fiat_idx] = 1.0
        alpha0[i, :, econ.consume_idx[i]] = 1.0
    alpha0 *= econ.lambda_feasible
    # Accept links kept alive (p. 51): type2 holds good3 -> sells for good1;
    # type3 holds good1 -> sells for good2. (type1 holds good2 -> sells for
    # fiat is already 1 via the baseline's fiat rule above.)
    alpha0[1, 3, 1] = 1.0
    alpha0[2, 1, 2] = 1.0
    # Missing spender links (p. 51): these consumption-seeking sales are
    # refused, breaking the circulation the SNE relies on.
    alpha0[1, 0, 2] = 0.0
    alpha0[2, 1, 3] = 0.0
    alpha0[0, 2, 1] = 0.0
    m0 = np.broadcast_to(_f_endowment_agg(econ), (econ.n_types, econ.n_goods)).copy()
    q0 = np.zeros((econ.n_types, econ.n_goods, econ.n_goods), dtype=np.float64)
    beta0 = _default_beta(econ)
    return alpha0, beta0, m0, q0


def _type1_autarky_sce(econ: Economy) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """"type1-autarky": our reconstruction of Economy D's type-1-excluded SCE
    endpoint reported at p. 54 (Figure 10, "Non-Nash SCE" panel).

    Reported endpoint: "The non-Nash SCE is simpler. Nearly every type
    settles on good 1 as the sole store of value. Types 3, 4, and 5 hold
    good 1 exclusively, and type 2 holds it with probability 0.72. The
    outlier is type 1, who holds nothing but her production good 3. Nobody
    accepts good 3, so type 1 has no route to her consumption good. ... The
    result is a self-fulfilling collapse." (p. 54). The paper gives no
    recipe for reaching it; this construction is ours.

    Construction (0-indexed: the paper's 1-indexed "type 1" is code index 0,
    its production good 3 is code good 2): type 0 gets the full
    skeptical-autarky treatment (q0[0] == 0, alpha0[0] == 0) so it is
    locked at its production good by the same tie-preserving mechanism as
    `_skeptical_autarky_sce`, and never revises because it never offers (its
    on-path set stays empty, so q0[0] is never touched by `solve_sce`'s
    belief update either). The other types start from the uniform
    `nash_prompt`-style default (alpha0 = 0.5 * lambda_feasible) with beliefs
    equal to the reciprocation kernel that default implies -- non-degenerate,
    so their subjective best response can genuinely search for (and find)
    the good-1 (code good-0) commodity-money convention the paper reports
    for types 2-5.
    """
    m0 = np.broadcast_to(_f_endowment_agg(econ), (econ.n_types, econ.n_goods)).copy()
    alpha0 = 0.5 * econ.lambda_feasible.astype(np.float64)
    alpha0[0] = 0.0
    mu0 = _f_endowment_mu0(econ)
    Q0 = reciprocation(econ, mu0, alpha0)
    q0 = np.broadcast_to(Q0, (econ.n_types, econ.n_goods, econ.n_goods)).copy()
    q0[0] = 0.0
    beta0 = _default_beta(econ)
    return alpha0, beta0, m0, q0


_SCE_BUILDERS = {
    "skeptical-autarky": _skeptical_autarky_sce,
    "skeptical-lowtrade": _skeptical_lowtrade_sce,
    "type1-autarky": _type1_autarky_sce,
}


def sce_prompt(
    econ: Economy, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Named initial (alpha0 (I,G,G), beta0 (I,G), m0 (I,G), q0 (I,G,G)) seed for `solve_sce`."""
    try:
        builder = _SCE_BUILDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown prompt {name!r}; valid names: {sorted(_SCE_BUILDERS)}"
        ) from None
    return builder(econ)
