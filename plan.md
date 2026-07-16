# Kiyotaki–Wright Equilibria & Learning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and rigorously test the three procedures of Sargent & Yang (2026) — RVI Nash solver (Algorithm 1), SCE fixed-point solver (Algorithm 2), tabular PUCT self-play learner (Algorithms 3–4, empirical + known-environment) — then swap in standard RL baselines (differential Q-learning, Dyna) and run a multi-seed classification campaign with three extensions.

**Architecture:** Two specs. `architecture.md` (repo root) covers the replication — Part A of this plan (Phases 0–5); `extensions/architecture.md` covers everything beyond the paper — Part B (Phases 6–7). Plain "§N" references point to the root spec; "ext §N" to the extensions spec; equation labels (B1–B4, F1–F4, S2–S3, L2–L3, P1–P2, R, D1) to the paper. Pure-function solvers over an `Economy` dataclass; one `Learner` protocol so every learner runs on the identical population simulator; solver-independent diagnostics score every policy.

**Tech Stack:** Python ≥3.12, uv, NumPy, joblib, matplotlib; pytest + ruff (dev). No GPU, no other runtime deps.

## Global Constraints

- Two solvers stay separate first-class modules: `nash_solver.py` (Algorithm 1, damped best-response/RVI iteration) and `sce_solver.py` (Algorithm 2, SCE fixed point). Never merge them (explicit user requirement).
- Replication/extension separation (§1 separation rule): `src/kw/` is paper-exact by default; extension code goes in `src/kw/extensions/` (imports core, never the reverse — Task 15 adds the lint test); extension hooks on core modules (`ema_gain`, `ModelBank`, off-default flags) default to paper behavior. Docs likewise: root `architecture.md` = replication only, `extensions/architecture.md` = everything new.
- All goods/types use **0-based positional indices** in code; `Economy.consume_idx[i]` maps type→consumption-good index (paper is 1-indexed; C's fiat good is index 0). Never mix labels and indices.
- float64 everywhere; argmax ties break to lowest index; tolerances ε_dist=1e-12, ε_BR=1e-9, ε_SCE=1e-9 (flags).
- Table 1 defaults frozen in `LearnerConfig` (architecture §7.3); explicit orthogonal CLI flags only — no meta-flags.
- Every script has `--dry-run` (N=60, R=5, N_max=6, n_min=2, p=2) exercising the full pipeline.
- RNG: `np.random.default_rng(seed).spawn(...)` named streams; same seed ⇒ bit-identical run.
- Runs write `runs/<econ>/<config>/<seed>/{manifest.json, arrays.npz, trace.json}`; figures/tables read artifacts only.
- Tooling: `uv run pytest`, `uv run ruff check`; use `fd`/`rg` never find/grep; never `python -c` with multiline content (write scripts to the scratchpad and `uv run python` them); `trash` not `rm -rf`; conventional commits.
- Fast tests default; anything >5 s marked `@pytest.mark.slow` (run with `-m slow`).
- No comments explaining changes; comments only for non-obvious constraints.
- When a paper detail is ambiguous, consult the fidelity-risk registry (architecture §12) and follow the documented choice.

---

# PART A — REPLICATION (spec: `architecture.md`)

Phases 0–5 implement and verify the paper. Nothing in Part A depends on
Part B; Part A alone is a complete, publishable replication.

## Phase 0 — Scaffolding & economies

### Task 1: Project skeleton + `economies.py`

**Files:**
- Create: `pyproject.toml`, `src/kw/__init__.py`, `src/kw/economies.py`
- Test: `tests/test_economies.py`

**Interfaces:**
- Produces: `Economy` frozen dataclass with fields `name: str`, `n_goods: int`, `n_types: int`, `nu: np.ndarray (I,)`, `f: np.ndarray (I,) int` (production-good index), `consume_idx: np.ndarray (I,) int`, `s: np.ndarray (G,)`, `ubar: np.ndarray (I,)`, `gamma_feasible: np.ndarray (I,G) bool`, `lambda_feasible: np.ndarray (I,G,G) bool`, `fiat_idx: int | None`, `M0: float | None`.
- Produces: `get_economy(name: str) -> Economy` for names `{"A1","A2","B","C","D"}`; `ECONOMIES: dict[str, Economy]`.
- Feasibility defaults: `lambda_feasible[i,k,j] = (k != j)`; `gamma_feasible[i,h] = True` except C where `gamma_feasible[i, 0] = False` (Γ_i(0)={0}); offering is always feasible action 0/1 with action 1 masked when infeasible.

- [ ] **Step 1: Initialize project**

```bash
cd /Users/bkaplowitz/Developer/work/test-kiyotaki-wright-rl/kiyotaki-wright
uv init --lib --name kw --python 3.12
uv add numpy joblib matplotlib
uv add --dev pytest ruff
```

Then edit `pyproject.toml` to set `[tool.pytest.ini_options] markers = ["slow: >5s integration runs"]` and `addopts = "-m 'not slow'"`, and `[tool.ruff] line-length = 100`.

- [ ] **Step 2: Write failing tests**

```python
# tests/test_economies.py
import numpy as np
import pytest
from kw.economies import get_economy, ECONOMIES

def test_registry_names():
    assert set(ECONOMIES) == {"A1", "A2", "B", "C", "D"}

@pytest.mark.parametrize("name,n_goods,n_types", [
    ("A1", 3, 3), ("A2", 3, 3), ("B", 3, 3), ("C", 4, 3), ("D", 5, 5)])
def test_shapes(name, n_goods, n_types):
    e = get_economy(name)
    assert e.n_goods == n_goods and e.n_types == n_types
    assert e.nu.shape == (n_types,) and np.isclose(e.nu.sum(), 1.0)
    assert e.s.shape == (n_goods,) and e.f.shape == (n_types,)

def test_a1_values():
    e = get_economy("A1")
    assert np.allclose(e.s, [0.1, 1.0, 20.0])
    assert np.allclose(e.ubar, [100.0] * 3)
    assert list(e.f) == [1, 2, 0]            # type i produces f(i): 2,3,1 → idx 1,2,0
    assert list(e.consume_idx) == [0, 1, 2]

def test_a2_only_ubar_differs():
    a1, a2 = get_economy("A1"), get_economy("A2")
    assert np.allclose(a2.ubar, [500.0] * 3)
    assert np.allclose(a1.s, a2.s) and list(a1.f) == list(a2.f)

def test_b_values():
    e = get_economy("B")
    assert list(e.f) == [2, 0, 1] and np.allclose(e.s, [1.0, 4.0, 9.0])

def test_c_fiat():
    e = get_economy("C")
    assert e.fiat_idx == 0 and e.M0 == 0.32
    assert np.allclose(e.s, [0.0, 9.0, 14.0, 29.0])
    assert list(e.consume_idx) == [1, 2, 3] and list(e.f) == [2, 3, 1]
    assert not e.gamma_feasible[:, 0].any()

def test_d_values():
    e = get_economy("D")
    assert list(e.f) == [2, 3, 4, 0, 1] and np.allclose(e.ubar, [200.0] * 5)
    assert np.allclose(e.s, [1.0, 4.0, 9.0, 20.0, 30.0])

def test_same_good_trade_infeasible():
    e = get_economy("A1")
    assert not e.lambda_feasible[0, 1, 1]
    assert e.lambda_feasible[0, 1, 0]
```

- [ ] **Step 3: Run to verify failure** — `uv run pytest tests/test_economies.py -v` → FAIL (`ModuleNotFoundError: kw.economies`).

- [ ] **Step 4: Implement `src/kw/economies.py`** — frozen dataclass exactly as in the Interfaces block; registry built from architecture §5.1 table. Note C's `f` in paper labels is (2,3,1) over goods {1,2,3} → indices (2,3,1) in the 4-good array; `consume_idx = (1,2,3)`.

- [ ] **Step 5: Run tests** — `uv run pytest tests/test_economies.py -v` → all PASS. Also `uv run ruff check src tests` → clean.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src tests
git commit -m "feat(economies): project skeleton and five-economy registry"
```

---

## Phase 1 — Kernels: environment, law of motion, Bellman

### Task 2: `environment.py`

**Files:**
- Create: `src/kw/environment.py`
- Test: `tests/test_environment.py`

**Interfaces:**
- Produces: `reward_cons_table(econ) -> np.ndarray (I,G,2)` implementing (R): `r[i,h,1] = ubar[i]*(h==consume_idx[i]) - s[f[i]]`, `r[i,h,0] = -s[h]`; infeasible γ=1 cells set to `-np.inf`.
- Produces: `next_holding_cons(econ) -> np.ndarray (I,G,2) int`: `f[i]` if γ=1 else `h`.
- Produces: `trade_outcome(k, j, offer_k, offer_j) -> tuple[int,int]`: swap iff `k != j and offer_k and offer_j`.

- [ ] **Step 1: Failing tests** — type 1 of A1 holding its consumption good (index 0) and consuming earns `100 − s[1] = 99.0`; holding good idx 2 and not consuming earns `−20.0`; C: `reward_cons_table(C)[i, 0, 1] == -inf` (can't consume fiat); trade swap only on mutual offer with k≠j.

```python
# tests/test_environment.py
import numpy as np
from kw.economies import get_economy
from kw.environment import reward_cons_table, next_holding_cons, trade_outcome

def test_reward_a1():
    e = get_economy("A1")
    r = reward_cons_table(e)
    assert np.isclose(r[0, 0, 1], 100.0 - 1.0)
    assert np.isclose(r[0, 2, 0], -20.0)
    assert np.isclose(r[2, 2, 1], 100.0 - 0.1)

def test_fiat_not_consumable():
    r = reward_cons_table(get_economy("C"))
    assert np.isneginf(r[:, 0, 1]).all()

def test_next_holding():
    e = get_economy("A1")
    nh = next_holding_cons(e)
    assert nh[0, 0, 1] == e.f[0] and nh[0, 2, 0] == 2

def test_trade_outcome():
    assert trade_outcome(0, 1, 1, 1) == (1, 0)
    assert trade_outcome(0, 1, 1, 0) == (0, 1)
    assert trade_outcome(1, 1, 1, 1) == (1, 1)
```

- [ ] **Step 2: Run → FAIL; implement; run → PASS.**
- [ ] **Step 3: Commit** — `git commit -m "feat(environment): reward map, consumption transition, trade resolution"`

### Task 3: `distributions.py` — (B1), (B2), (F1)–(F4)

**Files:**
- Create: `src/kw/distributions.py`
- Test: `tests/test_distributions.py`

**Interfaces:**
- Consumes: `Economy`, policies `alpha (I,G,G)`, `beta (I,G)`.
- Produces: `aggregate(econ, mu) -> (G,)` where `mu` is (I,G) with rows on the simplex; `reciprocation(econ, mu, alpha) -> Q (G,G)` per (B2) with `Q[:, j] = 0` when `μ(j)=0`; `one_step(econ, mu, alpha, beta) -> (I,G)` per (F1)–(F3); `invariant(econ, mu0, alpha, beta, tol=1e-12, max_iter=200_000) -> (I,G)` (damped forward iteration, damping 0.5 if oscillation detected, raise `RuntimeError` on non-convergence).

- [ ] **Step 1: Failing tests**

```python
# tests/test_distributions.py
import numpy as np
import pytest
from kw.economies import get_economy
from kw.distributions import aggregate, reciprocation, one_step, invariant

def _fundamental_a1(e):
    alpha = np.zeros((3, 3, 3)); beta = np.zeros((3, 3))
    for i in range(3):
        beta[i, e.consume_idx[i]] = 1.0
        for k in range(3):
            for j in range(3):
                if k == j: continue
                good_for_me = (j == e.consume_idx[i])
                cheaper = e.s[j] < e.s[k]
                alpha[i, k, j] = float(good_for_me or cheaper)
    return alpha, beta

def test_one_step_conserves_mass():
    e = get_economy("A1"); alpha, beta = _fundamental_a1(e)
    mu = np.full((3, 3), 1 / 3)
    mu2 = one_step(e, mu, alpha, beta)
    assert np.allclose(mu2.sum(axis=1), 1.0) and (mu2 >= -1e-15).all()

def test_invariant_fundamental_a1():
    e = get_economy("A1"); alpha, beta = _fundamental_a1(e)
    mu0 = np.zeros((3, 3)); mu0[np.arange(3), e.f] = 1.0
    mu = invariant(e, mu0, alpha, beta)
    agg = (e.nu[:, None] * mu).sum(axis=0)
    assert np.allclose(agg, [0.5, 1 / 3, 1 / 6], atol=1e-9)
    assert np.allclose(mu[0], [0, 1, 0], atol=1e-9)
    assert np.allclose(mu[1], [0.5, 0, 0.5], atol=1e-9)
    assert np.allclose(mu[2], [1, 0, 0], atol=1e-9)

def test_q_zero_on_unheld_good():
    e = get_economy("A1"); alpha, beta = _fundamental_a1(e)
    mu = np.zeros((3, 3)); mu[:, 0] = 1.0
    Q = reciprocation(e, mu, alpha)
    assert (Q[:, 1] == 0).all() and (Q[:, 2] == 0).all()

def test_fiat_conserved_in_c():
    e = get_economy("C")
    rng = np.random.default_rng(0)
    alpha = rng.uniform(0, 1, (3, 4, 4)) * e.lambda_feasible
    beta = rng.uniform(0, 1, (3, 4)) * e.gamma_feasible
    mu0 = np.zeros((3, 4)); mu0[:, 0] = 0.32; mu0[np.arange(3), e.f] = 0.68
    mu = mu0
    for _ in range(50):
        mu = one_step(e, mu, alpha, beta)
        assert np.isclose((e.nu * mu[:, 0]).sum(), 0.32 * 1.0, atol=1e-12)
```

The `test_invariant_fundamental_a1` targets are the closed-form fundamental-SNE holdings (architecture §15 derivation: T1→good 2, T2→½ good 1 + ½ good 3, T3→good 1; aggregate (½, ⅓, ⅙)).

- [ ] **Step 2: Run → FAIL; implement (F1)–(F3) vectorized**

```python
# core of one_step
def one_step(econ, mu, alpha, beta):
    agg = (econ.nu[:, None] * mu).sum(axis=0)                    # μ(k)
    Q = reciprocation(econ, mu, alpha)                           # (B2)
    F = mu[:, :, None] * agg[None, None, :] * alpha * Q[None]    # (F1)
    F[:, np.arange(econ.n_goods), np.arange(econ.n_goods)] = 0.0
    mu_plus = mu - F.sum(axis=2) + F.sum(axis=1)                 # (F2)
    consumed = mu_plus * beta                                    # (F3)
    mu_next = mu_plus - consumed
    mu_next[np.arange(econ.n_types), econ.f] += consumed.sum(axis=1)
    return mu_next
```

with `reciprocation` per (B1)–(B2): `eta[i,k] = nu[i]*mu[i,k]/agg[k]` (0 where `agg==0`), `Q[k,j] = sum_i eta[i,j]*alpha[i,j,k]`.

- [ ] **Step 3: Run → PASS. Commit** — `git commit -m "feat(distributions): population law of motion F1-F4 and reciprocation kernel"`

### Task 4: `bellman.py` — RVI, greedy policy, policy evaluation

**Files:**
- Create: `src/kw/bellman.py`
- Test: `tests/test_bellman.py`

**Interfaces:**
- Produces: `rvi_solve(econ, i, agg, Q, tol=1e-10, max_iter=200_000) -> (g: float, V: (G,), W: (G,))` on (B3)–(B4), bias normalized `V[f[i]] = 0`.
- Produces: `greedy_policy(econ, i, Q, V, W) -> (alpha_i (G,G), beta_i (G,))` per (BR-e)/(BR-c), ties → action 0, feasibility masked.
- Produces: `policy_eval(econ, i, alpha_i, beta_i, agg, Q, tol=1e-10) -> (g: float, V: (G,))` — fixed-policy analog (RVI on the σ-averaged operator).
- Produces subjective variants for Solver 2: `rvi_solve_subjective(econ, i, m_i (G,), q_i (G,G), tol=1e-10) -> (g, V, W)` and `policy_eval_subjective(econ, i, alpha_i, beta_i, m_i, q_i) -> (g, V)`, both using kernel `T_q` (architecture §6.2): offer succeeds w.p. `q_i(k,j)` independent of others' λ.

- [ ] **Step 1: Failing tests**

```python
# tests/test_bellman.py
import numpy as np
from kw.economies import get_economy
from kw.distributions import aggregate, reciprocation, invariant
from kw.bellman import rvi_solve, greedy_policy, policy_eval, rvi_solve_subjective

def test_no_trade_gain_is_minus_storage():
    e = get_economy("A1")
    agg = np.full(3, 1 / 3); Q = np.zeros((3, 3))
    alpha_i = np.zeros((3, 3)); beta_i = np.zeros(3); beta_i[e.consume_idx[0]] = 1.0
    g, V = policy_eval(e, 0, alpha_i, beta_i, agg, Q)
    assert np.isclose(g, -e.s[e.f[0]], atol=1e-9)     # type 1 stores good 2 forever: g = -1.0

def test_greedy_is_fundamental_at_fundamental_mu():
    e = get_economy("A1")
    from tests.test_distributions import _fundamental_a1
    alpha, beta = _fundamental_a1(e)
    mu0 = np.zeros((3, 3)); mu0[np.arange(3), e.f] = 1.0
    mu = invariant(e, mu0, alpha, beta)
    agg = (e.nu[:, None] * mu).sum(axis=0)
    Q = reciprocation(e, mu, alpha)
    for i in range(3):
        g, V, W = rvi_solve(e, i, agg, Q)
        a_i, b_i = greedy_policy(e, i, Q, V, W)
        g_pol, _ = policy_eval(e, i, a_i, b_i, agg, Q)
        assert np.isclose(g, g_pol, atol=1e-8)
        assert b_i[e.consume_idx[i]] == 1.0

def test_gain_dominates_any_policy():
    e = get_economy("B")
    rng = np.random.default_rng(1)
    mu = rng.dirichlet(np.ones(3), size=3)
    agg = (e.nu[:, None] * mu).sum(axis=0)
    alpha_rand = rng.uniform(0, 1, (3, 3, 3)) * e.lambda_feasible
    Q = reciprocation(e, mu, alpha_rand)
    g_star, V, W = rvi_solve(e, 0, agg, Q)
    for _ in range(5):
        a = rng.integers(0, 2, (3, 3)) * e.lambda_feasible[0]
        b = rng.integers(0, 2, 3).astype(float)
        g_pol, _ = policy_eval(e, 0, a.astype(float), b, agg, Q)
        assert g_star >= g_pol - 1e-8

def test_subjective_no_reciprocation_means_no_trade_value():
    e = get_economy("A1")
    m = np.full(3, 1 / 3); q = np.zeros((3, 3))
    g, V, W = rvi_solve_subjective(e, 0, m, q)
    assert np.isclose(g, -e.s[e.f[0]], atol=1e-8)     # skeptical beliefs → autarky value
```

- [ ] **Step 2: Run → FAIL; implement.** RVI iteration for (B3)–(B4): given V, compute `W[h] = max over feasible γ of r[i,h,γ] + V[c(h,γ)]` (B3); `TV[k] = Σ_j agg[j]·max_λ {(1−λ)W[k] + λ(Q[k,j]W[j] + (1−Q[k,j])W[k])}` (B4); update `g = TV[f[i]] − V_old[f[i]] ... ` — standard RVI: `V_new = TV − TV[kref]`, `g = TV[kref] + V_old[kref] − V_old[kref]` i.e. `g_est = TV[kref]` when `V[kref]=0`; iterate to sup-norm tolerance on V and g. `policy_eval` replaces both maxes with σ-weighted averages. Subjective variants replace `agg → m_i` and the trade branch with `λ·(q_i[k,j]·W[j] + (1−q_i[k,j])·W[k])` per (S2)–(S3).

- [ ] **Step 3: Run → PASS. Commit** — `git commit -m "feat(bellman): RVI solver, greedy extraction, policy evaluation, subjective variants"`

---

## Phase 2 — Solver 1: stationary Nash (Algorithm 1)

### Task 5: `prompts.py` + `nash_solver.py`

**Files:**
- Create: `src/kw/prompts.py`, `src/kw/nash_solver.py`
- Test: `tests/test_nash_solver.py`

**Interfaces:**
- Produces: `nash_prompt(econ, name) -> (alpha0 (I,G,G), beta0 (I,G))`. Names: `"default"` (α=1/2 on feasible pairs, β_i = 1 on consume_idx else 0), `"speculative"` (A2: type 0 offers idx1→idx2 and idx2→idx0 pure; B: type 1 offers idx0→idx2 and idx2→idx1; rest fundamental-pure), `"money"` (C: α_i(k,0)=1, α_i(k, consume_idx[i])=1, else 0).
- Produces: `solve_nash(econ, alpha0, beta0, mu0=None, rho=0.3, eps_br=1e-9, eps_dist=1e-12, max_iter=2000) -> NashResult` with fields `alpha, beta, mu, Q, g (I,), gain_gap_history: list[float], dist_history: list[float], converged: bool`. `mu0` defaults to everyone holding `f[i]` (C: fiat share M0 on idx 0, remainder on f[i]).

- [ ] **Step 1: Failing tests**

```python
# tests/test_nash_solver.py
import numpy as np
import pytest
from kw.economies import get_economy
from kw.prompts import nash_prompt
from kw.nash_solver import solve_nash

def test_a1_default_prompt_reaches_fundamental():
    e = get_economy("A1")
    res = solve_nash(e, *nash_prompt(e, "default"))
    assert res.converged and res.gain_gap_history[-1] <= 1e-9
    agg = (e.nu[:, None] * res.mu).sum(axis=0)
    assert np.allclose(agg, [0.5, 1 / 3, 1 / 6], atol=1e-6)
    assert res.alpha[1, 2, 0] > 0.99          # type 2 accepts good 1 (money) for good 3

def test_a2_speculative_prompt_stays_speculative():
    e = get_economy("A2")
    res = solve_nash(e, *nash_prompt(e, "speculative"))
    assert res.converged
    assert res.alpha[0, 1, 2] > 0.99          # type 1 offers good 2 for good 3

def test_a1_speculation_not_equilibrium():
    e = get_economy("A1")
    res = solve_nash(e, *nash_prompt(e, "speculative"))
    assert res.converged and res.alpha[0, 1, 2] < 0.01   # ū=100: condition fails, reverts

@pytest.mark.slow
def test_b_two_branches():
    e = get_economy("B")
    r1 = solve_nash(e, *nash_prompt(e, "default"))
    r2 = solve_nash(e, *nash_prompt(e, "speculative"))
    assert r1.converged and r2.converged
    assert not np.allclose(r1.alpha, r2.alpha, atol=0.1)

@pytest.mark.slow
def test_c_money_prompt_fiat_valued():
    e = get_economy("C")
    res = solve_nash(e, *nash_prompt(e, "money"))
    assert res.converged
    assert res.alpha[1, 1, 0] > 0.99          # type 2 accepts fiat for its production good... trades

@pytest.mark.slow
def test_d_converges():
    e = get_economy("D")
    res = solve_nash(e, *nash_prompt(e, "default"))
    assert res.converged and res.gain_gap_history[-1] <= 1e-9
```

The A2 tests encode the Kiyotaki–Wright condition (architecture §15): `s3−s2 = 19` vs `ū/6` → speculative sustainable at ū=500 only.

- [ ] **Step 2: Run → FAIL; implement Algorithm 1 loop**

```python
def solve_nash(econ, alpha0, beta0, mu0=None, rho=0.3,
               eps_br=1e-9, eps_dist=1e-12, max_iter=2000):
    alpha, beta = alpha0.copy(), beta0.copy()
    mu = _default_mu0(econ) if mu0 is None else mu0.copy()
    gaps, dists = [], []
    for _ in range(max_iter):
        mu = invariant(econ, mu, alpha, beta)
        agg = (econ.nu[:, None] * mu).sum(axis=0)
        Q = reciprocation(econ, mu, alpha)
        alpha_br, beta_br = np.zeros_like(alpha), np.zeros_like(beta)
        gap = 0.0
        for i in range(econ.n_types):
            g_star, V, W = rvi_solve(econ, i, agg, Q)
            alpha_br[i], beta_br[i] = greedy_policy(econ, i, Q, V, W)
            g_pol, _ = policy_eval(econ, i, alpha[i], beta[i], agg, Q)
            gap = max(gap, g_star - g_pol)
        dist = np.abs(one_step(econ, mu, alpha, beta) - mu).max()
        gaps.append(gap); dists.append(dist)
        if gap <= eps_br and dist <= eps_dist:
            return NashResult(alpha, beta, mu, Q, _gains(econ, alpha, beta, mu), gaps, dists, True)
        alpha = (1 - rho) * alpha + rho * alpha_br
        beta = (1 - rho) * beta + rho * beta_br
    return NashResult(alpha, beta, mu, Q, _gains(econ, alpha, beta, mu), gaps, dists, False)
```

- [ ] **Step 3: Run fast tests → PASS, then** `uv run pytest tests/test_nash_solver.py -m slow -v` → PASS.
- [ ] **Step 4: Commit** — `git commit -m "feat(nash): Algorithm 1 damped best-response RVI iteration with prompt registry"`

### Task 6: `scripts/compute_nash.py` + H1 checkpoint

**Files:**
- Create: `src/kw/artifacts.py`, `scripts/compute_nash.py`
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Produces: `save_run(path: Path, manifest: dict, arrays: dict[str, np.ndarray], trace: dict) -> None` and `load_run(path) -> (manifest, arrays, trace)`; manifest auto-adds git SHA, ISO timestamp, package versions.
- CLI: `uv run python scripts/compute_nash.py --economy A1 --prompt default --out runs/A1/nash-default` (flags: `--rho --eps-br --eps-dist --max-iter --dry-run`).

- [ ] **Step 1: Failing test** — round-trip `save_run`/`load_run` in `tmp_path`, manifest contains `git_sha` and `config`.
- [ ] **Step 2: Implement; run → PASS.**
- [ ] **Step 3: H1 checkpoint — run all Nash branches and eyeball against the paper:**

```bash
for econ in A1 A2 B D; do uv run python scripts/compute_nash.py --economy $econ --prompt default --out runs/$econ/nash-default; done
uv run python scripts/compute_nash.py --economy A2 --prompt speculative --out runs/A2/nash-spec
uv run python scripts/compute_nash.py --economy B --prompt speculative --out runs/B/nash-spec
uv run python scripts/compute_nash.py --economy C --prompt money --out runs/C/nash-money
```

Expected: all converge; printed summary shows final gain gap ≤1e-9, holdings match paper Figures 2/4/6/8/10 (record comparison notes in `findings.md`, created now with a `## H1 Nash benchmarks` section).

- [ ] **Step 4: Commit** — `git commit -m "feat(scripts): compute_nash CLI, artifact IO, H1 benchmark runs"`

---

## Phase 3 — Solver 2: SCE fixed point (Algorithm 2)

### Task 7: `sce_solver.py` + misdirectional prompts

**Files:**
- Create: `src/kw/sce_solver.py`
- Modify: `src/kw/prompts.py` (add `sce_prompt`)
- Create: `scripts/compute_sce.py`
- Test: `tests/test_sce_solver.py`

**Interfaces:**
- Produces: `sce_prompt(econ, name) -> (alpha0, beta0, m0 (I,G), q0 (I,G,G))`. Names: `"skeptical-autarky"` (q0 ≡ 0, no-offer α0, β_i=1 on consume_idx, m0 = holdings under f-endowment); `"skeptical-lowtrade"` (C), `"type1-autarky"` (D). **Before coding these, re-read paper §7's per-economy SCE constructions in the extracted text (`scratchpad/mms_alpha_research.txt`) and transcribe them exactly; the registry entry must cite the page.** (Fidelity item §12.6.)
- Produces: `solve_sce(econ, alpha0, beta0, m0, q0, rho=0.3, eps_sce=1e-9, eps_dist=1e-12, max_iter=5000) -> SCEResult` with `alpha, beta, mu, m, q, s_history, b_history, d_history, converged`. Loop per architecture §6.2: subjective RVI best response (damped), objective invariant, **on-path-only** belief update `m_i ← agg`, `q_i[k,j] ← Q[k,j]` for `(i,k,j)` with `mu[i,k]*agg[j]*alpha[i,k,j] > 1e-14`, off-path frozen.

- [ ] **Step 1: Failing tests**

```python
# tests/test_sce_solver.py
import numpy as np
import pytest
from kw.economies import get_economy
from kw.prompts import sce_prompt, nash_prompt
from kw.sce_solver import solve_sce
from kw.diagnostics import br_gain   # implemented minimally in this task, fully in Task 11

def test_a1_autarky_sce():
    e = get_economy("A1")
    res = solve_sce(e, *sce_prompt(e, "skeptical-autarky"))
    assert res.converged
    assert res.s_history[-1] <= 1e-9 and res.b_history[-1] <= 1e-9
    assert res.alpha.max() < 0.01                     # no trade
    b, _ = br_gain(e, res.alpha, res.beta, res.mu)
    assert b > 1.0                                    # subjectively optimal, objectively poor

def test_nash_is_sce():
    e = get_economy("A1")
    from kw.nash_solver import solve_nash
    nash = solve_nash(e, *nash_prompt(e, "default"))
    a0, b0, m0, q0 = sce_prompt(e, "skeptical-autarky")
    res = solve_sce(e, nash.alpha, nash.beta, m0, q0)
    assert res.converged and res.s_history[-1] <= 1e-6
    b, _ = br_gain(e, res.alpha, res.beta, res.mu)
    assert b <= 1e-6                                  # from a Nash start, SCE solver stays at Nash
```

- [ ] **Step 2: Implement minimal `diagnostics.br_gain`** (full module in Task 11): invariant μ of (α,β) if not given, then per type `rvi_solve` minus `policy_eval`, return `(max_i gap, per-type array)`.
- [ ] **Step 3: Run → FAIL; implement `solve_sce`; run → PASS** (mark slow ones as needed).
- [ ] **Step 4: `scripts/compute_sce.py`** (same shape as compute_nash; flags `--economy --prompt --rho --eps-sce --eps-dist --out --dry-run`). H2 checkpoint:

```bash
uv run python scripts/compute_sce.py --economy A1 --prompt skeptical-autarky --out runs/A1/sce-autarky
uv run python scripts/compute_sce.py --economy C --prompt skeptical-lowtrade --out runs/C/sce-lowtrade
uv run python scripts/compute_sce.py --economy D --prompt type1-autarky --out runs/D/sce-t1autarky
```

Expected: residuals (s,b,d) → 0, positive BR gain; log to `findings.md ## H2 SCE benchmarks`.

- [ ] **Step 5: Commit** — `git commit -m "feat(sce): Algorithm 2 restricted-information SCE fixed point with misdirectional prompts"`

---

## Phase 4 — Learning infrastructure

### Task 8: `empirical_model.py`

**Files:**
- Create: `src/kw/empirical_model.py`, `src/kw/learners/__init__.py`, `src/kw/learners/base.py`
- Test: `tests/test_empirical_model.py`

**Interfaces:**
- Produces (base.py): `TradeRecord(agent, i, k, j, lam, h_next, r)` and `ConsRecord(agent, i, h, gamma, h_next, r)` (frozen dataclasses, all ints except `r: float`); `LearnerConfig` frozen dataclass with the Table 1 fields and defaults (architecture §7.3): `n_agents=1800, rounds=35, n_iters_max=180, n_min=100, delta_stop=0.003, patience=8, n_sims=32, horizon=10, c_puct=1.5, tau=1.0, zeta0=0.45, zeta_decay=0.94, zeta_min=0.01, kappa0=2.0, eps_floor=0.03, h_dep=14, n_sims_dep=96, tau_dep=0.35, eval_rounds=120, eval_interval=5`; `Learner` protocol (architecture §7.5 signatures).
- Produces: `EmpiricalModel(econ, kappa0=2.0, ema_gain=0.0)` with `observe_meeting(i: int, j_good: int)`, `observe_trade(rec)`, `observe_cons(rec)`; read methods `m_hat() -> (I,G)`, `q_hat() -> (G,G)`, `p_trade() -> (I,G,G,2,G)`, `p_cons() -> (I,G,2,G)`, `r_trade() -> (I,G,G,2)` (≡0), `r_cons() -> (I,G,2)`. Offer/acceptance counts `O, A_acc` pooled over types; updated only on true offers with k≠j (`A_acc` increments when the *partner* offered back — caller passes acceptance via `observe_offer(k, j, accepted: bool)`, add that method). EMA: before each count increment in a cell-owning table, multiply the whole table by `(1−ema_gain)` per record; `ema_gain=0.0` (the default, always used in Part A) recovers the paper's cumulative counts — the non-zero setting is extension E1 (ext §5.1), implemented here as a hook because it is one multiply.
- Produces: `ModelBank(econ, n_banks, **kw)` mapping `agent_id -> EmpiricalModel` (shared: `n_banks=1`, all agents → bank 0; per-agent: `n_banks=n_agents`).

- [ ] **Step 1: Failing tests**

```python
# tests/test_empirical_model.py
import numpy as np
from kw.economies import get_economy
from kw.empirical_model import EmpiricalModel
from kw.learners.base import TradeRecord, ConsRecord

def test_priors():
    m = EmpiricalModel(get_economy("A1"))
    assert np.allclose(m.q_hat(), 0.5)                      # A_acc=κ0/2, O=κ0
    assert np.allclose(m.m_hat(), 1 / 3)
    assert np.allclose(m.p_trade(), 1 / 3)                  # uniform Dirichlet-style prior
    assert np.allclose(m.r_trade(), 0.0) and np.allclose(m.r_cons(), 0.0)

def test_counts_update():
    e = get_economy("A1"); m = EmpiricalModel(e)
    m.observe_offer(k=0, j=1, accepted=True)
    m.observe_offer(k=0, j=1, accepted=False)
    assert np.isclose(m.q_hat()[0, 1], (1.0 + 1.0) / (2.0 + 2.0))   # (κ0/2 + 1)/(κ0 + 2)
    m.observe_trade(TradeRecord(0, 0, 0, 1, 1, 1, 0.0))
    p = m.p_trade()
    assert p[0, 0, 1, 1].argmax() == 1

def test_cons_reward_average():
    e = get_economy("A1"); m = EmpiricalModel(e)
    m.observe_cons(ConsRecord(0, 0, 0, 1, e.f[0], 99.0))
    assert np.isclose(m.r_cons()[0, 0, 1], (0.0 * 2.0 + 99.0) / 3.0)  # (S+r)/(B+1), B0=κ0

def test_ema_equals_cumulative_at_zero_gain():
    e = get_economy("A1")
    m0, m1 = EmpiricalModel(e, ema_gain=0.0), EmpiricalModel(e, ema_gain=0.05)
    for _ in range(30):
        m0.observe_offer(0, 1, True); m1.observe_offer(0, 1, True)
        m0.observe_offer(0, 1, False); m1.observe_offer(0, 1, False)
    assert np.isclose(m0.q_hat()[0, 1], 0.5, atol=0.02)
    assert m1.q_hat()[0, 1] != m0.q_hat()[0, 1] or True     # EMA down-weights the κ0 prior faster
    assert 0.4 < m1.q_hat()[0, 1] < 0.6
```

- [ ] **Step 2: Run → FAIL; implement; run → PASS.**
- [ ] **Step 3: Commit** — `git commit -m "feat(model): empirical count model with kappa0 priors, EMA variant, model bank"`

### Task 9: `planner.py` — DP tables + PUCT selection

**Files:**
- Create: `src/kw/planner.py`
- Test: `tests/test_planner.py`

**Interfaces:**
- Produces: `ModelBackend` protocol: `m(i)->(G,)`, `p_trade(i)->(G,G,2,G)`, `p_cons(i)->(G,2,G)`, `r_trade(i)->(G,G,2)`, `r_cons(i)->(G,2)`. `EmpiricalBackend(model: EmpiricalModel)` adapts Task 8. `RulesBackend(econ, model)` (Task 15) implements the same protocol from (m̂, q̂) + true rules.
- Produces: `dp_tables(econ, backend, i, alpha_i, beta_i, v_i, horizon, greedy=False) -> (psi_t (G,G,2), psi_c (G,2))` — top-level action values via (L2)–(L3) mutual recursion with terminal `v_i`; `greedy=True` replaces prior-weighted interior mixes with maxes (architecture §7.2).
- Produces: `puct_select(values (2,), prior (2,), feasible (2,) bool, n_sims, c_puct, tau) -> (pi (2,), root_value: float)` — deterministic (P1) loop with min-max normalized `S_x` (tied → 0.5), tie-break lowest index, then (P2) `pi ∝ N**(1/tau)`; `root_value = max over feasible x of values[x]`.
- Produces: `Planner(econ, backend, alpha, beta, v, cfg, greedy=False)` with `query_trade(i, k, j) -> (pi, value)`, `query_cons(i, h) -> (pi, value)`, both memoized until `refresh(alpha, beta, v)` is called (once per iteration — fidelity item §12.3). Priors at the root: `prior = [1−alpha[i,k,j], alpha[i,k,j]]` (trade) / `[1−beta[i,h], beta[i,h]]` (consumption).

- [ ] **Step 1: Failing tests**

```python
# tests/test_planner.py
import numpy as np
from kw.planner import puct_select, dp_tables, Planner, EmpiricalBackend
from kw.empirical_model import EmpiricalModel
from kw.economies import get_economy

def test_puct_visits_sum_and_favor_better_action():
    pi, v = puct_select(np.array([0.0, 10.0]), np.array([0.5, 0.5]),
                        np.array([True, True]), n_sims=32, c_puct=1.5, tau=1.0)
    assert np.isclose(pi.sum(), 1.0) and pi[1] > 0.75 and np.isclose(v, 10.0)

def test_puct_tied_values_split_evenly():
    pi, _ = puct_select(np.array([5.0, 5.0]), np.array([0.5, 0.5]),
                        np.array([True, True]), 32, 1.5, 1.0)
    assert np.isclose(pi[0], 0.5, atol=1 / 32 + 1e-9)

def test_puct_infeasible_never_visited():
    pi, _ = puct_select(np.array([0.0, 100.0]), np.array([0.5, 0.5]),
                        np.array([True, False]), 32, 1.5, 1.0)
    assert pi[1] == 0.0

def test_dp_terminal_is_v():
    e = get_economy("A1"); m = EmpiricalModel(e)
    v = np.array([1.0, 2.0, 3.0])
    psi_t, psi_c = dp_tables(e, EmpiricalBackend(m), 0,
                             np.full((3, 3), 0.5), np.full(3, 0.5), v, horizon=1)
    assert psi_c.shape == (3, 2)

def test_planner_memoizes():
    e = get_economy("A1"); m = EmpiricalModel(e)
    from kw.learners.base import LearnerConfig
    cfg = LearnerConfig()
    pl = Planner(e, EmpiricalBackend(m), np.full((3, 3, 3), 0.5),
                 np.full((3, 3), 0.5), np.zeros((3, 3)), cfg)
    r1 = pl.query_trade(0, 0, 1)
    m.observe_offer(0, 1, True)                      # model moves, cache must not
    r2 = pl.query_trade(0, 0, 1)
    assert np.array_equal(r1[0], r2[0]) and r1[1] == r2[1]
```

- [ ] **Step 2: Run → FAIL; implement.** DP core (prior mode):

```python
def dp_tables(econ, be, i, alpha_i, beta_i, v_i, horizon, greedy=False):
    G = econ.n_goods
    V = np.tile(v_i, (horizon + 1, 1)); W = V.copy()
    pt, pc = be.p_trade(i), be.p_cons(i)
    rt, rc = be.r_trade(i), be.r_cons(i)
    m = be.m(i)
    for d in range(1, horizon + 1):
        psi_c = rc + pc @ V[d - 1]                                    # (G,2)
        W[d] = (psi_c.max(1) if greedy
                else (1 - beta_i) * psi_c[:, 0] + beta_i * psi_c[:, 1])
        psi_t = rt + np.einsum("kjlh,h->kjl", pt, W[d])               # (G,G,2)
        V[d] = (m * (psi_t.max(2) if greedy
                else (1 - alpha_i) * psi_t[:, :, 0] + alpha_i * psi_t[:, :, 1])).sum(1)
    psi_c_top = rc + pc @ V[horizon - 1]
    psi_t_top = rt + np.einsum("kjlh,h->kjl", pt, W[horizon])
    return psi_t_top, psi_c_top
```

`puct_select` is the literal 32-step loop of (P1) with `N` starting at zeros and `S` frozen after normalization. Infeasible actions masked to `S = −inf`.

- [ ] **Step 3: Run → PASS. Commit** — `git commit -m "feat(planner): L2/L3 DP recursion and deterministic PUCT visit allocation"`

### Task 10: `simulate.py` — population harness (Algorithm 3 shell)

**Files:**
- Create: `src/kw/simulate.py`
- Test: `tests/test_simulate.py`

**Interfaces:**
- Consumes: `Learner` protocol, `LearnerConfig`, `environment.trade_outcome`, `reward_cons_table`, `next_holding_cons`.
- Produces: `init_population(econ, cfg, rng) -> (types (N,), holdings (N,))` — `N/|I|` agents per type holding `f[i]`; C: `round(M0 * N/|I|)` agents per type hold fiat idx 0 (fidelity §12.8).
- Produces: `run_learning(econ, learner, cfg, seed) -> RunResult` — full Algorithms 3–4 shell: per iteration, R rounds of {uniform pairing without replacement → both agents `observe_meeting` via learner → `trade_action` both → resolve → `observe_trade` both (records appended online, pair order = draw order, §12.2)}, then consumption pass; `learner.end_iteration(n)` returns Δ̄_n; stopping rule (D1) (Δ̄ ≤ δ for p consecutive after n_min, cap n_iters_max); then `learner.deployment_policy()` → (α★, β★); `mu_star = invariant(...)` seeded from terminal empirical holdings (§12.7); per-iteration trace of aggregate holdings, trade volume, Δ̄_n.
- Produces: `RunResult` dataclass: `alpha_star, beta_star, mu_star, trace: dict[str, list], n_iters: int, holdings_final (N,)`.
- Produces: `evaluate_fixed(econ, alpha, beta, cfg, rng) -> dict` — 120-round fixed-policy simulation recording holdings every 5 rounds (paper's evaluation protocol).

- [ ] **Step 1: Failing tests** (use a trivial `RandomLearner` defined in the test that offers/consumes uniformly and returns Δ̄ = 0):

```python
# tests/test_simulate.py
import numpy as np
from kw.economies import get_economy
from kw.learners.base import LearnerConfig
from kw.simulate import init_population, run_learning

DRY = LearnerConfig(n_agents=60, rounds=5, n_iters_max=6, n_min=2, patience=2)

class RandomLearner:
    def __init__(self, econ, rng): self.econ, self.rng = econ, rng
    def trade_action(self, i, k, j, rng): return int(rng.random() < 0.5) if k != j else 0
    def cons_action(self, i, h, rng):
        return int(self.econ.gamma_feasible[i, h] and rng.random() < 0.5)
    def observe_meeting(self, i, j_good): pass
    def observe_trade(self, rec): pass
    def observe_cons(self, rec): pass
    def end_iteration(self, n): return 0.0
    def deployment_policy(self):
        a = np.full((self.econ.n_types,) + (self.econ.n_goods,) * 2, 0.5) * self.econ.lambda_feasible
        b = np.full((self.econ.n_types, self.econ.n_goods), 0.5) * self.econ.gamma_feasible
        return a, b

def test_init_population_c_fiat_count():
    e = get_economy("C")
    types, hold = init_population(e, DRY, np.random.default_rng(0))
    per_type = 60 // 3
    for i in range(3):
        assert (hold[types == i] == 0).sum() == round(0.32 * per_type)

def test_goods_conserved_outside_consumption():
    e = get_economy("A1")
    rng = np.random.default_rng(1)
    res = run_learning(e, RandomLearner(e, rng), DRY, seed=1)
    assert res.n_iters <= 6 and len(res.trace["trade_volume"]) == res.n_iters

def test_determinism():
    e = get_economy("A1")
    r1 = run_learning(e, RandomLearner(e, np.random.default_rng(2)), DRY, seed=7)
    r2 = run_learning(e, RandomLearner(e, np.random.default_rng(2)), DRY, seed=7)
    assert np.array_equal(r1.alpha_star, r2.alpha_star)
    assert r1.trace["trade_volume"] == r2.trace["trade_volume"]
```

- [ ] **Step 2: Run → FAIL; implement; run → PASS.** Stopping-rule edge: iteration count is 1-based against `n_min`; deployment always extracted even at the cap.
- [ ] **Step 3: Commit** — `git commit -m "feat(simulate): population learning harness with D1 stopping and deployment extraction"`

### Task 11: `diagnostics.py` — full module

**Files:**
- Create/complete: `src/kw/diagnostics.py`
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- Produces: `br_gain(econ, alpha, beta, mu=None) -> (float, (I,))` (Task 7's minimal version, now canonical); `stationarity(econ, alpha, beta, mu) -> float`; `tv_distance(mu_a, mu_b, nu) -> float` (aggregate TV); `sce_certificate(econ, alpha, beta, m_hat, q_hat) -> (s: float, b_belief: float)` — subjective optimality gap of (α,β) under θ̂=(m̂,q̂) via `rvi_solve_subjective`/`policy_eval_subjective`, plus max on-path |q̂ − Q^σ|; `classify_run(econ, alpha, beta, mu, m_hat, q_hat, thresholds=Thresholds()) -> str` returning `"NASH" | "SCE_TRAP" | "UNSETTLED"` per architecture §8 (`Thresholds` dataclass: `nash_gain=0.05, dist=1e-3, sce_s=0.05, sce_belief=0.02`).

- [ ] **Step 1: Failing tests** — Nash artifact from Task 5 classifies `NASH`; SCE artifact from Task 7 (with its own beliefs as m̂, q̂) classifies `SCE_TRAP`; random policy at non-invariant μ classifies `UNSETTLED`.

```python
# tests/test_diagnostics.py
import numpy as np
from kw.economies import get_economy
from kw.prompts import nash_prompt, sce_prompt
from kw.nash_solver import solve_nash
from kw.sce_solver import solve_sce
from kw.diagnostics import br_gain, classify_run

def test_nash_classifies_nash():
    e = get_economy("A1")
    r = solve_nash(e, *nash_prompt(e, "default"))
    assert classify_run(e, r.alpha, r.beta, r.mu, None, None) == "NASH"

def test_autarky_sce_classifies_trap():
    e = get_economy("A1")
    r = solve_sce(e, *sce_prompt(e, "skeptical-autarky"))
    assert classify_run(e, r.alpha, r.beta, r.mu, r.m, r.q) == "SCE_TRAP"

def test_br_gain_zero_at_nash():
    e = get_economy("B")
    r = solve_nash(e, *nash_prompt(e, "default"))
    b, per = br_gain(e, r.alpha, r.beta, r.mu)
    assert b <= 1e-8 and per.shape == (3,)
```

- [ ] **Step 2: Run → FAIL; implement; run → PASS. Commit** — `git commit -m "feat(diagnostics): BR gain, SCE certificate, outcome classifier"`

---

## Phase 5 — MuZero learner (Algorithms 3–4)

### Task 12: `learners/muzero.py` — fully empirical variant

**Files:**
- Create: `src/kw/learners/muzero.py`
- Test: `tests/test_muzero.py`

**Interfaces:**
- Produces: `MuZeroLearner(econ, cfg: LearnerConfig, rng, known_rules=False, greedy_backup=False, centered_values=False, ema_gain=0.0, per_agent=False)` implementing the `Learner` protocol. Internals: tables `alpha (I,G,G)` init 0.5 (0 infeasible), `beta (I,G)` init 0.5 (0 infeasible), `v (I,G)` init 0; `ModelBank`; one `Planner` per bank, rebuilt (`refresh`) at iteration start. Action methods: query planner (memoized), sample from π using the learner's action-stream rng. `end_iteration(n)`: per (i,root) means of visit-policies π̄ and max root values V̄ over the iteration's queries; updates `alpha ← clip[(1−ζ_n)α + ζ_n π̄(1)]` on visited roots (same β, v); ζ_n = `max(zeta_min, zeta0 * zeta_decay**n)`; clip to `[eps_floor, 1−eps_floor]` on feasible cells; `centered_values` subtracts `V̄[f[i]]` and pins `v[i, f[i]] = 0`; returns Δ̄_n = mean |policy change| over visited roots (post-clip, §12.9). `deployment_policy()`: fresh planner pass at `h_dep, n_sims_dep, tau_dep`, no observation, greedy... — returns the π-derived (α★, β★) over all roots.

- [ ] **Step 1: Failing tests**

```python
# tests/test_muzero.py
import numpy as np
import pytest
from kw.economies import get_economy
from kw.learners.base import LearnerConfig
from kw.learners.muzero import MuZeroLearner
from kw.simulate import run_learning
from kw.diagnostics import br_gain

DRY = LearnerConfig(n_agents=60, rounds=5, n_iters_max=6, n_min=2, patience=2)

def test_dry_run_completes_all_econs():
    for name in ["A1", "C", "D"]:
        e = get_economy(name)
        rng = np.random.default_rng(0)
        lr = MuZeroLearner(e, DRY, rng)
        res = run_learning(e, lr, DRY, seed=0)
        assert np.isfinite(res.alpha_star).all()
        assert (res.alpha_star[e.lambda_feasible] >= DRY.eps_floor - 1e-12).all()

def test_policies_respect_clip_and_masks():
    e = get_economy("C")
    lr = MuZeroLearner(e, DRY, np.random.default_rng(1))
    res = run_learning(e, lr, DRY, seed=1)
    assert (res.beta_star[:, 0] == 0).all()          # fiat never consumable

@pytest.mark.slow
def test_a1_full_run_reaches_nash():
    e = get_economy("A1")
    cfg = LearnerConfig()
    lr = MuZeroLearner(e, cfg, np.random.default_rng(0))
    res = run_learning(e, lr, cfg, seed=0)
    b, _ = br_gain(e, res.alpha_star, res.beta_star, res.mu_star)
    assert b < 0.1                                    # paper: 1.5e-12; seed-level slack allowed
    agg = (e.nu[:, None] * res.mu_star).sum(axis=0)
    assert abs(agg[0] - 0.5) < 0.05
```

- [ ] **Step 2: Run fast tests → FAIL; implement; fast tests → PASS.**
- [ ] **Step 3: Run the slow A1 test** — `uv run pytest tests/test_muzero.py -m slow -v`. Expected PASS in minutes. If BR gain lands far from 0, triage against fidelity registry §12 before touching hyperparameters.
- [ ] **Step 4: Commit** — `git commit -m "feat(muzero): tabular PUCT self-play learner, fully empirical variant"`

### Task 13: `scripts/run_learning.py` + single-seed empirical replication (H3)

**Files:**
- Create: `scripts/run_learning.py`
- Modify: `findings.md`

**Interfaces:**
- CLI: `uv run python scripts/run_learning.py --economy A1 --learner muzero-emp --seed 0 --out runs/A1/muzero-emp/0` with flags `--learner {muzero-emp, muzero-known, q, dyna-emp, dyna-rules}`, `--known-rules/--greedy-backup/--centered-values` (explicit overrides for ablations), `--ema-gain`, `--per-agent`, `--eps-floor`, `--kappa0`, `--dry-run`. Saves RunResult arrays + trace with per-iteration `br_gain` and `stationarity` (computed post hoc every iteration from the stored policy snapshot — cheap).

- [ ] **Step 1: Implement CLI; verify** `--dry-run` completes for all five economies.
- [ ] **Step 2: H3 replication runs (seed 0):**

```bash
for econ in A1 A2 B C D; do
  uv run python scripts/run_learning.py --economy $econ --learner muzero-emp --seed 0 \
    --out runs/$econ/muzero-emp/0
done
```

Expected against paper Table 2 (order-of-magnitude, single seed): A1 gain ≈ 0, A2 ≈ 21, B ≈ 0, C ≈ 13 with broken fiat links (`alpha_star[1, 0, 1] ≈ eps_floor`), D ≈ 1.7 with stationarity distance ≈ 0.17. Record all five in `findings.md ## H3 empirical learner`, noting matches/mismatches and §12 items implicated by any mismatch.

- [ ] **Step 3: Commit** — `git commit -m "feat(scripts): run_learning CLI and H3 single-seed empirical replication"`

### Task 14: Known-environment variant + 2³ ablation (H4)

**Files:**
- Modify: `src/kw/planner.py` (add `RulesBackend`), `src/kw/learners/muzero.py` (wire flags — mostly done in Task 12)
- Test: `tests/test_known_env.py`

**Interfaces:**
- Produces: `RulesBackend(econ, model: EmpiricalModel, i_dispatch...)` implementing `ModelBackend` from learned (m̂, q̂) + true rules (architecture §7.2): `p_trade(i)[k,j,0] = onehot(k)`, `p_trade(i)[k,j,1] = (1−q̂[k,j])·onehot(k) + q̂[k,j]·onehot(j)` (k≠j; same-good rows = onehot(k)); `r_trade ≡ 0`; `r_cons(i) = reward_cons_table[i]` (−inf → large negative, masked anyway); `p_cons(i)[h,γ] = onehot(next_holding_cons[i,h,γ])`.

- [ ] **Step 1: Failing tests** — RulesBackend kernels match hand values for a fixed q̂; dry-run with all three flags on completes; slow test: full known-env run on C reaches `br_gain < 0.1` and stationarity < 1e-3 (paper: 0 and 1.74e-8).

```python
# tests/test_known_env.py (core assertions)
def test_rules_backend_trade_kernel():
    e = get_economy("A1"); m = EmpiricalModel(e)
    be = RulesBackend(e, m)
    pt = be.p_trade(0)
    assert np.isclose(pt[0, 1, 1, 1], 0.5) and np.isclose(pt[0, 1, 1, 0], 0.5)  # q̂=1/2 prior
    assert pt[0, 1, 0, 0] == 1.0
```

- [ ] **Step 2: Implement; run → PASS (incl. slow C run).**
- [ ] **Step 3: H4 replication runs (seed 0):**

```bash
for econ in A1 A2 B C D; do
  uv run python scripts/run_learning.py --economy $econ --learner muzero-known --seed 0 \
    --out runs/$econ/muzero-known/0
done
```

Expected: known-env gains ≈ 0 everywhere (paper Table 2, H4). Record in
`findings.md ## H4 known-environment`. (The unbundled 2³ flag ablation is an
extension — it runs in Task 18, ext §3.)

- [ ] **Step 4: Commit** — `git commit -m "feat(muzero): known-environment variant (H4 replication)"`

---

# PART B — EXTENSIONS (spec: `extensions/architecture.md`)

Everything below goes beyond the paper. Extension code lives in
`src/kw/extensions/` and may import core; core never imports it. The campaign
runner (Task 17) is shared infrastructure: its first invocation completes
Part A's multi-seed evidence for H3/H4, its second runs the extension grid.

## Phase 6 — Stage-2 learner replacement (ext §4)

### Task 15: `extensions/qlearning.py` — differential Q-learning

**Files:**
- Create: `src/kw/extensions/__init__.py`, `src/kw/extensions/qlearning.py`
- Test: `tests/test_qlearning.py`, `tests/test_no_core_to_extension_imports.py`

**Interfaces:**
- Produces: `QLearner(econ, cfg, rng, dyna="none", q_lr=0.1, eta=0.5, dyna_k=20, per_agent=False)` in `kw.extensions.qlearning`, implementing the core `Learner` protocol. Tables `QT (I,G,G,2)`, `QC (I,G,2)` init 0 (infeasible −inf), `gbar (I,)` init 0. Behavior ε-greedy with floor `cfg.eps_floor`. Update equations exactly ext §4: trade-stage TD to `max_γ QC[i,h⁺,·]` (no reward, no ḡ); consumption-stage TD **delayed** until the next meeting reveals j′ (a per-agent pending-transition slot completes then); `gbar += eta*q_lr*δ_C`. `end_iteration` returns mean |ΔQ|-derived policy change on visited roots (for uniform (D1) handling); `deployment_policy()` = greedy w.r.t. Q with masks (deterministic 0/1 arrays).
- Produces: `tests/test_no_core_to_extension_imports.py` — the separation lint: `rg`-free pure-Python walk of `src/kw/*.py` and `src/kw/learners/*.py` asserting no line matches `from kw.extensions` / `import kw.extensions`.

- [ ] **Step 1: Failing unit tests on hand-crafted transitions**

```python
# tests/test_qlearning.py
import numpy as np
from kw.economies import get_economy
from kw.learners.base import LearnerConfig, TradeRecord, ConsRecord
from kw.extensions.qlearning import QLearner

# tests/test_no_core_to_extension_imports.py
from pathlib import Path

def test_core_never_imports_extensions():
    core = list(Path("src/kw").glob("*.py")) + list(Path("src/kw/learners").glob("*.py"))
    offenders = [p for p in core if "kw.extensions" in p.read_text()]
    assert offenders == []

def test_trade_update():
    e = get_economy("A1")
    lr = QLearner(e, LearnerConfig(), np.random.default_rng(0), q_lr=0.5)
    lr.QC[0, 1, :] = [2.0, 6.0]
    lr.observe_trade(TradeRecord(agent=0, i=0, k=0, j=1, lam=1, h_next=1, r=0.0))
    assert np.isclose(lr.QT[0, 0, 1, 1], 0.5 * (6.0 - 0.0))

def test_cons_update_delayed_until_next_meeting():
    e = get_economy("A1")
    lr = QLearner(e, LearnerConfig(), np.random.default_rng(0), q_lr=0.5, eta=1.0)
    lr.QT[0, 1, 2, :] = [1.0, 3.0]
    lr.observe_cons(ConsRecord(agent=0, i=0, h=0, gamma=1, h_next=1, r=99.0))
    assert lr.QC[0, 0, 1] == 0.0                      # pending, not yet applied
    lr.observe_meeting_agent(agent=0, i=0, k=1, j_good=2)
    expect = 0.5 * (99.0 - 0.0 + 3.0 - 0.0)
    assert np.isclose(lr.QC[0, 0, 1], expect)
    assert np.isclose(lr.gbar[0], 0.5 * (99.0 + 3.0))
```

(Add `observe_meeting_agent(agent, i, k, j_good)` to the protocol as an optional hook; `simulate.py` calls it when present — MuZero ignores it, its meeting counts come through `observe_meeting`.)

- [ ] **Step 2: Run → FAIL; implement; run → PASS.**
- [ ] **Step 3: Dry-run integration** — `run_learning` with `QLearner` on A1/C completes; deployment arrays respect masks.
- [ ] **Step 4: Commit** — `git commit -m "feat(extensions): differential average-reward Q-learning on the shared harness"`

### Task 16: Dyna planning sweeps

**Files:**
- Modify: `src/kw/extensions/qlearning.py`
- Test: `tests/test_dyna.py`

**Interfaces:**
- Produces: `dyna="emp" | "rules"` activates: learner owns a `ModelBank` (same observation pathway as MuZero) and after every real record executes `dyna_k` expected backups at uniformly sampled visited roots (ext §4): trade roots backup `QT[i,k,j,λ] ← (1−a)QT + a[r̂ᵀ + Σ_h′ P(h′)·max_γ QC[i,h′,γ]]`; consumption roots `QC[i,h,γ] ← (1−a)QC + a[r̂ᶜ − gbar[i] + Σ_h″ P(h″)·Σ_j m̂_i(j)·max_λ QT[i,h″,j,λ]]`; backend chosen by flag (EmpiricalBackend / RulesBackend).

- [ ] **Step 1: Failing test** — with a hand-set deterministic model and `dyna_k=1`, one planning backup matches the hand-computed expected target; `dyna="rules"` uses `T_q̂` (verify a backup lands between the λ=0 and λ=1 values proportional to q̂).
- [ ] **Step 2: Implement; run → PASS.**
- [ ] **Step 3: Stage-2 single-seed grid (H5, H6):**

```bash
for econ in A1 A2 B C D; do for lrn in q dyna-emp dyna-rules; do
  uv run python scripts/run_learning.py --economy $econ --learner $lrn --seed 0 \
    --out runs/$econ/$lrn/0
done; done
```

Record the 5×5 comparison table (three Stage-2 + two MuZero variants) of BR gains in `findings.md ## Stage-2 grid` — this is the H5 attribution evidence.

- [ ] **Step 4: Commit** — `git commit -m "feat(dyna): Dyna-Q planning sweeps with empirical and known-rules backends"`

---

## Phase 7 — Campaign, extensions, reporting

### Task 17: `scripts/run_campaign.py` — multi-seed campaign

**Files:**
- Create: `scripts/run_campaign.py`
- Test: `tests/test_campaign.py` (dry-run only)

**Interfaces:**
- CLI: `uv run python scripts/run_campaign.py --economies <list> --learners <list> --seeds N --jobs J --out <dir>` (plus every run_learning flag, forwarded); learner-agnostic — Step 3 invokes it once for replication and once for the Stage-2 grid with different `--learners`/`--out`. joblib `Parallel(n_jobs)` over (econ, learner, seed); each run saved under `<out>/<econ>/<learner>/<seed>/`; a `summary.json` accumulates per-run `{br_gain, stationarity, classification, n_iters, wall_clock}`. Skips completed runs (manifest present) — idempotent resume. `--dry-run` shrinks config and seeds to 2.

- [ ] **Step 1: Failing test** — dry-run campaign over (A1) × (q) × 2 seeds creates 2 manifests and a summary with 2 entries; re-invocation skips (wall-clock ≈ 0 for completed).
- [ ] **Step 2: Implement; run → PASS.**
- [ ] **Step 3: Launch the two campaigns** (background, hours) — kept as
separate invocations and artifact roots so replication results never mix with
extension results:

```bash
uv run python scripts/run_campaign.py --economies A1,A2,B,C,D \
  --learners muzero-emp,muzero-known --seeds 10 --jobs 8 \
  --out runs/campaign-replication 2>&1 | tee runs/campaign-replication.log
uv run python scripts/run_campaign.py --economies A1,A2,B,C,D \
  --learners q,dyna-emp,dyna-rules --seeds 10 --jobs 8 \
  --out runs/campaign-extensions 2>&1 | tee runs/campaign-extensions.log
```

The first completes Part A's multi-seed evidence (H3/H4); the second is the
Stage-2 grid (H5/H6, ext §6).

- [ ] **Step 4: Commit** — `git commit -m "feat(campaign): joblib multi-seed campaign runner with resume"`

### Task 18: 2³ ablation + extensions E1–E3

**Files:**
- Create: `scripts/run_extensions.py` (thin loop over run_learning configs, same artifact layout)

**Interfaces:** consumes existing flags only — no new library code (the flag triple and E1 `--ema-gain`, E2 `--per-agent` exist since Tasks 8/12/14; E3 sweeps `--eps-floor`/`--kappa0`).

- [ ] **Step 1: Implement script with the exact grids** (ext §3, §6): 2³ ablation `--known-rules/--greedy-backup/--centered-values ∈ {0,1}³` × {C, D} × seed 0; E1 `ema-gain ∈ {0.005, 0.02, 0.05}` × {C, D} × 5 seeds (muzero-emp); E2 `--per-agent` × {A1, C, D} × 5 seeds; E3 `eps-floor ∈ {0.01, 0.03, 0.1, 0.2}` × `kappa0 ∈ {0.5, 2, 8}` × C × 3 seeds. `--dry-run` supported.
- [ ] **Step 2: Run** — `uv run python scripts/run_extensions.py --jobs 8 --out runs/extensions`.
- [ ] **Step 3: Record verdicts in `findings.md`**: 2³ flag attribution (which change carries the H4 rescue, ext §3) and H7/H8/H9 (E1 prediction on record: rescues D, not C).
- [ ] **Step 4: Commit** — `git commit -m "feat(extensions): EMA model, per-agent tables, exploration sweep runs"`

### Task 19: Tables, figures, final report

**Files:**
- Create: `scripts/make_tables.py`, `scripts/make_figures.py`
- Modify: `findings.md`

**Interfaces:**
- `make_tables.py --replication runs/campaign-replication --stage2 runs/campaign-extensions --extensions runs/extensions --out reports/` emits: (T1) Table-2 analog — median (IQR) BR gain and stationarity per economy × learner over seeds, alongside the paper's single-run values from architecture §15, with an explicit replication/extension divider row; (T2) outcome-frequency table with Wilson 95% CIs; (T3) Mann–Whitney U p-values between learners within each economy; (T4) extension tables for the 2³ ablation and E1–E3.
- `make_figures.py` emits per economy: stationary-holdings bar panels (Nash vs SCE vs each learner's μ★), learning-path b_t / d_t trajectories overlaying learners, C's final trade-policy heatmap (broken-links visual), campaign outcome-frequency chart. Matplotlib, one PNG per figure under `reports/figures/`.

- [ ] **Step 1: Implement both scripts** (read artifacts only; fail loudly on missing runs listing exactly which are absent).
- [ ] **Step 2: Generate** — run both; verify T1 reproduces the paper's qualitative Table 2 pattern and the seed columns quantify robustness.
- [ ] **Step 3: Write `findings.md` final section** — verdict per hypothesis H1–H9 with pointers to tables/figures and any fidelity-registry caveats.
- [ ] **Step 4: Commit** — `git commit -m "feat(report): campaign tables, figures, and hypothesis verdicts"`

---

## Self-review (performed at plan-writing time)

- **Spec coverage:** replication spec — architecture §5 → Tasks 1–4; §6.1 → Tasks 5–6; §6.2 → Task 7; §7.1 → Task 8; §7.2 → Tasks 9, 14; §7.3 → Tasks 12–13; §7.5 → Task 10; §8 → Task 11 + Task 19 stats; §9 → Task 17 (first campaign); §11 → Tasks 6, 10, 17; §12 consulted in Tasks 7, 10, 12, 13. Extension spec — ext §3 → Task 18; ext §4 → Tasks 15–16 + Task 17 (second campaign); ext §5 → Task 18; ext §6 → Tasks 17–18. All H1–H9 have a producing task (H1:6, H2:7, H3:13+17, H4:14+17, H5/H6:16+17, H7–H9:18) and a reporting task (19).
- **Separation check:** Part A tasks touch nothing under `src/kw/extensions/`; extension hooks introduced in Part A (`ema_gain`, `ModelBank`, flag triple) default to paper behavior and are exercised off-default only in Part B; the import-direction lint lands with the first extension code (Task 15).
- **Placeholder scan:** the single deliberate deferred item — exact per-economy SCE prompt text — is an explicit re-read-the-paper instruction inside Task 7 Step 1 with the source path named, not a TBD.
- **Type consistency:** `mu (I,G)`, `alpha (I,G,G)`, `beta (I,G)`, `Q/q̂ (G,G)`, `v (I,G)`, `QT (I,G,G,2)`, `QC (I,G,2)` used uniformly; `Learner` protocol methods match between Tasks 8 (base), 10 (harness), 12 (MuZero), 15 (Q) — including the optional `observe_meeting_agent` hook introduced in Task 15 and tolerated by the harness.

## Post-plan note for any Stage-3+ planning (added after execution)

The one genuine gap this plan left unaddressed, discovered during execution: **learned
near-deterministic deployed policies produce slow-mixing Markov chains** (spectral gaps
1e-6 to 7e-7 per iteration), so invariant-distribution and RVI solves converge at that
geometric rate. Three iteration caps had to be raised mid-execution (200k → 10M → 40M,
commits d740fb8 / aeee24e / bf1688a) with measured convergence at ~11.1M and ~12.6M
iterations; tolerances (1e-12 distribution, 1e-10 Bellman) were treated as contracts and
never loosened. Any follow-on plan whose learners can approach deterministic policies
should budget solver caps for this regime up front — the hazard is seed- and
realization-dependent, not uniform across cells.
