# Architecture: Kiyotaki–Wright Replication

**Scope: replication only.** This document is the complete spec for reproducing
Sargent & Yang (2026), *"How Should I Trade? Stationary Nash, Self-Confirming,
and Learned Equilibria in Kiyotaki–Wright Exchange Economies"*
(`MMS Alpha Research.pdf`, cited below as "the paper") — the classic
Kiyotaki–Wright environment, both equilibrium solvers, both learner variants,
and the diagnostics that test the paper's claims. Everything **beyond** the
paper — Stage-2 learner replacement (Q-learning/Dyna), the known-environment
unbundling ablation, and the EMA / per-agent / exploration extensions — is
specified separately in **`extensions/architecture.md`** and never leaks into
this document. Equation and algorithm labels — (B1)–(B4), (F1)–(F4), (S1)–(S3),
(L1)–(L3), (P1)–(P2), (R), (D1), Algorithms 1–4, Table 1/2 — refer to the paper.

## 1. Purpose and scope

Build a single Python codebase that:

1. **Replicates** the paper's three computational procedures in all five economies
   (A1, A2, B, C, D):
   - **Solver 1 (Nash):** relative-value-iteration (RVI) damped best-response
     iteration for stationary Nash equilibrium (Algorithm 1).
   - **Solver 2 (SCE):** restricted-information self-confirming-equilibrium
     fixed-point solver (Algorithm 2).
   - **Learner:** tabular MuZero-style PUCT self-play (Algorithms 3–4), in both
     the *fully empirical* and *known-environment* variants.
2. **Tests the paper's conclusions rigorously**: multi-seed re-runs with outcome
   classification, plus the paper's own diagnostics (complete-information
   best-response gain, stationarity residual) computed independently by Solver-1
   machinery.

Solvers 1 and 2 are deliberately kept as two separate iteration/fixed-point
procedures (they answer different questions and use different belief updates);
they share the Bellman and law-of-motion kernels but are never merged.

**Separation rule (replication vs extensions):** every core module implements
paper-exact behavior *by default*. An extension is allowed to exist only as
(a) a non-default flag on a core module (e.g. `ema_gain=0.0`, `per_agent=False`
— hooks listed in `extensions/architecture.md`), or (b) code under
`src/kw/extensions/` speced in `extensions/architecture.md`. A replication run
never depends on extension code, and default-flag runs are bit-identical to a
build with the hooks removed.

## 2. Claims under test

| ID | Claim (source) | Test |
|----|----------------|------|
| H1 | Algorithm 1 reproduces the KW/MMS stationary Nash equilibria: A1 fundamental (good 1 money), A2 speculative branch sustainable iff `s3−s2 < (π3h−π2h)·ū/3` flips at ū=500, B dual branches via prompts, C fiat SNE at M0=0.32, D benchmark (paper §5, §7) | Analytic A2 condition; BR gain ≈ 0; holdings match paper Figures 2/4/6/8/10 |
| H2 | Algorithm 2 with misdirectional prompts yields non-Nash SCEs (A1 autarky, C low-trade, D type-1 autarky) with residuals s,b,d ≈ 0 but positive complete-info gain (paper §6.2, §7) | Residual triple + independent BR-gain check |
| H3 | Fully empirical learner: A1, B reach Nash; A2 reaches fundamental-shaped holdings with gain ≈ 21; C collapses to a low-trade non-Nash outcome (fiat links broken: α₂(0,2)≈0, α₃(1,2)≈0); D plateaus at gain ≈ 1.7 (paper Table 2, §7.4–7.5) | Single-seed replication, then seed-robustness frequencies |
| H4 | Known-environment learner reaches gain ≈ 0 in all five economies (paper Table 2) | Same |

H1–H4 are the replication claims — everything here traces to a specific
paper statement. The extension hypotheses H5–H9 (learner-replacement
attribution, EMA, sharing, exploration) are defined in
`extensions/architecture.md` §2. The paper's own single-run numbers (reference
targets) are collected in §15.

## 3. Design principles

- Unix paradigm: small orthogonal modules, explicit flags (one flag = one
  parameter), loud errors, no fault-tolerance around user error.
- Reuse: NumPy for arrays, joblib for the seed campaign, pytest for tests,
  matplotlib for figures. No custom frameworks.
- Solvers and diagnostics are **pure functions** `(Economy, config) → artifact`;
  learners implement one shared interface (the fairness contract, §7.5) — this
  is also what lets the extension learners of `extensions/architecture.md` slot
  into the identical simulator later without touching replication code.
- Every run writes a manifest (config, git SHA, seed) + `.npz`/JSON artifacts;
  figures and tables are built only from artifacts, never recomputed inline.
- Determinism: one master seed → named `numpy.random.Generator` streams
  (`spawn`); float64 everywhere; all argmax tie-breaks fixed to lowest index.

## 4. System overview

```
economies.py ──► distributions.py ◄──► bellman.py
     │                  │                  │
     │           ┌──────┴────────┬─────────┴────────┐
     │           ▼               ▼                  ▼
     │      nash_solver.py  sce_solver.py    diagnostics.py ◄─── every policy
     │           (Alg 1)        (Alg 2)      (BR gain, residuals,
     │                                        classifier)
     ▼
simulate.py (population loop, Alg 3) ──► empirical_model.py (M̂)
     │                                          │
     ▼                                          ▼
learners/ (PUCT/MuZero)  ◄───────────── planner.py (L2/L3 DP + P1/P2)
     │
     ▼
artifacts (.npz + manifest.json) ──► scripts/make_figures.py, make_tables.py
```

Data flow: an `Economy` feeds either a solver (returns policy + invariant law +
residuals) or the simulator+learner (returns deployment policy + learning-path
traces). Every resulting policy — regardless of origin — is scored by the same
`diagnostics.py` functions. That solver-independence of the diagnostic suite is
the linchpin of "testing the conclusions rigorously."

## 5. Domain model

### 5.0 Background — the classic Kiyotaki–Wright environment

The paper's economies are instances of Kiyotaki & Wright (1989). A continuum
(here: N agents) of |I| types trades indivisible goods in pairwise random
matching; type i derives utility ū_i only from consuming good i, immediately
produces its production good f(i) after consuming, and pays a per-period
storage cost s(g) on whatever single good it holds. Because f(i) ≠ i, all
consumption requires trade, and because storage costs are ranked, agents face
KW's central trade-off between **storability** (hold the cheap-to-store good)
and **marketability** (hold the good your counterparties accept). Stationary
equilibria come in two flavors:

- **Fundamental**: agents accept a non-consumption good only if it is cheaper
  to store than their current holding — the lowest-storage-cost good endogenously
  becomes commodity money (Economy A1, where good 1 with s=0.1 circulates).
- **Speculative**: agents accept a *more expensive* good because its superior
  marketability shortens the wait to consume — sustainable exactly when the
  utility flow at stake outweighs the storage premium, the KW condition
  `s₃ − s₂ < (π₃ₕ(1) − π₂ₕ(1))·ū/3` (Economy A2 flips it by raising ū from 100
  to 500; Economy B supports both a fundamental and a speculative branch).

Economy C adds an intrinsically useless, zero-storage-cost good 0 in fixed
supply M0 that cannot be consumed or produced — **fiat money** — whose value is
purely a self-fulfilling acceptance pattern. Economy D scales barter to five
types/goods with no closed-form benchmark. The paper's twist on this classic
setup is game-theoretic and computational: it recasts KW's steady states as
average-reward stationary Markov Nash equilibria (§5.4), weakens them to
restricted-information self-confirming equilibria (§6.2), and asks which of
these a population of model-based learners actually reaches (§7).

### 5.1 Economy primitives (`economies.py`)

```python
@dataclass(frozen=True)
class Economy:
    name: str
    goods: tuple[int, ...]          # G; includes 0 for fiat in C
    n_types: int                    # |I|
    nu: np.ndarray                  # (|I|,) population shares
    f: np.ndarray                   # (|I|,) production map, f[i] in G
    s: np.ndarray                   # (|G|,) storage costs indexed by good
    ubar: np.ndarray                # (|I|,) consumption utilities
    gamma_feasible: np.ndarray      # (|I|,|G|) bool: 1 ∈ Γ_i(h)?
    lambda_feasible: np.ndarray     # (|I|,|G|,|G|) bool: 1 ∈ Λ_i(k,j)?
    fiat_good: int | None           # 0 in C, else None
    M0: float | None                # aggregate fiat share (C: 0.32)
```

Registry (paper §5) — types are 1-indexed in the paper, 0-indexed in code:

| Econ | G | ν | f | s | ū | Notes |
|------|---|---|---|---|---|-------|
| A1 | {1,2,3} | 1/3 each | f(1)=2, f(2)=3, f(3)=1 | (0.1, 1, 20) | 100 | fundamental SNE, good 1 = money |
| A2 | {1,2,3} | 1/3 | same as A1 | same | **500** | speculative branch sustainable |
| B | {1,2,3} | 1/3 | f(1)=3, f(2)=1, f(3)=2 | (1, 4, 9) | 100 | two SNE branches |
| C | {0,1,2,3} | 1/3 | f(1)=2, f(2)=3, f(3)=1 | (0, 9, 14, 29) | 100 | Γ_i(0)={0}; M0=0.32 |
| D | {1..5} | 1/5 | f(1)=3, f(2)=4, f(3)=5, f(4)=1, f(5)=2 | (1, 4, 9, 20, 30) | 200 | MMS five-good barter |

### 5.2 Environment mechanics (`environment.py`)

Pure functions of the period structure: reward `r_i(h, γ) = γ(ū_i·1{h=i} −
s(f(i))) + (1−γ)(−s(h))` (R); next-holding map `c_i(h, γ) = f(i) if γ else h`;
trade resolution `χ = 1{k≠j and both offer}`. Policies are arrays
`alpha[i,k,j] ∈ [0,1]`, `beta[i,h] ∈ [0,1]` with feasibility masks applied
(β_i(0)=0 in C).

### 5.3 Population law (`distributions.py`)

- Aggregates: `μ(k) = Σ_i ν_i μ_i(k)`, `η_i(k) = ν_i μ_i(k)/μ(k)` (B1),
  reciprocation `Q(k,j) = Σ_ℓ η_ℓ(j) α_ℓ(j,k)` (B2), with Q(k,j)=0 when μ(j)=0.
- One-step map `F_σ`: flows `F_i(k,j) = μ_i(k) μ(j) α_i(k,j) Q(k,j)` (F1),
  post-trade μ⁺ (F2), consumption/production step (F3). Note F_σ is quadratic in
  μ (μ enters Q and the matching marginal) — fixed points via damped forward
  iteration to `‖F_σ(μ) − μ‖∞ ≤ ε_dist` (default 1e-12), starting from a caller-
  supplied μ0 (this matters when the chain has several recurrent classes, §12).
- Invariant: fiat stock `Σ_i ν_i μ_i(0) = M0` is conserved by F_σ in Economy C
  (good 0 is neither consumed nor produced) — enforced at init, asserted along
  iterates.

### 5.4 Average-reward Bellman kernel (`bellman.py`)

Everything downstream reuses these three functions:

- `rvi_solve(econ, i, mu, Q) -> (g_star, V, W)`: relative value iteration on
  (B3)–(B4) for one type in a fixed environment, bias normalized at a reference
  good `V_i(k_ref)=0` (k_ref = f(i), see §12).
- `greedy_policy(econ, i, mu, Q, V, W) -> (alpha_i, beta_i)`: pointwise (BR-e)
  offer iff `Q(k,j)·(W(j) − W(k)) > 0` and (BR-c) consume iff
  `ū_i(h) − s(f(i)) + V(f(i)) > −s(h) + V(h)`, ties → deterministic convention
  (§12), feasibility masks applied.
- `policy_eval(econ, i, sigma_i, mu, Q) -> (g_sigma, V_sigma)`: the linear
  evaluation equations of §3 for a *fixed* policy — used by Algorithm 1's gain
  gap and by the best-response-gain diagnostic.

Unichain checks are numerical: build the induced holding chain and verify a
single recurrent class; warn (not fail) otherwise, since some SCE policies are
legitimately multichain (paper fn. 4).

## 6. Solvers

### 6.1 Solver 1 — stationary Nash (Algorithm 1, `nash_solver.py`)

Damped best-response iteration. State: policy profile σ_m. Per iteration:
1. Iterate F_σ to (μ_m, Q_m).
2. Per type: `rvi_solve` → (g★, V, W); `greedy_policy` → σ̄; `policy_eval`(σ_m)
   → gain gap `e_i = g★_i − g^σ_i`.
3. Residuals `e_m = max_i e_i`, `d_m = ‖F_σ(μ_m) − μ_m‖∞`; stop when
   `e_m ≤ ε_BR` and `d_m ≤ ε_dist`.
4. Damped update `σ_{m+1} = (1−ρ)σ_m + ρ σ̄_m` (default ρ = 0.3, flag).

Returns `NashArtifact(alpha, beta, mu, Q, g, residual_history)`.

**Directional prompts** (initial policies, paper §6.1): default α=1/2,
β_i(i)=1 else 0; A2-speculative and B-speculative full pure patterns (A2: type 1
offers 2→3 and 3→1; B: type 2 offers 1→3 and 3→2); C money-acceptance:
α_i(k,0)=1, α_i(k,i)=1, 0 elsewhere. Stored declaratively in a prompt registry
(`prompts.py`) shared with Solver 2 so each named prompt is a data object, not
code.

### 6.2 Solver 2 — restricted-information SCE (Algorithm 2, `sce_solver.py`)

A genuinely different fixed point: state is (σ_m, θ_m, μ_m) with subjective
environments θ_i = (m_i, q_i). Per iteration:
1. Per type, solve the *subjective* Bellman (S2)–(S3) under θ_i — reuses
   `rvi_solve` with transition kernel `T_{q_i}` (§4.1) and meeting law m_i in
   place of (μ, Q) — and damp σ toward the subjective greedy policy.
2. Recompute the objective invariant μ and Q^σ.
3. Belief update **on-path only**: `m_i ← μ`; `q_i(k,j) ← Q^σ(k,j)` for
   (i,k,j) in `O_m = {(i,k,j): μ_i(k) μ(j) σ^e_i(1|k,j) > 0}`; off-path q_i
   frozen.
4. Residuals: subjective optimality `s_m = max_i (g^θ_i − g^{σ,θ}_i)`, on-path
   belief `b_m = max_{O_m} |q_i − Q^σ|`, stationarity `d_m`; stop when
   `max(s_m, b_m) ≤ ε_SCE` and `d_m ≤ ε_dist`.

Returns `SCEArtifact(alpha, beta, mu, theta, residuals)` **plus** the
complete-information BR gain computed by `diagnostics.py` — the paper's
certificate that the SCE is not Nash.

**Misdirectional prompts**: skeptical off-path beliefs, e.g. q_i ≡ 0 with a
no-trade initial policy (A1 autarky); per-economy constructions are extracted
verbatim from paper §7 during implementation and added to the prompt registry
(§12 item 6).

## 7. Learning stack

### 7.1 Empirical model M̂ (`empirical_model.py`)

Count tables, all initialized at prior count κ0 = 2.0 (`A_acc` at κ0/2 so
q̂⁰ = 1/2; reward sums at 0 so r̂⁰ = 0):

| Table | Shape | Ratio it feeds |
|-------|-------|----------------|
| `C[i,j]` meetings | (I,G) | m̂_i(j) = C_i(j)/Σ_ℓ C_i(ℓ) |
| `O[k,j]`, `A_acc[k,j]` offers/accepted (pooled over offerer type) | (G,G) | q̂(k,j) = A_acc/O |
| `N_T[i,k,j,λ,h′]` | (I,G,G,2,G) | P̂ᵀ_i(h′\|k,j,λ) |
| `N_C[i,h,γ,h′]` | (I,G,2,G) | P̂ᶜ_i(h′\|h,γ) |
| `B_T,S_T[i,k,j,λ]`, `B_C,S_C[i,h,γ]` | — | r̂ᵀ, r̂ᶜ = S/B |

Record types `TradeRecord(i,k,j,λ,h⁺,r_T)` / `ConsRecord(i,h,γ,h⁺,r_C)` applied
**online** as they arrive (paper §6.3.1). Same-good matches (k=j) append the
trade transition (h⁺=k) but never touch O/A_acc; r_T ≡ 0.

**Extension hooks (paper-exact defaults):** the model exposes `ema_gain`
(default 0.0 = the paper's cumulative counts) and a `ModelBank` indirection for
table ownership (default: one population-shared bank, as in the paper). Both
exist solely for the extension experiments and are speced in
`extensions/architecture.md` (E1, E2); at their defaults the model reproduces
the paper's M̂ exactly.

### 7.2 PUCT planner (`planner.py`)

Key implementation fact (paper p. 34): within one search the model is frozen
and the action value ℓ_x is visit-independent, so `L̄_x` never changes after its
first visit. Therefore the planner is exactly:

1. **DP tables** V̂(h,d), Ŵ(h,d) for d = 0..H via the mutual recursion
   (L2)–(L3): Ŵ(h,d) = (1−β(h))Ψᶜ(h,0;d) + β(h)Ψᶜ(h,1;d);
   V̂(h,d) = Σ_j m̂(j)[(1−α(h,j))Ψᵀ(h,j,0;d) + α(h,j)Ψᵀ(h,j,1;d)];
   Ψᵀ(k,j,λ;d) = r̂ᵀ + Σ_{h′} P̂ᵀ(h′|k,j,λ)·Ŵ(h′,d);
   Ψᶜ(h,γ;d) = r̂ᶜ + Σ_{h′} P̂ᶜ(h′|h,γ)·V̂(h′,d−1);
   terminal V̂(·,0) = Ŵ(·,0) = vⁿ. Cost O(|G|²·H) per (type, model snapshot).
2. **Deterministic bandit loop**: normalize the two root values to S_x ∈ [0,1]
   (S_x = 1/2 when tied), run the literal `n_sims` selection steps of (P1)
   `argmax_x S_x + c_puct·p_x·√(N_tot+1)/(1+N_x)` (ties → lowest index),
   return visit policy π(x) ∝ N_x^{1/τ} (P2) and the max root action value.

The loop is kept literal (not closed-form) for fidelity, and it is cheap.
Memoization contract (paper §6.3.5): the planner is constructed once per
iteration; the **first** query of a root computes and caches (π, max-value) from
the model *as of that moment*; later queries in the same iteration reuse the
cache even though M̂ has since moved.

Two model backends behind one lookup interface:
- `EmpiricalBackend`: (m̂_i, P̂ᵀ, P̂ᶜ, r̂ᵀ, r̂ᶜ) from M̂ (fully empirical case).
- `RulesBackend` (M̂_rules, paper §6.4): m̂_i and q̂ learned; trade reward ≡ 0,
  trade kernel = T_q̂, consumption reward = true r_i, consumption kernel =
  deterministic c_i.

Two backup modes:
- `prior` (default): future own actions weighted by current priors α, β as in
  (L2)–(L3).
- `greedy` (known-env change #2): Ŵ(h,d) = max_γ Ψᶜ(h,γ;d) and
  V̂(h,d) = Σ_j m̂(j) max_λ Ψᵀ(h,j,λ;d).

Deployment mode `PUCT_dep`: horizon H_dep=14, 96 sims, τ_dep=0.35, no records.

### 7.3 MuZero-style learner (`learners/muzero.py`, Algorithms 3–4)

One class, three orthogonal flags. The paper studies exactly two settings —
all-off (fully empirical) and all-on (known-environment, §6.4's three bundled
changes) — and those two are the replication targets. That the flags *also*
enable an unbundled 2³ ablation is an extension (`extensions/architecture.md`
§3), not part of replication.

| Flag | off (fully empirical) | on (known-environment) |
|------|----------------------|------------------------|
| `known_rules` | EmpiricalBackend | RulesBackend |
| `greedy_backup` | prior-weighted (L2)–(L3) | greedy interior backup |
| `centered_values` | v ← (1−ζ)v + ζ V̄ | v ← (1−ζ)v + ζ(V̄ − V̄(k_ref)), v(k_ref)=0 |

Iteration loop (Algorithm 3): R rounds of {uniform random pairing → meeting
counts → PUCT_T at both roots → sample offers → exchange iff both offer and
k≠j → records online} then a population-wide consumption pass (PUCT_C, sample,
physical transition, record). End of iteration (Algorithm 4): root-wise mean
visit policies π̄ and mean max-values V̄ (unvisited roots keep current values);
updates `α ← clip_ε[(1−ζ_n)α + ζ_n π̄]`, same for β, v; ζ_n = max(0.01,
0.45·0.94ⁿ); clip_ε projects onto [ε, 1−ε] (0 for infeasible actions).
Stopping (D1): mean |Δpolicy| over *visited* roots ≤ δ=0.003 for p=8
consecutive iterations, after n_min=100, cap N_max=180. Then deployment
extraction, μ★ = Fix F_σ★ (power iteration from the terminal population
distribution), Q★ from (B2).

All Table 1 defaults are encoded as a frozen `LearnerConfig` dataclass:
N=1800, R=35, n_sims=32, H=10, c_puct=1.5, τ=1.0, ζ0=0.45, decay 0.94,
ζ_min=0.01, κ0=2.0, ε=0.03, n_min=100, δ=0.003, p=8, N_max=180, H_dep=14,
n_sims_dep=96, τ_dep=0.35, eval_rounds=120, eval_record_interval=5.
Initialization: α⁰=1/2, β⁰=1/2 (0 infeasible), v⁰=0; each agent holds f(i);
in C the fiat stock is spread evenly: 0.32·(N/|I|) agents per type hold good 0.

### 7.4 Extension learners (not replication)

Learners other than MuZero — model-free differential Q-learning and Dyna-style
planning, the "Stage-2 learner replacement" — are extensions. Their update
equations, comparison grid, and hypotheses live in `extensions/architecture.md`
§4; their code lives in `src/kw/extensions/`. They plug into the simulator
exclusively through the §7.5 protocol below, which is why they can be added
without modifying any module in this document.

### 7.5 Learner interface and fairness contract (`learners/base.py`, `simulate.py`)

```python
class Learner(Protocol):
    def trade_action(self, i, k, j, rng) -> int
    def cons_action(self, i, h, rng) -> int
    def observe_trade(self, rec: TradeRecord) -> None
    def observe_cons(self, rec: ConsRecord) -> None
    def end_iteration(self, n: int) -> float        # returns Δ̄_n for (D1)
    def deployment_policy(self) -> tuple[np.ndarray, np.ndarray]  # (alpha*, beta*)
```

`simulate.py` owns everything else — population init, matching, trade
resolution, stopping rule, deployment extraction, fixed-policy evaluation (120
rounds, record every 5), learning-path tracing — and is **identical across all
learners**. Fairness contract: same N, R, N_max, matching-RNG construction,
ε floor, and evaluation pipeline; the only thing that varies is the `Learner`.
Experience budgets are matched by construction (same number of environment
interactions); compute is *not* matched (PUCT does more per decision) — we
report wall-clock per config rather than pretending parity.

## 8. Diagnostics and evaluation (`diagnostics.py`)

- `br_gain(econ, alpha, beta, mu=None) -> (b, per_type)`: the paper's headline
  metric `b = max_i [g_i^BR(μ,σ) − g_i^σ(μ,σ)]` at the invariant law of σ —
  computed with `rvi_solve` + `policy_eval`, fully independent of any learner.
- `stationarity(econ, alpha, beta, mu) -> d = ‖F_σ(μ)−μ‖∞`.
- Learning-path traces b_t^sp, d_t^sp at every iteration (paper §7 preamble).
- Distributional distances: TV(μ★, μ_ref) against each Nash branch and each SCE
  artifact; per-type holding matrices; trade volume per round.
- Approximate-SCE certificate: build θ̂ from a learner's final (m̂, q̂), check
  subjective-optimality residual s and on-path belief residual b_belief of the
  deployment policy — this is what lets us say "trapped in an SCE" rather than
  just "not Nash".
- **Outcome classifier** (per run):
  `NASH` if b ≤ 0.05 and d ≤ 1e-3;
  `SCE_TRAP` if b > 0.05, d ≤ 1e-3, s ≤ 0.05, b_belief ≤ 0.02;
  `UNSETTLED` otherwise. Thresholds are flags; sensitivity reported.
- **Statistics:** ≥10 seeds per (economy × config); outcome frequencies with
  Wilson 95% intervals; median/IQR of b★; Mann–Whitney U between learner
  configs within an economy. Nothing fancier — the effect sizes in Table 2 are
  orders of magnitude, so the statistics only need to establish seed-robustness.

## 9. Experiment matrix (replication)

- Solver benchmarks: every (economy × prompt) pair for Solver 1 and Solver 2
  (deterministic — no seeds needed). Tests H1, H2.
- Learning replication: single-seed qualitative replication first (paper's
  setting), then 5 economies × {muzero-emp, muzero-known} × 10 seeds = 100 runs
  with outcome classification. Tests H3, H4.

The extension matrix (Stage-2 grid, 2³ ablation, E1–E3) is in
`extensions/architecture.md` §6 and runs through the same campaign runner.

## 10. Repository layout

```
pyproject.toml                  # uv-managed, python ≥3.12, deps: numpy, joblib,
                                #   matplotlib, pytest (dev), ruff (dev)
src/kw/                         # REPLICATION (this document)
  economies.py    environment.py    distributions.py    bellman.py
  prompts.py      nash_solver.py    sce_solver.py
  empirical_model.py    planner.py
  learners/{__init__,base,muzero}.py
  simulate.py     diagnostics.py    artifacts.py
src/kw/extensions/              # EXTENSIONS (see extensions/architecture.md)
  __init__.py     qlearning.py
extensions/
  architecture.md               # spec for everything beyond the paper
scripts/
  compute_nash.py  compute_sce.py  run_learning.py  run_campaign.py
  run_extensions.py  make_figures.py  make_tables.py
tests/            # mirrors src/kw module names
runs/              # artifacts (gitignored except manifests of reported runs)
architecture.md   plan.md   findings.md (written as results land)
```

Module responsibility = its section above; no module imports a sibling except
along the §4 arrows. `src/kw/extensions/` may import from `src/kw/`; nothing in
`src/kw/` imports from `extensions` (enforced by a lint-level test). `scripts/`
are thin CLI shims (argparse, explicit flags, `--dry-run` runs a miniature
config N=60, R=5, N_max=6 through the full pipeline).

## 11. Numerical conventions & reproducibility

- float64; tolerances ε_dist=1e-12 (invariant law), ε_BR=1e-9 (Nash),
  ε_SCE=1e-9; damping ρ=0.3 default.
- RNG: `np.random.default_rng(master_seed).spawn(...)` → named streams
  (matching, trade-sampling, consumption-sampling, init). Same master seed ⇒
  bit-identical run.
- Artifacts: `runs/<econ>/<config>/<seed>/manifest.json` + `arrays.npz` +
  `trace.json` (b_t, d_t, holdings path, Δ̄_n).
- Every reported figure/table names the artifacts it was built from.

## 12. Fidelity-risk registry (underdetermined paper details)

Decisions the paper does not fully pin down. Each gets a documented choice and,
where cheap, a sensitivity check:

1. (P1)/(BR) argmax ties → lowest action index.
2. Within-round processing order of matched pairs = order returned by the
   matching draw; records append in that order.
3. Planner memoization vs online model updates: cache at first root query per
   iteration (as stated on paper p. 37–38).
4. "Reward sums initialized consistently": r̂⁰ = 0 ⇒ S⁰ = 0 with B⁰ = κ0.
5. Reference good k_ref = f(i) per type (only shifts bias normalization).
6. Exact per-economy misdirectional prompt constructions: extract from §7 text
   at implementation time; default template is q_i ≡ 0 off-path + no-trade σ0.
7. Multichain Fix F_σ★: power-iterate from the terminal population distribution
   (selects the paper's "recurrent class generated by the stated construction").
8. Economy C integer init: 600 agents/type, 192 of them holding good 0.
9. Δ̄_n compares post-clip stored tables.
10. Whether q̂ feeds the empirical planner: it does not — P̂ᵀ carries acceptance
    implicitly; q̂ is used by RulesBackend only (paper §6.3.1 last sentence).
11. Deployment PUCT uses tables at n★ exactly; no exploration floor applied to
    the reported (α★, β★) beyond what τ_dep leaves.
12. Greedy-backup interpretation as in §7.2 (max over own actions inside the DP
    recursions; root action still via visit counts).

Any replication mismatch gets triaged against this list first.

## 13. Compute budget

Per PUCT query: O(|G|²·H) DP + 32-step loop ⇒ microseconds. Per iteration:
35 rounds × (900 pair-meetings + 1800 consumption decisions), all root-memoized
⇒ the cost is dominated by record bookkeeping; a run (≤180 iterations) is
minutes in plain NumPy (decisions batched by root). Replication campaign ≈ 100
learning runs ⇒ an hour or two on a laptop with joblib (extensions bring the
project total to ≈ 700 runs, still a few hours); no GPU, no cluster.

## 14. Alternatives considered

- **JAX / Julia** instead of NumPy: rejected — tabular problem is tiny; JIT
  overhead and ecosystem friction buy nothing. `vmap` over seeds specifically
  was considered and rejected: the (D1) stopping rule ends different seeds at
  different iterations, so lockstep batching either wastes compute (run all to
  N_max) or needs masking machinery; the paper-prescribed *online,
  order-dependent* count updates and per-iteration planner memoization are
  dict/mutation-shaped, not `lax.scan`-shaped; and joblib process-parallelism
  already saturates the ~700-run campaign in hours on CPU. Revisit only if run
  counts grow ~100× (then: pure-JAX inner simulator with fixed N_max + masked
  stopping, vmap over seeds).
- **Literal tree search** for PUCT: rejected — paper proves root values are
  visit-independent; DP + literal selection loop is exact and faster. The
  selection loop itself is kept literal (not closed-form) to preserve count
  fidelity.
- **Separate classes per learner variant**: rejected in favor of three flags on
  one MuZero class — the variants differ in exactly three orthogonal
  components, and the flags give the 2³ ablation.
- **Merging Solvers 1 and 2** into one generic fixed-point engine: rejected
  (explicit user requirement and good sense — different belief updates,
  different residuals; they share kernels, not control flow).
- **wandb tracking**: local artifacts + matplotlib are primary; a `--wandb`
  flag on `run_campaign.py` is optional sugar, never a dependency.

## 15. Paper reference targets

| Quantity | Fully empirical | Known-env |
|----------|-----------------|-----------|
| BR gain A1 | 1.50e-12 | 0 |
| BR gain A2 | 21.3 | 0 |
| BR gain B | 0 | 0 |
| BR gain C | 12.8 | 0 |
| BR gain D | 1.71 | 7.11e-15 |
| Stationary distance C | 0.130 | 1.74e-8 |
| Stationary distance D | 0.173 | 8.69e-9 |

Qualitative targets: A1 learned holdings match RVI entry-by-entry (agg. good 1
≈ 0.49, good 3 ≈ 0.18 at convergence; iterations 1–20 nearer autarkic SCE,
21–107 nearer SNE); C final empirical policy has α₂(0,2)=0.001, α₂(2,0)=1.000,
α₃(1,2)≈0 (broken fiat links); A2 condition flips between ū=100 and ū=500
(s₃−s₂ = 19 vs ū/6 ∈ {16.7, 83.3}). Multi-seed runs test the *claims*, not
these exact single-seed numbers.
