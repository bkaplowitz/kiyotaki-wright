# Architecture: Extensions Beyond the Paper

Companion to the root `architecture.md` (the replication spec). This document
owns everything **not** in Sargent & Yang (2026): the Stage-2 learner
replacement, the known-environment unbundling ablation, and the three
learning-dynamics extensions. Section references of the form "core §N" point at
the root document; equation labels still refer to the paper.

## 1. Relationship to the replication

- Extension code lives in `src/kw/extensions/`; it may import from `src/kw/`
  but nothing in `src/kw/` imports back (lint-level test enforces this).
- Some extensions are non-default flags on core modules rather than new code —
  the complete list of such hooks is: `ema_gain` (empirical model, default 0.0),
  `ModelBank` size / `per_agent` (empirical model + learners, default shared),
  the MuZero flag triple `known_rules`/`greedy_backup`/`centered_values` when
  set to anything other than all-off or all-on, and off-default `eps_floor` /
  `kappa0`. At default values every hook is paper-exact (core §1 separation
  rule).
- Every extension learner implements the core §7.5 `Learner` protocol and runs
  on the unmodified simulator under the same fairness contract: identical N, R,
  N_max, matching-RNG construction, ε floor, and evaluation pipeline.
  Experience budgets are matched by construction (same number of environment
  interactions); compute is *not* matched (PUCT does more per decision) — we
  report wall-clock per config rather than pretending parity.
- Every extension run is scored by the same `diagnostics.py` metrics and
  outcome classifier (core §8), so replication and extension results are
  directly comparable.

## 2. Extension hypotheses

Continues the H-numbering from core §2 (H1–H4 are the replication claims).

| ID | Claim (ours unless noted) | Test |
|----|---------------------------|------|
| H5 | *Attribution:* the known-environment rescue (H4) comes mainly from model contents (known one-step rules), not planner sophistication — Dyna-rules should behave like known-env PUCT, Dyna-emp like empirical PUCT | Stage-2 grid §4 |
| H6 | Model-free Q-learning (realized payoffs only, EWA δ≈0) is at least as prone to SCE traps as the empirical learner | Stage-2 grid §4 |
| H7 | *(paper §8 conjecture)* EMA/constant-gain model fixes Economy D's stale-anchor plateau but **not** Economy C's missing-data collapse | E1, §5.1 |
| H8 | Population-shared tables are not the cause of the traps; per-agent learners land in the same attractor class, slower | E2, §5.2 |
| H9 | Economy C's collapse is structural (multi-link reciprocal chain, joint-exploration ~ε² per link), not cured by moderately larger ε or weaker priors | E3, §5.3 |

## 3. Known-environment unbundling (2³ ablation)

The paper's §6.4 known-environment variant bundles three changes; the core
MuZero learner implements them as orthogonal flags (core §7.3). The paper only
reports all-off vs all-on. We run the full 2³ grid on the two failure economies
(C and D) to attribute the rescue: which flag (or interaction) closes the BR
gap? Prior expectation: `known_rules` (the model-contents change) dominates,
consistent with H5; `centered_values` mainly stabilizes value scales;
`greedy_backup` sharpens horizons. Single seed per cell first, spot re-seeds
(3) on any cell whose classification differs from its neighbors.

## 4. Stage-2 learner replacement (`src/kw/extensions/qlearning.py`)

The comparison grid — {planner: none / Dyna / PUCT} × {model: empirical /
known-rules}, with model-free occupying the planner-free row and the two PUCT
rows being the replication learners (included for comparison, produced by core
code):

| Config | Planner | Model | Purpose |
|--------|---------|-------|---------|
| `q` | none | none (realized payoffs only) | EWA δ≈0 pole; H6 |
| `dyna-emp` | Dyna sweeps | M̂ empirical | planner w/o knowledge |
| `dyna-rules` | Dyna sweeps | M̂_rules | knowledge w/o PUCT; H5 |
| `muzero-emp` | PUCT | M̂ empirical | paper baseline (core §7.3) |
| `muzero-known` | PUCT | M̂_rules (+2 flags) | paper rescue (core §7.3) |

**Tables:** per type (shared, like the paper; `--per-agent` optional):
`QT[i,k,j,λ]`, `QC[i,h,γ]`, scalar gain estimate `gbar[i]`. Behavior policy:
ε-greedy with the same floor ε=0.03. Deployment policy: greedy w.r.t. Q.

**Differential Q-learning updates** (average-reward, Wan–Naik–Sutton 2021,
adapted to the two-stage period; ḡ subtracted once per period at the
consumption stage — design decision, documented):

```
trade stage    δ_T = max_γ QC[i,h⁺,γ] − QT[i,k,j,λ]              (r_T = 0)
               QT  += a·δ_T
consumption    δ_C = r_C − gbar[i] + max_λ′ QT[i,h⁺⁺,j′,λ′] − QC[i,h,γ]
(delayed until next meeting reveals j′)
               QC  += a·δ_C ;  gbar[i] += η·a·δ_C
```

Learning rate `a` constant 0.1 by default (`--q-lr`), visit-count decay
optional (`--q-lr-visit`); η = 0.5.

**Dyna:** after each real record, K=20 (`--dyna-k`) expected backups at roots
sampled uniformly from the visited-root set, using the same `ModelBackend`
interface as the core planner: emp backend backs up through (P̂ᵀ, P̂ᶜ, r̂, m̂_i);
rules backend through (T_q̂, c_i, r_i, m̂_i). Dyna-rules is thus asynchronous
RVI on the subjective model — the closest planner-free analog of known-env
PUCT, which is exactly what makes it the H5 instrument.

## 5. Learning-dynamics extensions

### 5.1 E1 — EMA / constant-gain empirical model

Core hook: `ema_gain λg` on `EmpiricalModel` multiplies every count table by
(1−λg) once per record-arrival in its cell (per-iteration decay also exposed;
per-record is the default). λg=0 recovers the paper's cumulative ratios. This
isolates the paper's fictitious-play diagnosis: constant-gain tracking changes
how fast beliefs forget, **without** changing what data exists. Prediction on
record (H7): rescues D (stale anchors go away), does not rescue C (the missing
fiat-acceptance data never arrives regardless of averaging).

### 5.2 E2 — per-agent vs population-shared tables

Core hook: `ModelBank(n_banks)` — default one shared bank (paper); extension
sets one bank per agent (memory trivial: ≤1800 × a few KB) for the empirical
*model only* — the policy/value tables (α, β, v) remain population-shared
(the ratified `MuZeroLearner` interface, plan.md Task 12). Tests whether
experience pooling in the world-model causes or merely accelerates the
observed outcomes (H8).

### 5.3 E3 — exploration sweep

Core hooks: `eps_floor` and `kappa0` flags. Grid ε ∈ {0.01, 0.03, 0.1, 0.2} ×
κ0 ∈ {0.5, 2, 8} on Economy C. If C's collapse is exploration-limited, larger ε
or weaker priors should restore the fiat links; if structural (H9), the broken
links α₂(0,·), α₃(1,2) persist because rebuilding them needs *coordinated*
two-sided experimentation each round (~ε² per link along the chain).

## 6. Experiment matrix (extensions)

- Stage-2 grid: 5 economies × {q, dyna-emp, dyna-rules} × 10 seeds = 150 runs
  (compared against the replication campaign's 100 muzero runs, core §9).
- 2³ ablation: 8 cells × {C, D} × 1 seed (+ spot re-seeds) ≈ 16–40 runs.
- E1: λg ∈ {0.005, 0.02, 0.05} × {C, D} × 5 seeds = 30 runs (muzero-emp).
- E2: per-agent × {A1, C, D} × 5 seeds = 15 runs (muzero-emp).
- E3: 4 × 3 grid on C × 3 seeds = 36 runs (muzero-emp).

All through the same campaign runner and artifact layout as replication;
reported in the same tables (T1–T4) with an explicit replication/extension
divider.
