# Findings

Replication results are recorded here as they land, one section per
hypothesis (see `plan.md` "Hypotheses" / `architecture.md`). Each entry
names the artifacts (`runs/<econ>/<config>/`) it was built from; artifacts
themselves are gitignored (`runs/`) and can be regenerated with
`scripts/compute_nash.py`.

## H1 Nash benchmarks

**H1: our RVI Nash solver (Algorithm 1, `kw.nash_solver.solve_nash`)
reproduces the paper's stationary Nash equilibria.**

Reproduce with:

```bash
for econ in A1 A2 B D; do uv run python scripts/compute_nash.py --economy $econ --prompt default --out runs/$econ/nash-default; done
uv run python scripts/compute_nash.py --economy A2 --prompt speculative --out runs/A2/nash-spec
uv run python scripts/compute_nash.py --economy B --prompt speculative --out runs/B/nash-spec
uv run python scripts/compute_nash.py --economy C --prompt money --out runs/C/nash-money
```

All seven runs converge (`e^m <= eps_br=1e-9`, `d^m <= eps_dist=1e-12`) well
inside the `max_iter=2000` budget, most in 65-72 damped best-response
iterations; `runs/B/nash-spec` converges in 1 iteration because the paper's
full-pure speculative prompt (`.superpowers/sdd/task-5-report.md` Part 2,
`prompts._SPECULATIVE_FULL_ARCS["B"]`) is already the exact fixed point.
Paper values are read from `.superpowers/sdd/paper-check-nash.md` (page
cites therein); "converged alpha retains prompt residue at
payoff-indifferent entries" (task-6-brief.md) is expected and not compared
-- only holdings (`mu`, `aggregate_holdings`) are checked against the
paper's figures.

| run | economy | prompt | converged | iters | final gain gap | final dist |
|---|---|---|---|---|---|---|
| `runs/A1/nash-default` | A1 | default | True | 68 | 7.6e-10 | 1.9e-13 |
| `runs/A2/nash-default` | A2 | default | True | 72 | 8.5e-10 | 1.4e-13 |
| `runs/A2/nash-spec`    | A2 | speculative | True | 65 | 9.0e-10 | 0.0 |
| `runs/B/nash-default`  | B  | default | True | 68 | 7.5e-10 | 3.2e-13 |
| `runs/B/nash-spec`     | B  | speculative | True | 1 | 1.8e-11 | 3.5e-13 |
| `runs/C/nash-money`    | C  | money | True | 65 | 7.1e-10 | 6.1e-13 |
| `runs/D/nash-default`  | D  | default | True | 68 | 9.7e-10 | 7.5e-13 |

### A1, default -> fundamental (paper Fig. 2, p. 44)

| type | our `mu` | paper | verdict |
|---|---|---|---|
| T1 | good2=1.00 | good2=1.00 | MATCH (exact) |
| T2 | good1=0.50, good3=0.50 | good1=0.50, good3=0.50 | MATCH (exact) |
| T3 | good1=1.00 | good1=1.00 | MATCH (exact) |

Aggregate `[0.5000, 0.3333, 0.1667]`. Verdict: **MATCH**, exact to solver
precision (paper's own numbers are already exact fractions 1/2, 1/3, 1/6).

### A2, default -> fundamental branch

Same holdings as A1-default (`T1 g2=1.00; T2 g1=0.5,g3=0.5; T3 g1=1.00`,
aggregate `[0.5, 0.3333, 0.1667]`). `paper-check-nash.md` does not print a
separate A2-fundamental holdings table (its Fig. 4 panels are not
transcribed), but this matches the paper's qualitative claim (p. 63) that
both A1 and A2 reach the fundamental commodity-money branch from the
uniform/default prompt, and the fundamental trading network does not depend
on u-bar_i. Verdict: **CONSISTENT** (no numeric Fig. 4 table available to
check against; structural claim confirmed).

### A2, speculative -> speculative branch (KW condition satisfied at ubar=500)

Our `mu`: T1 good2=0.7071, good3=0.2929; T2 good1=0.5858, good3=0.4142; T3
good1=1.00. Aggregate `[0.5286, 0.2357, 0.2357]`.
`paper-check-nash.md` does not transcribe a numeric Fig. 4-speculative
table either; it confirms only the qualitative KW-condition result (p. 21:
speculation sustainable at ubar_1=500 since 83.3 > 19) and that
`alpha[0,1,2] > 0.99` (type 1 offers good 2 for good 3), which our run
satisfies (`res.alpha[0,1,2] = 1.0`, pinned by
`test_a2_speculative_prompt_stays_speculative`). Verdict: **CONSISTENT**
(qualitative check only; no numeric table to compare).

### B, default -> fundamental (paper Fig. 6 "Fundamental RVI Nash", p. 47)

| type | our `mu` | paper | diff |
|---|---|---|---|
| T1 | good2=0.2929, good3=0.7071 | good2=0.29, good3=0.71 | <0.003 |
| T2 | good1=1.00 | good1=1.00 | exact |
| T3 | good1=0.5858, good2=0.4142 | good1=0.59, good2=0.41 | <0.003 |

Aggregate `[0.5286, 0.2357, 0.2357]` vs paper's implied aggregate good-1
share ~0.53. Verdict: **MATCH** (within the paper's 2-dp rounding; our
values are the closed forms 2-sqrt(2)~=0.5858 / sqrt(2)-1~=0.4142 that the
paper's printed 0.59/0.41 round to).

### B, speculative -> speculative (paper Fig. 6 "Speculative RVI Nash", p. 49)

| type | our `mu` | paper | diff |
|---|---|---|---|
| T1 | good2=0.5858, good3=0.4142 | good2=0.59, good3=0.41 | <0.003 |
| T2 | good1=0.7071, good3=0.2929 | good1=0.71, good3=0.29 | <0.003 |
| T3 | good2=1.00 | good2=1.00 | exact |

Aggregate `[0.2357, 0.5286, 0.2357]`. Verdict: **MATCH**, and converges in a
single damped best-response iteration -- the paper's full-pure speculative
prompt is already this fixed point. This also confirms discrepancy D1 in
`paper-check-nash.md` was fixed: the full coordinated pattern (T3 holds
good 2 exclusively, refusing good 1) is what makes it a genuine Nash
equilibrium, not the earlier partial-arc seed.

### C, money -> fiat SNE (paper Fig. 8 "RVI Nash", p. 51-53)

| type | our `mu` | paper | diff |
|---|---|---|---|
| T1 | good0=0.2632, good2=0.7368 | good0=0.26, good2=0.74 | <0.004 |
| T2 | good0=0.3140, good1=0.2618, good3=0.4242 | good0=0.31, good1=0.26, good3=0.42 | <0.004 |
| T3 | good0=0.3829, good1=0.6171 | good0=0.38, good1=0.62 | <0.004 |

Aggregate fiat share `agg[0] = 0.3200` exactly matches the paper's imposed
`M0 = 0.32` (test-pinned in `test_c_matches_fig8_fiat_holdings`, tol 1e-6).
Full aggregate `[0.3200, 0.2930, 0.2456, 0.1414]`. Verdict: **MATCH** (all
per-type holdings within the paper's 2-dp rounding).

### D, default -> commodity-money SNE (paper Fig. 10 "RVI Nash", p. 54-56)

| type | our `mu` (nonzero) | paper | diff |
|---|---|---|---|
| T1 | good2=0.1133, good3=0.8867 | good2=0.11, good3=0.89 | <0.003 |
| T2 | good1=0.7250, good3=0.0732, good4=0.2018 | good1=0.73, good4=0.20 | good1/good4 <0.005; good3=0.073 not listed by paper |
| T3 | good1=0.5655, good2=0.0406, good4=0.0239, good5=0.3700 | good1=0.57, good5=0.37 | good1/good5 <0.005; good2/good4 not listed by paper |
| T4 | good1=1.00 | good1=1.00 | exact |
| T5 | good1=0.7680, good2=0.2320 | good1=0.77, good2=0.23 | <0.002, sums to 1.00 both sides |

Aggregate good-1 share `agg[0] = 0.6117` vs paper `0.612`
(diff 3e-4, test-pinned in `test_d_matches_fig10_good1_share`, tol 5e-4).
Full aggregate `[0.6117, 0.0772, 0.1920, 0.0451, 0.0740]`. Verdict:
**MATCH** on every value the paper prints; T2 and T3's paper-listed
holdings sum to 0.93/0.94 (not 1.00), so `paper-check-nash.md`'s
transcription of Fig. 10 appears to omit a small third holding for those
two types (our solver's extra mass, 0.073 for T2 and 0.065 for T3,
plausibly fills that gap) -- a source-extraction gap, not a solver
discrepancy, since T1, T4, T5 (whose paper-listed holdings do sum to 1.00)
match closely.

### H1 verdict

**Confirmed.** All 7 branches converge to `gain_gap <= 1e-9` and
`dist <= 1e-12`; holdings match the paper's printed Figures 2, 6, 8, 10 to
within its 2-3 decimal-place rounding on every economy/prompt combination
the brief's matrix specifies (A1/A2/B/D-default, A2/B-speculative,
C-money). No numeric Fig. 4 (A2) table is available in
`paper-check-nash.md` to check A2 against, so those two rows are confirmed
qualitatively only (correct branch, correct key speculative arc). D's
per-type table has an apparent small transcription gap for T2/T3 (paper
figures do not sum to 1.00) that does not affect the pinned aggregate
good-1 share, which matches to 3 significant figures.

## H2 SCE benchmarks

**H2: misdirectional prompts drive Algorithm 2
(`kw.sce_solver.solve_sce`) to restricted-information SCE that are
subjectively optimal (`s <= 1e-9`) yet objectively poor (positive
complete-information best-response gain, `kw.diagnostics.br_gain`).**

Reproduce with:

```bash
uv run python scripts/compute_sce.py --economy A1 --prompt skeptical-autarky --out runs/A1/sce-autarky
uv run python scripts/compute_sce.py --economy C --prompt skeptical-lowtrade --out runs/C/sce-lowtrade
uv run python scripts/compute_sce.py --economy D --prompt type1-autarky --out runs/D/sce-t1autarky
```

The paper gives no per-economy prompt recipes for its SCE runs; the three
`kw.prompts.sce_prompt` entries are our reconstructions, engineered so the
solver reaches the paper's reported *endpoints* (page cites below are to
those endpoints in `.superpowers/sdd/paper-check-sce.md`, not to any prompt
construction). Paper values are read from `paper-check-sce.md`.

| run | economy | prompt | converged | iters | final s | final b | final d | br_gain max |
|---|---|---|---|---|---|---|---|---|
| `runs/A1/sce-autarky`  | A1 | skeptical-autarky  | True | 1 | 0.0 | 0.0 | 0.0 | 19.90 |
| `runs/C/sce-lowtrade`  | C  | skeptical-lowtrade | True | 1 | 0.0 | 7.7e-14 | 1.0e-15 | 33.33 |
| `runs/D/sce-t1autarky` | D  | type1-autarky      | True | 2 | 1.4e-10 | 0.0 | 7.8e-16 | 80.00 |

### A1, skeptical-autarky -> autarkic SCE (paper Fig. 2 right panel, p. 45)

| type | our `mu` | paper | verdict |
|---|---|---|---|
| T1 | good2=1.00 | good2=1.00 | MATCH (exact) |
| T2 | good3=1.00 | good3=1.00 | MATCH (exact) |
| T3 | good1=1.00 | good1=1.00 | MATCH (exact) |

Every type parks on its production good f(i) forever (test-pinned exactly
in `test_a1_autarky_sce`). The skeptical beliefs `q_i = 0` are never
contradicted because nobody offers, so no reciprocation rate is ever
observed -- the paper's "pessimistic prompts are self-fulfilling"
mechanism (p. 44-45). `br_gain = 19.9` (T2): under complete information T2
could trade its stored good 3 (cost 20/period) once for good 1 (cost
0.1/period) and park there, a deviation its own beliefs say is worthless.
Verdict: **MATCH**.

### C, skeptical-lowtrade -> low-trade fiat SCE (paper Fig. 8 bottom, p. 51-53)

| type | our `mu` | paper | diff |
|---|---|---|---|
| T1 | good3=1.000 | good3=0.999 | 0.001 |
| T2 | good0=0.960, good3=0.040 | good0=0.932 | 0.028 |
| T3 | good1=1.000 | good1=0.973 | 0.027 |

Aggregate `[0.3200, 0.3333, 0.3333, 0.0133]` vs paper's
`(0.320, 0.335, 0.333, 0.012)` -- max diff 1.7e-3 (test-pinned at 5e-3 in
`test_c_lowtrade_sce`). At this SCE trade has (essentially) stopped, so
`F_sigma` is (near-)identity and *every* distribution is invariant: the
frozen type-level split depends on the transient path taken before trade
died out, which the paper does not publish. Our forward iteration from the
f-endowment start lands within 0.03 of the paper's per-type values and
within 2e-3 on the aggregate; the hidden-acceptor / suppressed-spender
belief structure (the substance of the misdirection) is reproduced
exactly. `br_gain = 33.3` (T1): T1 could objectively sell its stored good
through the hidden acceptor arcs it believes are dead. Verdict:
**CONSISTENT** (aggregate within rounding-scale 2e-3; type-level split
transient-dependent as expected).

### D, type1-autarky -> partial-autarky SCE (paper Fig. 10 bottom, p. 54-56)

| type | our `mu` | paper | verdict |
|---|---|---|---|
| T1 | good3=1.00 | good3=1.00 | MATCH (exact) |
| T2 | good1=1.00 | good1=0.72 | policy-level gap (see below) |
| T3 | good1=1.00 | good1=1.00 | MATCH |
| T4 | good1=1.00 | good1=1.00 | MATCH |
| T5 | good1=1.00 | good1=1.00 | MATCH |

Paper type 1 (code index 0) sits in autarky on its production good 3 with
beliefs `q_1 = 0` while the other four types keep trading toward the
commodity money good 1 -- the paper's "one skeptic in a working economy"
endpoint. The T2 row is a policy-level replication gap, not a
distributional one: under our converged SCE policy, T2's holdings reach
good1 = 1.00 from *every* initial distribution -- the recurrent class is
unique -- so recurrent-class mu-selection (p. 19) cannot produce the
paper's 0.72. The paper's Fig. 10 point therefore carries a *different SCE
policy* (one that keeps T2 interior at 0.72) than the one our run
converges to (T2 fully specialized on good 1). Which SCE Algorithm 2
reaches is selection- and seed-dependent: the Delta(argmax) best response
leaves off-path and indifferent policy entries free (p. 57), and the paper
publishes no prompt recipes that would disambiguate (paper-check-sce.md
item 3). Our reconstruction reaches a valid SCE -- `s <= 1e-9`,
`b <= 1e-9`, `d <= 1e-12`, with `br_gain = 80.0` for T1 (the working
economy around the skeptic would objectively buy its good) -- that is not
Fig. 10's exact point; the test pins the honest lower bound
`mu[T2, good1] > 0.7` (`test_d_type1_autarky_sce`). Verdict: **MATCH** on
T1/T3/T4/T5 (the values Fig. 10 prints as 1.00); **VALID SCE, DIFFERENT
ENDPOINT (policy-level gap)** on T2.

### H2 verdict

**Confirmed.** All three misdirectional prompts converge to genuine SCE
(subjective optimality gap `s <= 1e-9`, on-path belief error `b <= 1e-9`,
distributional residual `d <= 1e-12`) that complete information reveals to
be far from best responses: `br_gain` of 19.9 (A1), 33.3 (C), 80.0 (D) --
each dominated by the misdirected type -- versus 0 at every H1 Nash point
(`test_nash_is_sce` pins the converse: started from the A1 Nash
equilibrium, `solve_sce` stays there with `br_gain <= 1e-6`). Subjectively
optimal, objectively poor, exactly as the paper's §7 misdirection
experiments report -- with the caveat that D's endpoint is a valid SCE but
not Fig. 10's exact point (policy-level gap on T2, above). The A2 variant
(paper fn. 10, deviation gain ~21.4) was not run -- optional in the brief
and skipped.

## H3 empirical learner

**H3: `MuZeroLearner` (Algorithms 3-4, fully empirical variant -- all three
ablation flags off) reproduces the paper's Table 2 per-economy
best-response gains at Table-1 scale, single seed, order-of-magnitude.**

Reproduce with:

```bash
for econ in A1 A2 B C D; do
  uv run python scripts/run_learning.py --economy $econ --learner muzero-emp --seed 0 \
    --out runs/$econ/muzero-emp/0
done
```

All five runs use `LearnerConfig()` Table-1 defaults (`n_agents=1800,
rounds=35, n_iters_max=180, n_sims=32, ...`), seed 0, and stop via the (D1)
rule (`simulate.run_learning`, not this script). `br_gain`/`stationarity`
are computed twice: once per iteration from the raw training-table
snapshot (`kw.diagnostics.br_gain(econ, alpha_n, beta_n, mu=None)`, i.e.
against the invariant distribution *induced by that iteration's own
policy*) for the trace, and once at the end from the extracted deployment
policy (`alpha_star`, `beta_star`, `mu_star`) -- the deployment numbers are
what the table below and architecture.md §15 compare against. Wall clock:
~55-60s per economy on a single core (`elapsed_seconds` in each
`trace.json`); the post-hoc diagnostics loop (up to 108 `invariant()` +
`br_gain()` calls) adds no measurable overhead because every training
snapshot is eps-floor-ergodic.

| econ | n_iters | converged (D1) | deployment br_gain (max over types) | paper target | verdict | deployment stationarity | paper target |
|---|---|---|---|---|---|---|---|
| A1 | 108 | True | 5.05e-11 | 1.50e-12 | MATCH | 0.0 | -- |
| A2 | 108 | True | 6.18e-12 | 21.3 | MISMATCH | 0.0163 | -- |
| B  | 108 | True | 0.0 | 0 | MATCH | 0.0106 | -- |
| C  | 108 | True | 0.594 | 12.8 | MISMATCH | 0.0223 | 0.130 |
| D  | 108 | True | 0.626 | 1.71 | MISMATCH | 0.0235 | 0.173 |

`converged` is the (D1) Δ̄-stabilization predicate, re-derived from the
saved `trace["delta"]` tail -- it is **not** a Nash/`br_gain` check. All
five runs report `converged=True` at the identical `n_iters=108`; see
"Why n_iters=108 everywhere" below before reading the per-economy verdicts,
since it is the load-bearing fact behind two of the three mismatches
(C, D; A2's mismatch is a basin-selection story, not a stopping-rule
artifact).

### A1 -- MATCH (near-exact Nash)

`br_gain = 5.05e-11` (paper `1.50e-12`, both machine-precision zero at this
scale; identical to the Task-12 slow-test value at the same seed, since
that test's `test_a1_full_run_reaches_nash` runs the identical
`(econ, cfg, seed=0)` triple). Deployment aggregate holdings
`[0.5000, 0.3333, 0.1667]` vs architecture.md §15's qualitative target
"good 1 ≈ 0.49, good 3 ≈ 0.18" -- MATCH.

Per-type training-table trajectory (`trace["agg_holdings"]`, per-type
distribution over held goods) shows the claimed autarky -> SNE transition,
but on a different clock than architecture.md §15's "iterations 1-20
nearer autarkic SCE, 21-107 nearer SNE": T2's holdings are
`[0.073, 0.062, 0.865]` at n=1 (autarky-like, concentrated on its own
production good, matching the H2 A1-autarky endpoint `T2 good3=1.00`) but
already `[0.502, 0.005, 0.493]` by n=5 -- within sampling noise of the SNE
mix `[0.5, 0, 0.5]` -- and stays there through n=108. **MISMATCH on
timing**: our run reaches the SNE-like mix by iteration ~5, not ~21.
Meanwhile the *training-table* `br_gain_series` drops 17.4 (n=1) -> 6.5
(n=5) -> 1.90 (n=10) and then sits at exactly 1.9025 for n=10..108 without
further movement -- the training tables (eps-floor-exploratory,
zeta-mixed) reach their own local fixed point well before n=108 and stop
moving; only the separate, near-greedy `deployment_policy()` pass (adj 8:
deeper horizon, more sims, unclipped) closes the remaining gap to
`5e-11`. This decoupling -- training-table convergence != deployment
convergence != true Nash convergence -- recurs in C and D below.

### A2 -- MISMATCH (learner found the speculative Nash; paper's Table 2 apparently did not)

`br_gain = 6.18e-12` (paper target `21.3`) and deployment aggregate
holdings `[0.5286, 0.2357, 0.2357]` -- this is an **exact match, to 4
significant figures, of H1's own `runs/A2/nash-spec` speculative-branch
Nash point** (`findings.md` H1 table: aggregate `[0.5286, 0.2357,
0.2357]`, T1 `good2=0.7071/good3=0.2929`, etc.), not the fundamental
branch A1 also reaches. In other words: starting from the same symmetric
uniform initial policy used for every other economy (no misdirectional
prompt, no speculative seeding), self-play at seed 0 discovers the
coordination equilibrium that requires T1 to speculatively accept good 2
for good 3 (`alpha[0,1,2] -> 1`) -- the branch architecture.md §15
describes as unlikely to be found ("A2 condition flips between ū=100 and
ū=500 (s₃−s₂=19 vs ū/6∈{16.7,83.3})"; our A2 is instantiated at
`ubar=500.0` per `economies.py`, the side of the condition where
speculation is the *rewarding* deviation, per H1's own A2-speculative run).

The paper's Table 2 `br_gain≈21.3` for A2 is presumably what a self-play
run that gets stuck on the *fundamental* branch (A1-like holdings) would
score when evaluated at ubar=500, since the fundamental branch is not
best-response once speculation pays 500-scale utility. Our seed-0 run
simply lands in the better basin. This is a real, well-characterized
mismatch, not a red flag: architecture.md §15's own closing line --
"Multi-seed runs test the *claims*, not these exact single-seed numbers"
-- anticipates exactly this kind of single-seed variance in a
coordination-selection problem. §12 items 1 (argmax ties -> lowest action
index) and 7 (multichain fix F_σ⋆ power-iterates from the terminal
population distribution) both bear on which basin a run falls into; no
tuning was applied to chase either outcome.

### B -- MATCH (exact)

`br_gain = 0.0` exactly (paper target `0`), aggregate holdings
`[0.5286, 0.2357, 0.2357]` matching H1's `runs/B/nash-spec` speculative
branch. B has no fundamental/speculative "condition flip" ambiguity
(H1: both `nash-default` and `nash-spec` are genuine Nash points for B),
so there is no analogous basin-selection risk here -- straightforward
replication.

### C -- MISMATCH (br_gain ~21x and stationarity ~6x lower than paper)

`br_gain = 0.594` (paper `12.8`), `stationarity = 0.0223` (paper `0.130`).
Deployment aggregate `[0.3200, 0.3192, 0.1805, 0.1803]` vs the H1 C-money
SNE aggregate `[0.320, 0.2930, 0.2456, 0.1414]` -- fiat matches the
conserved `M0=0.32` exactly; the deployment run sits *above* the true
Nash point on good 2 (0.3192 vs 0.2930) and good 4 (0.1803 vs 0.1414) and
short only on good 3 (0.1805 vs 0.2456), consistent with the nonzero
`br_gain`. Same
training/deployment decoupling as A1: the training-table `br_gain_series`
sits at ~2.16-2.19 for the last 10 iterations (an order of magnitude
*above* the deployment number), so the deployment PUCT pass (adj 8) is
doing real work here but does not fully close the gap within its fixed
budget (`h_dep=14, n_sims_dep=96, tau_dep=0.35`) the way it does for A1.

**Fiat link, deployment (unclipped) policy -- NOT the paper's
broken-link pathology.** architecture.md §15 names three cells for C's
fiat-SNE qualitative signature (paper 1-indexed types): `α₂(0,2)=0.001,
α₂(2,0)=1.000, α₃(1,2)≈0`. Reading these directly as `alpha[i-1,k,j]`
under the codebase's own `alpha[i,k,j]` = "type i holds k, meets
counterparty holding j" convention (confirmed at `simulate.py:131-132`
and architecture.md's `F_i(k,j)` formula, §6) and fixing the direction of
"healthy" using `economies.py`'s `consume_idx=[1,2,3]` for C together
with `prompts.py:97`'s `alpha[i, :, consume_idx[i]] = 1.0` prior (which
establishes `j = consume_idx[i]` as "trade k away for my consumption
good" -- i.e. `alpha[1,0,2]` is T2 buying its own consumption good, good
2, with fiat: the textbook healthy monetary-trade direction) gives:

| cell | code array | our value | paper | reading |
|---|---|---|---|---|
| α₂(0,2) | `alpha[1,0,2]` | 0.9993 | 0.001 | MISMATCH -- ours healthy, paper broken |
| α₂(2,0) | `alpha[1,2,0]` | 0.00498 | 1.000 | MISMATCH -- ours healthy, paper broken |
| α₃(1,2) | `alpha[2,1,2]` | 2.24e-6 | ~0 | MATCH |

Our T2 (deployment) eagerly buys its consumption good with fiat
(`alpha[1,0,2]≈1`) and refuses to give up a consumption good it already
holds for fiat (`alpha[1,2,0]≈0`) -- the rational, healthy fiat-monetary
policy, and the mirror image of the paper's stated `(0.001, 1.000)` pair,
which describes a T2 that refuses to buy what it needs with money and
immediately dumps what it needs for money. This is corroborated, not just
asserted, by the aggregate diagnostics: a genuinely broken fiat link is a
first-order welfare loss and should force a *large* `br_gain` (which is
exactly why the paper's own C reports `br_gain=12.8`); our C
`br_gain=0.594` is inconsistent with a broken link by that same logic.
**Conclusion: C did not reproduce the paper's broken-fiat-link
pathology -- it learned the healthy fiat-consumption policy instead,
which is the same phenomenon, not a separate one, as its
lower-than-paper `br_gain`.** (An earlier "argument-order swap" reading
considered during analysis -- treating `α₂(0,2)` as `alpha[1,2,0]` and
vice versa -- is discarded here: it would require `alpha[1,0,2]` to be
simultaneously near-broken, `≈0.001`, and the run to be near-Nash,
`br_gain=0.594`, which is self-contradictory by the welfare-loss argument
above.) **Training-table (eps-clipped) values for the same three cells**
are `alpha_train_final[1,0,2]=0.446`, `[1,2,0]=0.927`, `[2,1,2]=0.107` --
compressed toward `[eps_floor, 1-eps_floor] = [0.03, 0.97]` as expected
(adj 1/5) and, notably, `[1,2,0]=0.927` retains far more of the
"wrong-direction" (dump-consumption-good-for-fiat) mass than the
deployment policy shows -- the deployment PUCT pass (adj 8) is cleaning
up training-table noise in the healthy direction, not degrading a
paper-matching training table into a healthier deployment one.

### D -- MISMATCH (br_gain ~2.7x low, stationarity ~7x low)

`br_gain = 0.626` (paper `1.71`) driven almost entirely by T2
(`br_gaps_deployment = [0.0606, 0.6260, 5.3e-15, 0, 0]`); `stationarity =
0.0235` (paper `0.173`). Deployment aggregate `[0.4804, 0.1540, 0.1663,
0.1085, 0.0908]` vs the H1 D-default SNE aggregate `[0.612, 0.077, 0.192,
0.045, 0.074]` -- good-1 share short by ~0.13, again consistent with a
run that has not fully closed on the true Nash point. Unlike C, D's
*training-table* `br_gain_series` (last 10 iterations: 3.53-3.54) also sat
roughly 5-6x above the deployment number and, distinctively, D's `delta`
tail is not monotone the way A1/A2/B/C's are -- it drifts up slightly from
`0.00049` (n=100) to `0.00066-0.00068` (n=104-108) after bottoming out
around n=100-102, still comfortably under `delta_stop=0.003` so (D1) still
fires at n=108, but indicating D's larger action space (5 types x 5 goods
vs A1-C's 3-4) has not fully settled by the shared stopping point. Both
diagnostics move the *same direction* relative to the paper (our run is
closer to a fixed point than the paper reports, not farther), mirroring
C's pattern rather than contradicting it.

### Why n_iters=108 everywhere

All five economies -- three of which have very different converged
`br_gain` outcomes (near-zero for A1/A2/B, 0.59-0.63 for C/D) -- stop at
the identical `n_iters=108` under (D1). This is explained by the ζ_n
schedule, not coincidence: `zeta_n = max(zeta_min=0.01, zeta0=0.45 *
zeta_decay=0.94^(n-1))` hits its floor `zeta_min` at n≈63 for *every*
economy (the schedule has no economy-dependence), after which the
per-iteration policy mixing step size is constant. From n≈63 onward, Δ̄_n
(§12 item 9: computed post-clip on stored tables) reflects how much the
*already-near-fixed-point* mixed table still moves at a constant mixing
rate, which for all five economies apparently drops below `delta_stop=
0.003` for a `patience=8` streak at almost the same iteration count. The
practical upshot: `converged=True` at a shared `n_iters` certifies that
the *training-table update dynamics* (ζ-mixing) have stabilized, not that
the trained policy is near a stage-game Nash -- A1/B's training tables
happen to stabilize near the true Nash, A2 got lucky into the speculative
basin, but C/D's training tables stabilize at a point still 0.6-2.2 away
from Nash by the `br_gain` measure, and (D1) does not distinguish these
cases. This is a property of the paper's own stopping rule as
implemented (§12 item 9), not a script defect; flagged here because it is
the load-bearing mechanism behind every mismatch above except A2's.

### H3 verdict

**Partial match, three single-seed mismatches recorded, no tuning
applied -- and all three point the same direction.** A1 and B reproduce
paper Table 2 to the stated order of magnitude (both effectively exact
zero). The three mismatches are not four unrelated discrepancies: A2, C,
and D all show our single seed converging *closer to a true stage-game
Nash* than the paper's own reported Table 2 run. A2's `br_gain≈0` vs the
paper's `≈21.3` is our seed landing on the harder-to-find speculative
Nash instead of the fundamental basin -- consistent with architecture.md
§15's own caveat that single-seed numbers test seed variance, not the
underlying claim; §12 items 1 and 7 are the plausible basin-selection
levers, left untouched. C and D show `br_gain` 0.59 vs 12.8 and 0.63 vs
1.71 (`stationarity` 0.022 vs 0.130, 0.024 vs 0.173) -- both economies'
training tables reach a ζ-mixing fixed point closer to, but still short
of, the true Nash by n=108 on this seed; the deployment-time PUCT
refinement pass (adj 8) narrows the gap further. C's specific fiat-link
cells make this concrete rather than just aggregate: our run learned the
*healthy* fiat-consumption policy (`alpha[1,0,2]≈1`, `alpha[1,2,0]≈0`)
where the paper's Table 2 run reports the *broken* mirror-image policy
(`α₂(0,2)=0.001, α₂(2,0)=1.000`) -- we did not reproduce that pathology,
and the much-lower `br_gain` is the same phenomenon, not a separate
discrepancy (a broken fiat link is a first-order welfare loss and would
force a large `br_gain`, which is exactly why the paper's own C shows
12.8). §12 item 9 (Δ̄ defined on post-clip stored tables, hence blind to
the true residual against a stage-game Nash) is the primary suspect for
why (D1) still certifies convergence before C/D fully close their
remaining gap, with items 3 (memoization) and 11 (deployment exploration
floor) as secondary candidates for why the deployment pass narrows but
does not eliminate that gap. No hyperparameter was changed and no library
code was edited to chase any of these three mismatches; multi-seed
campaigns (Task 17) are the right next step to determine whether the
paper's single-seed Table 2 runs (not ours) are the outliers here, and
whether that same over-performance shows up consistently across seeds or
is itself a lucky draw.

## H4 known-environment

**H4: `MuZeroLearner` known-environment variant (`muzero-known` -- all three
ablation flags on: `known_rules`, `greedy_backup`, `centered_values`; paper
§6.4) reaches best-response gain ≈ 0 in all five economies (paper Table 2,
H4 row). It plans against the *true* environment transition/reward rules
(`kw.planner.RulesBackend`) while learning only the meeting frequency m̂ and
the offer-acceptance rate q̂ from play, so the model error that keeps the
fully empirical H3 learner short of Nash is removed.**

Reproduce with:

```bash
for econ in A1 A2 B C D; do
  uv run python scripts/run_learning.py --economy $econ --learner muzero-known --seed 0 \
    --out runs/$econ/muzero-known/0
done
```

All five runs use `LearnerConfig()` Table-1 defaults (`n_agents=1800,
rounds=35, n_iters_max=180, n_sims=32, ...`), seed 0, the `muzero-known`
preset (`known_rules=greedy_backup=centered_values=True`), and stop via the
(D1) rule. The known-environment wiring (`muzero.py`) plugs the learned
(m̂, q̂) into `RulesBackend` for *both* the training and deployment planners
and additionally feeds `observe_offer` opposite-direction acceptance
evidence so q̂ can move off its 1/2 prior (fidelity 12.10; the empirical
variant never does this). `br_gain`/`stationarity` are the same two
deployment diagnostics H3 reports: `br_gain` at `mu_star` (the invariant of
the deployed policy) and `stationarity` at the finite-population empirical
terminal holdings `mu_emp = res.trace["agg_holdings"][-1]` (§12.7). Wall
clock ~59-61s per economy.

| econ | n_iters | converged (D1) | deployment br_gain (max over types) | paper target | verdict | deployment stationarity (mu_emp) | paper target |
|---|---|---|---|---|---|---|---|
| A1 | 108 | True | 0.0 | 0 | MATCH | 1.11e-16 | -- |
| A2 | 108 | True | 0.0 | 0 | MATCH | 0.0278 | -- |
| B  | 108 | True | 0.0 | 0 | MATCH | 0.0233 | -- |
| C  | 108 | True | 0.0 | 0 | MATCH | 0.0156 | 1.74e-8 |
| D  | 108 | True | 0.0202 | 0 | MATCH (residual 0.02) | 0.0125 | -- |

**`br_gain ≈ 0` everywhere is the H4 hypothesis, and it is met.** All five
economies reach a complete-information best response to their own induced
distribution -- the exact `≈ 0` that paper Table 2's H4 row reports and that
the fully empirical H3 learner misses on A2/C/D. Every run stops at the same
`n_iters=108` under (D1) for the same ζ-schedule reason documented in H3's
"Why n_iters=108 everywhere" (the stopping rule is policy-independent). The
per-economy notes below read the deployment aggregate against H1's exact
Nash points; the stationarity column is discussed once, separately, because
it is a finite-N artifact and not the H4 claim.

### A1 / A2 -- MATCH (fundamental Nash, gain 0)

Both reach `br_gain = 0` at aggregate `[0.5000, 0.3333, 0.1667]`, exactly
H1's A1/A2-default *fundamental* commodity-money Nash (`findings.md` H1:
`[0.5, 0.3333, 0.1667]`, converged there to `gain_gap 8.5e-10`). Note this
is the opposite basin from H3-empirical A2, whose seed-0 run landed on the
*speculative* branch `[0.5286, 0.2357, 0.2357]`; both branches are
legitimate A2 equilibria under the ubar=500 KW condition (task-13-report.md,
H1 A2 rows), and the known-environment planner at this seed selects the
fundamental one. Either way `br_gain = 0`, matching paper H4. (Paper Fig. 12,
p. 60 notes A2's known-environment policy keeps 0.5 mass at indifferent
entries -- payoff-irrelevant, consistent with a gain-0 equilibrium.)

### B -- MATCH (gain 0)

`br_gain = 0` at aggregate `[0.5000, 0.1667, 0.3333]`. This is a genuine
complete-information Nash (gain 0 certifies it), distinct from both H1 B
branches (`nash-default [0.5286, 0.2357, 0.2357]`, `nash-spec [0.2357,
0.5286, 0.2357]`): self-play from the uniform prior under the true rules
selects its own basin. No fundamental/speculative ambiguity affects the
gain-0 verdict.

### C -- MATCH, and reaches the true fiat SNE H3 missed

`br_gain = 0` at aggregate `[0.3200, 0.2930, 0.2456, 0.1414]` -- this is the
H1 C-money fiat SNE aggregate *to four significant figures* (`findings.md`
H1: `[0.3200, 0.2930, 0.2456, 0.1414]`, fiat pinned at `M0 = 0.32`). The
contrast with H3 is the substance of H4: the fully empirical C learner
stalled at `br_gain = 0.594` with the deployment aggregate `[0.3200,
0.3192, 0.1805, 0.1803]` sitting *off* the true Nash on goods 2-4, and
learned the *healthy* (not the paper's broken) fiat link. Handing the same
learner the true trade/consumption rules and only leaving m̂/q̂ to be learned
closes that gap to exactly zero and lands the population on the H1 Nash point.

### D -- MATCH (small residual 0.02)

`br_gain = 0.0202` (per-type `[0, 0.0149, 0.0202, 0, 0]`, driven by types 2-3)
at aggregate `[0.6130, 0.0764, 0.1902, 0.0473, 0.0732]` -- essentially the
H1 D-default commodity-money SNE aggregate `[0.6117, 0.0772, 0.1920, 0.0451,
0.0740]` (good-1 share 0.613 vs H1's 0.612). The residual `0.0202` is a
small deviation from paper H4's exact `0`, recorded per the mismatch
protocol rather than tuned away; at Table-1 scale it is ≈1% of D's own
H3-empirical gain (0.626) and two orders below D's SCE misdirection gain
(80.0), i.e. Nash to plotting precision. D's larger action space (5 types ×
5 goods) is the same reason its H3 delta tail was the least settled at
n=108; the tiny leftover gain here is consistent with that, not with a wiring
fault (br_gain is 0 to machine precision on A1/A2/B/C, which share the code
path).

### Stationarity is a finite-N floor here, not the paper's continuum figure

The deployment stationarity column (0.012-0.028 on A2/B/C/D) is **not**
comparable to the paper's `1.74e-8` (C, cited in the Task-14 brief), and the
gap is a sampling artifact, not an equilibrium failure. Three cross-checks
establish this:

- **A1 is the internal control.** A1's equilibrium has degenerate (pure)
  holdings, so its empirical terminal distribution carries no mixing noise
  and stationarity is `1.11e-16` -- machine zero. The other four economies
  have mixed-holding equilibria, and their `0.012-0.028` residuals are the
  `O(1/sqrt(n_agents))` histogram noise of `mu_emp`: `1/sqrt(1800) ≈ 0.024`,
  squarely the observed band. Same finite-N floor, same root cause as the
  H3 stationarity values (`0.011-0.024`), which this section's convention
  mirrors (both read `stationarity` at `mu_emp`, per `run_learning.py`).
- **`br_gain = 0` already certifies the equilibrium.** A best response to the
  deployed policy exists at gain 0; the population is at a Nash point. The
  residual is only the finite-sample deviation of the *measured* histogram
  from the deterministic invariant.
- **The paper's `1.74e-8` is the continuum-invariant residual, which does not
  discriminate an equilibrium.** Evaluating `stationarity` at the computed
  invariant `mu_star` instead of `mu_emp` gives `~1e-13` here -- but for a
  *non*-Nash policy too: D's `mu_star` residual is `7.9e-13` despite
  `br_gain = 0.0202`, because `mu_star = invariant(alpha_star, beta_star)`
  makes `one_step(mu_star) ≈ mu_star` by construction (it measures only the
  `invariant()` solver tolerance for *any* policy). It is therefore not used
  as a Nash check and not substituted into the table to manufacture a match
  with `1.74e-8`; the honest finite-N number is reported instead. This gap
  requires a much larger (continuum) population to close and is the expected
  behavior at `n_agents=1800` (architecture.md §12).

### H4 verdict

**Confirmed.** The known-environment variant reaches `br_gain ≈ 0` in all
five economies -- the paper Table 2 H4 claim -- with D carrying a small
`0.0202` residual (recorded, not tuned). The positive H4 signal is sharper
than the bare gains: C lands on the H1 fiat SNE it missed as a fully
empirical learner, D on the H1 commodity-money SNE, and A1/A2/B on
legitimate fundamental/self-play Nash points, so plugging the true rules into
the same PUCT learner (leaving only m̂/q̂ to be estimated) removes exactly the
H3 shortfall. The deployment-stationarity column (0.012-0.028) is a finite-N
`1/sqrt(n_agents)` floor, not the paper's continuum `1.74e-8`; the A1 pure-
holdings control (`1.11e-16`) and the shared root cause with H3's
stationarity values confirm this is a sampling artifact of `mu_emp`, and the
vacuous `mu_star` residual (`~1e-13` even for non-Nash D) was deliberately
*not* substituted to force a match. No hyperparameter was changed and no
library code edited for any of these outcomes.

## Stage-2 grid (H5, H6)

**H5 (attribution, ext §2): the known-environment rescue seen in H4 comes
mainly from model *contents* (the known one-step rules), not planner
sophistication — Dyna-rules should behave like known-env PUCT
(`muzero-known`) and Dyna-emp like empirical PUCT (`muzero-emp`). H6: fully
model-free differential Q-learning (realized payoffs only, the EWA δ≈0
pole) is at least as prone to SCE traps as the empirical learner.** The
instruments are the three Stage-2 learners (`q`, `dyna-emp`, `dyna-rules`;
Tasks 15–16, ext §4), which share one tabular Q-learning core and differ
only in planning backend — none, learned empirical model, known rules —
compared against the two MuZero variants' existing H3/H4 artifacts (the
`muzero-emp`/`muzero-known` columns are read from `runs/`, not re-run).

Reproduce with:

```bash
for econ in A1 A2 B C D; do for lrn in q dyna-emp dyna-rules; do
  uv run python scripts/run_learning.py --economy $econ --learner $lrn --seed 0 \
    --out runs/$econ/$lrn/0
done; done
```

All 15 runs use `LearnerConfig()` Table-1 defaults (`n_agents=1800,
rounds=35, n_iters_max=180, ...`), seed 0, learner defaults (`q_lr=0.1`,
`eta=0.5`, `dyna_k=20`, `kappa0=2.0` count priors). Deployment `br_gain` is
`max` over the per-type gaps at `mu_star`, exactly as in H3/H4. Two caveats
up front, disclosed rather than smoothed over. (1) **No Stage-2 cell
satisfies the (D1) stopping rule**: every run hits the `n_iters_max=180`
cap with `converged=False` (the `QLearner`'s hard 0/1 greedy policies keep
the (D1) `delta` metric high even when the policy is locked in), whereas
every MuZero run stops at `n_iters=108` with `converged=True`. Capped-cell
gains are therefore "at 180 iterations", not "at convergence"; the
extended-horizon check below tests whether the cap binds. (2) The
`A1/dyna-emp` cell ran on the pre-cache code (commit `6bb1427`, elapsed
6783 s); all other dyna cells ran after the kernel-cache commit
(`cb9cf29`, ~2400–3750 s). The cache was proven byte-identical
(`np.array_equal` on all 20 output arrays over 4 econ×backend configs)
before being committed, so the artifacts are interchangeable. Additionally,
`C/q`'s per-iteration `br_gain_series` carries 5 `nan` snapshots out of 180
(RVI non-convergence on hard-greedy early snapshots, recorded as `nan` per
`run_learning.py`'s tolerant-diagnostics policy); all deployment-policy
numbers below are finite.

Deployment `br_gain` (max over types), seed 0:

| econ | q | dyna-emp | dyna-rules | muzero-emp | muzero-known |
|---|---|---|---|---|---|
| A1 | 5.04912e-11 | 5.04912e-11 | 5.04912e-11 | 5.04841e-11 | 0.0 |
| A2 | 1.13687e-13 | 6.16751e-12 | 34.1605 | 6.18172e-12 | 0.0 |
| B  | 7.22222 | 2.88889 | 0.0 | 0.0 | 0.0 |
| C  | 0.517958 | 0.0 | 0.517958 | 0.59416 | 0.0 |
| D  | 3.67678 | 0.862965 | 0.092762 | 0.626035 | 0.0201752 |

### A1 — all five learners at the fundamental Nash

Every learner, including plain `q`, lands on H1's fundamental
commodity-money aggregate `[0.5, 0.3333, 0.1667]` with gain ≤ 5.05e-11.
A1's equilibrium is reachable by pure model-free TD; planning adds nothing
because there is nothing left to fix. Uninformative for H5, consistent with
H6 (no trap to fall into).

### A2 — `q` and `dyna-emp` at gain ≈ 0; `dyna-rules` destabilizes late

`dyna-emp` reproduces `muzero-emp` almost exactly — same speculative-branch
aggregate `[0.5286, 0.2357, 0.2357]`, gains 6.16751e-12 vs 6.18172e-12 —
the cleanest H5 pairing in the grid. Plain `q` reaches gain 1.13687e-13 at
aggregate `[0.1667, 0.5, 0.3333]`, a *third* basin distinct from both H1
branches (a mirrored arrangement with good 2 as the medium; recorded as
found, not catalogued in H1). The anomaly is `dyna-rules`: final gain
34.1605 (all on type 0) at the speculative aggregate — of the same order
as the paper's own A2 empirical-learner gain ≈ 21 — and its training series
shows it sat at gain 6.16751e-12 as late as iteration 129 before *leaving*:
a late destabilization, not a failure to find the equilibrium. This cell
violates the H5 prediction (`dyna-rules` ≠ `muzero-known`'s 0.0) and is
dissected in the extended-horizon check below.

### B — rules model rescues; empirical Dyna and `q` trap

`dyna-rules` reaches gain exactly 0.0 at `[0.5286, 0.2357, 0.2357]` (the
same aggregate `muzero-emp` converged to), matching `muzero-known`'s ≈ 0 as
H5 predicts. `q` traps hard at 7.22222 (type 0) on `[0.5, 0.1667, 0.3333]`
and `dyna-emp` at 2.88889 (per-type `[-8.3e-12, 2.889, 2.0]`) on
`[0.3333, 0.5, 0.1667]` — worse than `muzero-emp`'s 0.0, so the
`dyna-emp ≈ muzero-emp` half of H5 fails on B, while the ordering
q > dyna-emp > dyna-rules ≈ 0 is exactly the model-contents gradient H5
posits and `q`'s being worst is consistent with H6.

### C — `dyna-emp` finds the fiat SNE that `muzero-emp` missed

All three Stage-2 learners land on the fiat-SNE aggregate
`[0.32, 0.293, 0.2456, 0.1414]` (H1's C equilibrium, which `muzero-known`
also reaches at gain 0). `dyna-emp` attains gain exactly 0.0 there —
*better* than `muzero-emp`, which stopped at 0.59416 on a different
aggregate (H3's documented C collapse). `q` and `dyna-rules` sit at an
identical 0.517958 (all on the per-type index-1 entry) on the correct
aggregate: right distribution, one type short of best-responding. C thus
breaks the H5 pairing in the *favorable* direction for Dyna-emp (planning
sweeps over a learned model beat PUCT with the same model class here), and
`q`'s 0.518 < `muzero-emp`'s 0.594 makes C the one economy where H6's
"at least as prone" reads as a tie rather than a Q-learning deficit.

### D — clean H5 gradient

Every learner lands on a different aggregate, but the gains order exactly
as H5 predicts: rules-model variants an order of magnitude below
empirical-model variants (`dyna-rules` 0.092762 ~ `muzero-known` 0.0201752,
both small; `dyna-emp` 0.862965 ~ `muzero-emp` 0.626035, both order-1), and
model-free `q` worst at 3.67678 (H6). D's deployment stationarity for the
two dyna cells (0.129, 0.226) is well above the ~0.024 finite-N floor seen
elsewhere in the grid (all other Stage-2 cells: 0.010–0.143) — their
deployed policies' invariants disagree with the terminal empirical
holdings, another symptom of the not-converged-at-cap caveat.

### H5/H6 verdict

**H5: partially confirmed — the model-contents gradient holds on B and D
only, and is violated outright on A2 and on C's emp/rules pair.** On B and
D gains order as q ≥ dyna-emp ≥ dyna-rules (B: 7.22 ≥ 2.89 ≥ 0.0; D:
3.68 ≥ 0.863 ≥ 0.0928), i.e. injecting the known rules into a *dumb*
planner recovers most of what `muzero-known` gets from them — the
attribution claim, where it holds. A1 is uninformative (all five learners
at machine-zero gain). Elsewhere the gradient itself is violated, not
merely the muzero pairings: A2 — the grid's largest spread — orders the
*opposite* way, with `dyna-rules` posting the grid's worst gain (34.16)
while `q` sits at 1.14e-13; and C's emp/rules pair is a strict reversal,
not a tie (`dyna-emp` 0.0 vs `dyna-rules` 0.518). The strict muzero
pairings additionally fail in three places: `dyna-rules` destabilizes late
on A2 (34.16 vs `muzero-known` 0), `dyna-emp` under-performs `muzero-emp`
on B (2.89 vs 0), and over-performs it on C (0 vs 0.594). **H6:
consistent** —
`q` is worst or tied-worst in four of five economies with severe traps on
B (7.22) and D (3.68), never beating the empirical PUCT learner beyond
noise except C's 0.518 vs 0.594. All Stage-2 numbers carry the "capped at
180 iterations, (D1) unmet" caveat; the extended-horizon check below tests
which capped cells are sample-starved and which are genuinely settled.

### Extended-horizon check (sample-starvation hypothesis)

**Hypothesis under test: the trapped Stage-2 cells (final deployment
`br_gain > 1e-6` at the 180-iteration cap) are merely sample-starved — more
iterations would close the gap — rather than settled in a genuine trap.**
Three probes, all anchored to the seed-0 artifacts: (1) a nan-aware tail
analysis of each cell's per-iteration `br_gain_series` (final finite value
vs the finite value ~50 iterations earlier); (2) for the dyna cells, the
`diagnostics.sce_certificate` of the final learned beliefs (m̂, q̂) against
the deployment policy — subjective gap `s ≈ 0` with objective
`br_gain > 0` is the self-confirming-trap signature; (3) extended re-runs
*only* for cells whose tail was materially declining (dyna at cap 360, `q`
at cap 720), written to separate out dirs so the primary grid artifacts are
untouched. Eight cells qualify.

Tail analysis (artifact-exact):

| cell | gain @ iter 129 | gain @ iter 179 | tail-50 min/max | verdict |
|---|---|---|---|---|
| B/q | 7.22222 | 7.22222 | 0 / 33.3333 | plateaued (oscillating) |
| C/q | 7.32918 | 0.517958 | 0 / 11.2709 | declining |
| D/q | 6.95455 | 3.67678 | 1.17953 / 72 | declining |
| B/dyna-emp | 2.88889 | 2.88889 | 0 / 4.25463 | plateaued (oscillating) |
| D/dyna-emp | 10.446 | 0.862965 | 1.49e-13 / 17.3245 | declining |
| A2/dyna-rules | 6.16751e-12 | 34.1605 | 6.17e-12 / 34.1605 | rising |
| C/dyna-rules | 0.0 | 0.517958 | 0 / 0.517958 | rising |
| D/dyna-rules | 9.75025 | 0.092762 | 0.092762 / 20.9886 | declining |

The two *rising* cells deserve emphasis: A2/dyna-rules sat at machine-zero
gain through iteration ~129 and then *left* — more data made it worse, the
opposite signature of sample starvation (and an echo of the paper's own A2
instability, gain ≈ 21). Rising and plateaued cells were not extended per
the declining-only rule; they get the certificate diagnostic instead.

**SCE certificates.** The run artifacts do not persist the learner's belief
state, so each qualifying dyna cell was replayed in-process
(`QLearner(econ, LearnerConfig(), np.random.default_rng(0), dyna=...)` +
`kw.simulate.run_learning(..., seed=0)`; the CLI's diagnostics all run
*after* the simulation, so the replay is trajectory-identical), and
faithfulness was proven per cell by `np.array_equal` on
`alpha_star`/`beta_star`/`mu_star` against the saved `arrays.npz` — all
five cells byte-identical. `sce_certificate` inputs: `m̂ = model.m_hat()`;
for rules cells `q̂ = model.q_hat()` (the pooled acceptance belief the
`RulesBackend` actually plans with); emp cells never feed `observe_offer`,
so their pooled `q̂` sits at the uninformative 0.5 prior — both that
literal prior and the *implied* acceptance belief embedded in the learned
transition kernel, `q̃[i,k,j] = p_trade[i,k,j,λ=1,h′=j]`, are reported.
`mu = mu_star` from the artifacts. The five raw certificate JSONs
(including each cell's `replay_matches_artifacts` byte-identity proof) are
archived unmodified under `docs/certs/`.

| cell | objective br_gain | subjective gap s | on-path belief error b |
|---|---|---|---|
| A2/dyna-rules | 34.1605 | 1.10e-11 | 0.5028 |
| C/dyna-rules | 0.517958 | 0.0 | 0.2156 |
| D/dyna-rules | 0.092762 | 1.14e-11 | 0.2444 |
| D/dyna-emp | 0.862965 | 1.16e-11 (implied q̃; 0.941 under the literal 0.5 prior) | 0.4475 |
| B/dyna-emp | 2.88889 | 1.856 (implied q̃; 2.042 prior) | 0.537 |

Four of the five trapped dyna cells are *subjectively optimal* — `s ≈ 0`
to eleven digits — against their own learned beliefs while objectively
suboptimal by 0.09–34: the approximate self-confirming-trap signature.
This is a finding, not a bug: planning amplifies belief errors into
confidently-held wrong policies. The large on-path belief error `b`
(0.22–0.50, far above `classify_run`'s strict 0.02 SCE threshold) locates
the mechanism: `q̂` aggregates acceptance behavior over the entire
ε-greedy *training* history, so it is consistent with the data the learner
generated but wrong under the deployed policy — beliefs confirmed by
history rather than by the equilibrium path (so formally these classify as
UNSETTLED, not certified SCE_TRAP). The exception, B/dyna-emp
(`s ≈ 1.9` under both belief readings), is *not* belief-certified: its
Q-table never reached its own model's optimum, matching its oscillating
tail — an unsettled-optimization failure, not a belief trap. Caveat for
the emp cells: the certificate's subjective MDP is parameterized by
(m̂, q̃) under the true mechanics, whereas the emp planner backs up through
the raw learned `p_trade` (including its smoothing mass on physically
impossible transitions), so `s` for emp cells is measured against the
closest (m, q)-factored reading of its model, not its literal internals.

**Extended runs** (`--n-iters-max` override, new orthogonal CLI flag;
distinct out dirs; determinism verified — each extended run's
`br_gain_series` first 180 entries are exactly equal to its primary run's,
as expected since `n_iters_max` appears only as the loop bound):

```bash
uv run python scripts/run_learning.py --economy C --learner q --seed 0 \
  --n-iters-max 720 --out runs/C/q/0-ext720          # and likewise D/q
uv run python scripts/run_learning.py --economy D --learner dyna-emp --seed 0 \
  --n-iters-max 360 --out runs/D/dyna-emp/0-ext360   # and likewise dyna-rules
```

| cell | cap | gain @ 180 (primary) | gain @ extended cap | converged (D1) | sample-starved? |
|---|---|---|---|---|---|
| C/q | 720 | 0.517958 | 0.517958 | False | no — identical value; the late-window decline bottomed out at the same trap |
| D/q | 720 | 3.67678 | 3.05057 | False | largely no — 17% lower after 4× horizon, still order-1, tail wobbling (2.877 at iter 670) |
| D/dyna-emp | 360 | 0.862965 | 0.271048 | False | partially at best — endpoint 69% lower, but the series hit 8.89757 at iter 310: swinging over an order of magnitude within the last 50 iterations, oscillating rather than converging |
| D/dyna-rules | 360 | 0.092762 | 0.843137 | False | no — 9× *worse*; 0.233626 at iter 310 and rising, with the gap moving to different types (type 4: 0.843, type 1: 0.599) and a different aggregate |

### Extended-horizon verdict

**Sample starvation is rejected as the explanation for the trapped
cells.** No extended run met (D1) even at 2–4× the horizon, and no cell
closed its gap: C/q reproduced its 0.517958 trap to six digits at 720
iterations; D/q shaved 17% and stalled at order-1; D/dyna-emp's endpoint
fell to 0.271048 but its tail still swings between 0.27 and 8.9;
D/dyna-rules got 9× worse. What the three probes show instead is three
distinct failure modes. (i) *Belief-anchored self-confirming traps*: four
of five trapped dyna cells (A2/dyna-rules, C/dyna-rules, D/dyna-rules,
D/dyna-emp) deploy policies that are exactly optimal (`s ≈ 1e-11`) against
learned beliefs whose on-path acceptance error is 0.22–0.50 — more
same-policy data cannot fix a gap the learner cannot subjectively see; this
is the headline mechanism, and it is what Dyna planning *adds*: it
transmits belief error into locked-in policy with high confidence. (ii)
*Late-horizon instability of the rules-backend cells*: A2/dyna-rules left a
machine-zero-gain point after iteration ~129, and D/dyna-rules worsened
0.0928 → 0.8431 between caps 180 and 360 — the paper's A2 instability
signature, now seen twice. (iii) *Unsettled oscillation*: B/q, B/dyna-emp
(the one dyna cell whose plateau is not belief-certified, `s ≈ 1.9`), D/q,
and D/dyna-emp all oscillate without settling — a Q-learning convergence
failure at these hyperparameters, not a data-quantity problem. All
extended-run artifacts live beside the primary grid
(`runs/C/q/0-ext720`, `runs/D/q/0-ext720`, `runs/D/dyna-emp/0-ext360`,
`runs/D/dyna-rules/0-ext360`); every extended run's first-180
`br_gain_series` entries equal its primary run's exactly, confirming the
same-seed replay premise.

## Extensions grid (Task 18: 2³ ablation, E1–E3, decay addendum)

**Verdicts for the H4 flag attribution (ext §3) and H7/H8/H9 (ext §5),
plus the decay addendum (ext-runner docstring, addendum A), from the
169-cell extensions grid.**

Reproduce with:

```bash
uv run python scripts/run_extensions.py --jobs 8 --out runs/extensions
```

Artifacts under `runs/extensions/<arm>/<econ>/<cell>/<seed>/`, one summary
record per cell in `runs/extensions/summary.json` (169 records: ablation
16, e1 30, e2 15, e3 36, decay 72). All cells completed (driver rc=0);
every muzero cell stopped D1-converged at `n_iters=108` (the H3
ζ-schedule clock); decay cells are q/dyna learners and hit the
`n_iters_max=180` cap except the seven noted below. Reading notes that
apply grid-wide:

- **Headline: all 169 cells classify UNSETTLED** under the core §8
  classifier (`NASH` iff `br_gain <= 0.05` **and** `stationarity <=
  1e-3`) — no extension arm reaches the NASH classification anywhere,
  against 3/100 NASH in the replication campaign
  (`runs/campaign-replication/summary.json`) and 0/150 in the Stage-2
  campaign (`runs/campaign-extensions/summary.json`). As in H4, the
  stationarity leg is a finite-N `O(1/sqrt(1800)) ≈ 0.024` histogram
  floor for mixed-holdings equilibria, so several cells below reach
  `br_gain = 0.0` exactly yet still classify UNSETTLED; the informative
  cross-cell signal in this grid is `br_gain`.
- Six e3 cells (`eps0.1-{k0.5,k2,k8}` seeds {0,2}) had their deployment
  `br_gain` recomputed post-hoc from the saved deployment policies
  (`arrays.npz`) after the RVI iteration-cap raises (commits `aeee24e`,
  `bf1688a`); each carries a `br_gain_recompute` provenance block in its
  `trace.json`. The recompute is a deterministic function of the saved
  arrays — no rerun.
- `br_gain_series` aggregation is nan-aware throughout: 8 decay q-cells
  carry 22 per-snapshot cap-type RVI `nan` holes (snapshot policies are
  not persisted, so these are not recomputable); every *deployment*
  `br_gain` in the grid is finite.

### 2³ ablation — H4 flag attribution

Full `{known_rules, greedy_backup, centered_values} ∈ {0,1}³` grid on C
and D, seed 0, muzero-emp base. Deployment `br_gain`:

| kr | gb | cv | C | D |
|---|---|---|---|---|
| 0 | 0 | 0 | 0.594160 | 0.626035 |
| 0 | 0 | 1 | 0.594160 | 0.626035 |
| 0 | 1 | 0 | 0.0 | 0.543806 |
| 0 | 1 | 1 | 0.0 | 0.543806 |
| 1 | 0 | 0 | 0.517784 | 0.270960 |
| 1 | 0 | 1 | 0.517784 | 0.270960 |
| 1 | 1 | 0 | 0.0 | 3.44e-11 |
| 1 | 1 | 1 | 0.0 | 0.020175 |

Corner sanity: the all-off corners equal the H3 seed-0 baselines to the
last digit (C 0.5941598738745393, D 0.6260347618409483) and the all-on
corners equal the H4 `muzero-known` values (C 0.0, D 0.0201752), so the
flag plumbing is byte-consistent with the replication artifacts.

**Attribution — the rescue carrier differs by economy, contradicting ext
§3's prior expectation that `known_rules` dominates:**

- **C: `greedy_backup` alone carries the entire rescue.** Every gb=1
  cell has `br_gain = 0.0` exactly, with or without the known rules, and
  the empirical terminal holdings of `kr0-gb1-cv0` sit on the H1 C-money
  fiat SNE (per-type within finite-N noise of Fig. 8: T1 [0.272, –,
  0.728, –], T2 [0.303, 0.272, –, 0.425], T3 [0.385, 0.615, –, –]).
  `known_rules` alone only nudges C from 0.5942 to 0.5178 — the same
  0.52-band trap value the Stage-2 C/q and C/dyna-rules cells sit at
  (0.517958). Switching the planner's backup operator from average to
  max — with the *empirical* model unchanged — is what closes C's gap.
- **D: the `known_rules × greedy_backup` interaction carries it.**
  Neither flag alone suffices (gb: 0.626→0.544; kr: 0.626→0.271); both
  together reach 3.44e-11, landing on the H1 D commodity-money SNE
  (T1 [–, 0.103, 0.897, –, –], T2 [0.719, –, 0.075, 0.206, –], T4
  [1.0], T5 [0.8, 0.2]).
- **`centered_values` is inert outside the full stack — byte-identical
  artifacts.** In six of the eight cv0/cv1 pairs (every pair with kr=0
  or gb=0) the *entire* `arrays.npz` is byte-identical
  (`np.array_equal` on all arrays): flipping cv changed no decision on
  the whole trajectory. Only inside kr1-gb1 does cv do anything, and
  there it is mildly harmful on D (3.44e-11 → 0.020175, the H4
  residual) and payoff-neutral on C (0.0 → 0.0 with a slightly
  different endpoint).

All 16 cells `converged=True` at `n_iters=108` and all classify
UNSETTLED (the zero-gain cells via the finite-N stationarity floor), so
the brief's spot-re-seed trigger (a cell whose *classification* differs
from its neighbors) never fired; no re-seeds were run.

### E1 — EMA/constant-gain empirical model (H7)

`--ema-gain λg ∈ {0.005, 0.02, 0.05}` × {C, D} × seeds 0–4, muzero-emp.
Deployment `br_gain` (5 seeds, sorted) against the 10-seed muzero-emp
baselines from `runs/campaign-replication`:

| cell | C | D |
|---|---|---|
| baseline (λg=0, 10 seeds) | min 0.0047, med 3.80, max 31.85 | min 0.2705, med 1.713, max 1.714 |
| ema0.005 | 2.59, 16.4, 16.4, 16.7, 16.7 | 0.058, 0.270, 0.270, 0.270, 0.313 |
| ema0.02 | 0.511, 0.514, 12.5, 14.7, 14.7 | 0.057, 0.313, 0.507, 1.70, 3.06 |
| ema0.05 | 0.513, 0.514, 0.515, 0.515, 5.25 | 0.0018, 0.057, 0.058, 2.67, 3.97 |

**H7 (paper §8 conjecture, prediction on record: rescues D, not C) —
confirmed in direction at λg=0.005, with caveats at larger gains.**

- **D's stale-anchor plateau disappears at λg=0.005.** Six of ten
  baseline D seeds sit within 1e-3 of gain 1.714 (a seventh at 1.173) —
  the stale-anchor plateau, incidentally the paper's own Table-2 D value
  1.71. Under λg=0.005 no seed lands there: all five fall in
  [0.058, 0.313], at or below the best baseline seed (0.2705). That is
  precisely the conjectured mechanism — constant-gain forgetting
  removes the stale anchors — though "rescue" means gains of 0.06–0.31,
  not NASH classification (stationarity floor, above). At λg ∈ {0.02,
  0.05} D turns bimodal: some seeds reach the grid's best D-empirical
  values (0.0018, 0.057) while others destabilize past the baseline max
  (3.06, 3.97) — too much forgetting trades the plateau for variance.
- **C is never rescued, at any gain.** The best C cell across all 15 is
  0.511 — the familiar 0.51–0.60 trap band — and λg=0.005 makes C
  dramatically *worse* (16.4–16.7 on four of five seeds, versus
  baseline median 3.80): slow forgetting starves the already-sparse
  fiat-link counts faster than data replenishes them. The missing-data
  mechanism (H7's "not C" clause) is exactly what faster averaging
  cannot fix.

### E2 — per-agent vs population-shared tables (H8)

`--per-agent` × {A1, C, D} × seeds 0–4, muzero-emp. **Measurement
caveat that governs every number in this subsection: the per-agent
deployment artifact is the plain MEAN of the per-agent `pi_dep(1)`
policies over each type's agents (`learners/muzero.py`,
`deployment_policy`), NOT the policy of the mean model** — the
diagnostics score the population's mixture policy, which is the
population-level behavior under matching but is not any individual
agent's policy. Deployment `br_gain` of that mean policy (5 seeds,
sorted), against shared-table baselines:

| econ | per-agent (mean of per-agent policies) | shared baseline (10 seeds) |
|---|---|---|
| A1 | 1.9e-6, 1.9e-6, 2.3e-6, 0.0053, 0.0058 | 5.05e-11 (all 10 seeds) |
| C | 0.0046, 0.0062, 0.0077, 0.0095, 0.0124 | min 0.0047, med 3.80, max 31.85 |
| D | 0.189, 0.216, 0.224, 0.293, 0.329 | min 0.2705, med 1.713, max 1.714 |

**H8 ("population-shared tables are not the cause of the traps;
per-agent learners land in the same attractor class, slower") — refuted
as stated, at the population-mixture level.** Per-agent C does *not*
land in the shared-table attractor class: every seed's mixture policy
sits at gain 0.005–0.012, two to three-and-a-half orders of magnitude
below the shared baseline on the same seeds (0.594–31.85), and per-agent
D (0.19–0.33) sits at-or-below the best shared seed on every draw,
never on the 1.714 plateau. Nor is "slower" observed: every per-agent
cell D1-converges at the same `n_iters=108`. A1 moves the other way
(machine-zero → 1.9e-6–0.0058, still small): heterogeneity costs a
little where sharing was already optimal, and helps a lot where sharing
trapped. The honest scope limit is the caveat above — these diagnostics
certify the *population mixture* near-best-responds; whether individual
agents are trapped is not measured by this artifact (per-agent tables
are not persisted). Within that scope, experience pooling is implicated
in — not exonerated of — the C trap severity. All 15 cells UNSETTLED
(stationarity floor).

### E3 — exploration sweep (H9)

`--eps-floor ∈ {0.01, 0.03, 0.1, 0.2}` × `--kappa0 ∈ {0.5, 2, 8}` on C,
seeds 0–2, muzero-emp (default cell is eps0.03-k2). Deployment
`br_gain` by seed:

| cell | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| eps0.01-k0.5 | 9.960 | 0.517 | 15.15 |
| eps0.01-k2 | 9.960 | 0.517 | 15.15 |
| eps0.01-k8 | 9.960 | 16.00 | 15.59 |
| eps0.03-k0.5 | 0.594 | 11.29 | 8.851 |
| eps0.03-k2 (default) | 0.594 | 11.29 | 0.596 |
| eps0.03-k8 | 0.594 | 0.597 | 16.82 |
| eps0.1-k0.5 | 1.4357 | 12.30 | 1.4238 |
| eps0.1-k2 | 1.4357 | 12.30 | 1.4238 |
| eps0.1-k8 | 1.4357 | 16.15 | 1.4358 |
| eps0.2-(all κ0) | 17.3296 | 17.3296 | 17.3296 |

**H9 (C's collapse is structural, not exploration-limited) —
confirmed.** No cell in the 4×3×3 grid restores the fiat links: the
grid minimum is 0.5174 (eps0.01, seeds 1 at k0.5/k2) — the same
0.52-band trap — and both directions away from the default ε=0.03 are
monotonically bad. Larger ε strictly hurts: at ε=0.1 the best seeds sit
at 1.42–1.44 (the ε-floor's own exploration cost is already first-order
in the gain), and at ε=0.2 **all nine cells collapse onto a single
attractor with `br_gain = 17.3295719` identical to 10 significant
digits across every κ0 and every seed** — exploration noise at that
level erases both the prior and the matching-RNG seed from the outcome.
Smaller ε (0.01) also hurts (9.96–16.0 on most seeds). Weaker or
stronger priors do essentially nothing:

- **κ0 shapes search, not the deployed greedy policy.** For k0.5 vs k2
  (a 4× prior-strength change), the deployed policies are
  **byte-identical** in 8 of 12 (ε, seed) pairs — all of eps0.01 and
  eps0.1 plus eps0.03 seeds 0–1 (`np.array_equal` on
  `alpha_star`/`beta_star`/`mu_star`; for the eps0.1 recomputed cells
  the only non-equal array is the NaN-poisoned saved `br_gaps` vector,
  and NaN ≠ NaN). At eps0.03 seed 0 the *training* tables differ at the
  1e-3 level while the deployment extraction erases even that. The one
  k0.5-vs-k2 divergence is eps0.03 seed 2 (8.851 vs 0.596) — a basin
  flip, i.e. seed-lottery, not systematic improvement. k8 is never
  byte-identical to the others but lands in the same gain bands
  (e.g. eps0.1: 1.4357–1.4358 vs 1.4238–1.4357).

The coordinated-experimentation argument stands: rebuilding C's
multi-link reciprocal chain needs *joint* two-sided experimentation
(~ε² per link per round), which no one-dimensional exploration knob
supplies — turning ε up buys the links nothing before its own
exploration cost dominates the deployed policy's neighborhood. All 36
cells UNSETTLED.

### Decay addendum — ε-decay and lr-decay on the stuck Stage-2 cells

Both user hypotheses on record — (i) annealing exploration from a high
start (`--eps-start 0.5`, decayed on the ζ schedule to the 0.03 floor)
and (ii) Robbins–Monro-*style* per-visit learning-rate decay
(`--q-lr-visit`: step `q_lr / sqrt(n_visits(cell))`) — applied
separately and combined to the eight trapped Stage-2 cells (B,q)
(B,dyna-emp) (C,q) (C,dyna-rules) (A2,dyna-rules) (D,q) (D,dyna-emp)
(D,dyna-rules), seeds 0–2 = 72 cells. Framing note: `1/sqrt(n)` is
*not* square-summable, so no classical Robbins–Monro convergence
guarantee applies; the honest rationale is taming per-cell sampling
noise under sparse visitation, and the observed rescues are consistent
with exactly that (they occur where the Stage-2 failure was an
oscillating plateau) rather than with any asymptotic-convergence story.
Deployment `br_gain` (3 seeds, sorted) with the Stage-2 seed-0 baseline:

| cell family | baseline | eps | lr | eps-lr |
|---|---|---|---|---|
| B/q | 7.222 | 1.5, 2.0, 8.06 | **0, 0, 0** | **0, 0, 0** |
| B/dyna-emp | 2.889 | **0, 0, 0** | 0, 4.25, 4.25 | 0, 4.25, 4.25 |
| C/q | 0.518 | 0, 0, 0.518 | 0, 0.518, 0.596 | 0, 0, 0.518 |
| C/dyna-rules | 0.518 | 0.518 ×3 | 0, 0.518, 0.518 | 0, 0.518, 0.518 |
| A2/dyna-rules | 34.16 | 6e-12, 34.16, 34.16 | 0, 21.44, 21.44 | 5e-12, 21.44, 21.44 |
| D/q | 3.677 | 3.47, 5.46, 17.6 | 0.240, 0.251, 0.386 | 0.251, 0.314, 0.416 |
| D/dyna-emp | 0.863 | 0.271, 1.22, 3.76 | 0.012, 2.89, 12.5 | 2.52, 2.59, 3.05 |
| D/dyna-rules | 0.0928 | 0.271, 2.47, 3.62 | 0, 0.0058, 2.33 | 2.84, 3.05, 5.01 |

**Verdict: neither schedule change (nor both) rescues Nash convergence
for q or dyna on any economy — 0/72 NASH classifications — but the
gain-level structure is informative and asymmetric:**

- **lr-decay is the strongest single intervention observed, and it is
  learner- and economy-specific.** It eliminates B/q's 7.22 trap on all
  six lr-containing cells (`br_gain = 0.0` exactly) and produces the
  **first (D1) convergences ever recorded for a Stage-2 learner**:
  seven decay cells reach `converged=True` before the cap (B/q-eps-lr
  seeds 0–2 at n=108/162/110, B/q-lr/2 at 111, C/q-lr seeds 0–1 at
  144/171, C/q-eps-lr/2 at 144) — shrinking per-visit TD steps is what
  lets the (D1) Δ̄ metric settle for hard-greedy Q policies. It also
  tames D/q by an order of magnitude (3.68 → 0.24–0.42, never 0). But
  it *hurts* B/dyna-emp (4.25 > 2.889 on four of six lr cells) and is
  a lottery on D/dyna-emp (0.012 vs 12.5).
- **ε-decay from 0.5 rescues exactly one family**: B/dyna-emp goes to
  0.0 on all three seeds. Elsewhere it leaves traps in place (C/
  dyna-rules 0.518 ×3), or actively destabilizes (D/q up to 17.6;
  D/dyna-rules 0.093 → 0.27–3.62).
- **A2/dyna-rules' late-destabilization pathology persists** under
  every schedule: most cells land on 21.444 — the fundamental-branch
  gain at ubar=500, the paper's own A2 Table-2 band (≈21.3) — or the
  34.16 speculative-departure value, with occasional machine-zero
  seeds (3 of 9 at ≤1e-11 or 0). Schedules reshuffle the basin lottery;
  they do not close it.
- **C remains a seed lottery under every schedule** (zeros on 7 of 18
  C cells, the 0.518 trap on the rest): consistent with H9's
  structural reading — the schedules change *when* exploration and
  learning stop moving, not whether the fiat chain gets rebuilt.

Overall: 20 of 72 decay cells reach `br_gain = 0.0` exactly (plus two
at ≤1e-11), yet all 72 classify UNSETTLED via the stationarity floor,
and no family is rescued on all seeds by any schedule except B/q under
lr-decay and B/dyna-emp under ε-decay. The Stage-2 traps are therefore
not artifacts of the constant-step/constant-ε defaults alone — the
schedules move individual cells, but the trap phenomenology (seed
lotteries, basin flips, late destabilization) survives both hypotheses.

### Extensions verdict (headline)

**All 169 extension cells classify UNSETTLED — no extension arm reaches
the NASH classification anywhere** — matching the Stage-2 campaign's
150/150 UNSETTLED (the replication campaign's 3/100 NASH remain the
only NASH-classified runs in the project, all near-degenerate-holdings
draws that beat the finite-N stationarity floor). Within that uniform
classification the grid still resolves all four questions it was built
for: the H4 rescue is carried by `greedy_backup` alone on C and by the
`known_rules × greedy_backup` interaction on D, with `centered_values`
byte-inert outside the full stack (ablation); the paper's §8 EMA
conjecture H7 is confirmed in direction — λg=0.005 removes D's 1.714
stale-anchor plateau and never helps C (E1); H8 is refuted as stated —
per-agent learners' population-mixture policy lands orders of magnitude
closer to best response on C, not in the same attractor class slower,
with the mean-of-policies measurement caveat (E2); H9 is confirmed —
C's collapse is structural, no (ε, κ0) combination restores the fiat
links, ε=0.2 collapses all runs onto one 17.33-gain attractor, and κ0
mostly does not even change the deployed policy bytes (E3); and the
decay addendum shows schedule changes produce scattered exact-zero
gains (20/72) and the first Stage-2 (D1) convergences without rescuing
any cell family to NASH.

## Final verdict: H1–H9 with campaign tables and figures (Task 19)

The tables and figures cited below are regenerated read-only from the three
campaign summaries by `scripts/make_tables.py` (→ `reports/tables.md`, tables
T1–T4) and `scripts/make_figures.py` (→ `reports/figures/*.png`). Both scripts
fail loud (nonzero exit, naming every absent record) against the expected
100 + 150 + 169 run set before emitting anything; the summaries carry no null
br_gain/stationarity, so tables use plain (nan-safe) medians, and the 25
seed-0 series the figures plot contain no nulls. Seed choices are fixed and
disclosed in-figure: μ★ and learning paths at seed 0; the C trade-policy
heatmap contrasts seed 0 (trap band, br_gain 0.594, policy structurally near
Nash) with seed 8 (trapped, 13.98).

**Headline.** `3/100` replication runs classify NASH vs `0/150` Stage-2 and
`0/169` extensions (T2 totals; `outcome_frequency.png`) — the only NASH-class
runs in the project. But this classification count *understates* the result:
at the gain level (br_gain ≤ 0.05, ignoring stationarity) the same campaigns
reach `71/100`, `74/150`, `41/169`. The gap is the finite-N stationarity floor
(≈1/√1800 ≈ 0.024), not distance from Nash — every table and verdict reports
both readings so the negative "0/169 NASH" is not read as "never near Nash".

| H | Claim (abbrev.) | Verdict | Primary evidence |
|---|---|---|---|
| H1 | Algorithm 1 reproduces the KW/MMS stationary Nash equilibria | **Confirmed** | §H1 (verdict); Nash benchmark bars in `holdings_*.png` |
| H2 | Algorithm 2 yields non-Nash SCEs (residuals ≈0, positive full-info gain) | **Confirmed** | §H2 (verdict); SCE bars in `holdings_A1/C/D.png` |
| H3 | Fully empirical learner: A1/B→Nash, A2 gain≈21, C low-trade (broken fiat links), D plateau≈1.7 | **Confirmed (qualitative); C magnitude softer, classification blocked by finite-N floor** | T1/T2 muzero-emp rows; `learning_*.png`; `trade_policy_C.png` |
| H4 | Known-environment learner reaches gain≈0 in all five economies | **Confirmed at gain level; NASH classification only on A1 (finite-N floor)** | T1/T2 muzero-known rows; §H4 |
| H5 | Known-env rescue is from model contents, not planner (dyna-rules≈known, dyna-emp≈empirical) | **Refined / partly refuted — attribution is economy-dependent** | T4 ablation; T1/T3 Stage-2; §H5/H6 |
| H6 | Model-free Q is at least as trap-prone as the empirical learner | **Confirmed** | T1/T2/T3 `q` rows |
| H7 | EMA/constant-gain fixes D's plateau but not C's collapse | **Confirmed in direction** | T4 E1; §E1 |
| H8 | Per-agent tables land in the same attractor class, slower | **Refuted as stated (mean-of-policies caveat)** | T4 E2; §E2 |
| H9 | C's collapse is structural, not cured by moderate ε / weaker priors | **Confirmed** | T4 E3; §E3 |

**H1, H2 (solver level).** Established in §H1/§H2 from the `nash_*`/`sce_*`
benchmark runs, not the learner campaigns; here they anchor the learner figures
as the black (Nash) and silver (SCE) bars in `holdings_*.png`, visibly distinct
attractors (e.g. C's SCE-low-trade hoards good 1/2 and nearly abandons good 3).

**H3.** T1 muzero-emp: A1 median 5.05e-11 and B median 0 sit at Nash (paper
1.50e-12, 0); A2 median **21.2 ≈ paper 21.3** with IQR [~0, 21.2] exposing the
fundamental/speculative branch lottery (`holdings_A2.png`: muzero-emp on the
speculative branch); D median **1.71 = paper 1.71**. Attribution note: the A2
`br_gain=34.16` from Finding #3 is the *Stage-2 dyna* speculative-departure
realization (dyna-emp seeds 2,5; dyna-rules seeds 0,6,8; surfaced in T4), not
the fully-empirical learner — no muzero-emp seed exceeds 21.44 across the
10-seed campaign and the Stage-1 single run `runs/A2/muzero-emp/0` is ≈6.2e-12
(machine zero), so the paper's Table-2 21.3 maps to muzero-emp's median 21.2
while 34.16 belongs to a different learner family. C is the one softened
magnitude — median **3.80 [0.529, 13.3] vs the paper's single-run 12.8**: our
10 seeds split into a low-gain trap band (four seeds ≈0.5, plus one genuinely
near-Nash draw: seed 5 at 0.0047) and a trapped cluster (7–32), so the median
lands below the paper's one draw, while the *mechanism* the claim
names is confirmed — broken fiat links are explicit in `trade_policy_C.png`
(seed 8). Caveat: classification stationarity is evaluated at the terminal
empirical μ while br_gain uses μ★ (17-m1), a best-effort reconstruction; four
C/muzero-emp deployment gains carry `br_gain_recompute` provenance (treated as
first-class).

**H4.** T1 muzero-known medians are ≈0 in every economy (A1 0, A2 0, B 0, C 0,
D 4.65e-12) — the claim's substance. Classification NASH is nonetheless 1/10
(A1 only): on C/D the population one-step residual sits at the finite-N floor
(~0.01–0.02 > the 1e-3 NASH threshold), so these are gain-level Nash that miss
the classification gate (T2 muzero-known gain≤0.05: A1 100%, B 100%, C 90%,
D 90%). This is Finding #9, not a failure to reach Nash (§H4).

**H5/H6.** H6 is clean (T2: `q` NASH 0/10 everywhere; T1 trap gains B 2.89,
C 0.518, D 5.33 — as trap-prone or worse than empirical). H5 is economy-
dependent: the 2³ ablation (T4) shows C's rescue is carried by `greedy_backup`
(planner) alone → gain 0, with `known_rules` alone leaving the 0.518 trap —
opposite to "mainly model contents"; on D only the `known_rules × greedy_backup`
interaction closes the gap (3.44e-11). So H5's attribution holds only in part
(Finding #4; §H5/H6).

**H7, H8, H9 (extensions).** H7 confirmed in direction (T4 E1: D ema0.005 →
0.270, off the 1.71 plateau; C ema0.005 → 16.4, worse). H8 refuted as stated
(T4 E2: C per-agent gain 0.008, D 0.224 — orders of magnitude closer to best
response than the shared-table traps, not "same class slower"); **mandatory
caveat: E2 artifacts are the MEAN of per-agent π_dep(1) policies, not the policy
of a mean model — per-agent trap status is untestable post-hoc**. H9 confirmed
(T4 E3: no (ε, κ0) cell reaches Nash, min median 0.596; ε=0.2 collapses all
cells onto a single 17.3-gain attractor). Decay addendum (T4): 0/72 NASH, but
lr-decay produced the first Stage-2 (D1) training-loop convergences ever
recorded — exactly 7 cells (B/q-eps-lr 3/3, B/q-lr 1/3, C/q-eps-lr 1/3,
C/q-lr 2/3), read from the trace `converged` flag; the 1/√n schedule is not
square-summable, so this is noise-taming, not a classical convergence guarantee.

**Fidelity registry pointers.** Slow-mixing hazard is seed-dependent, not
uniform (C/muzero-emp seed 4 spectral gap ≈4.6e-6; caps raised to 40M, tols
never loosened). Stale-cap-era values are identified by null br_gain + log
warnings, not git_sha (17-m5); none survive in the current summaries. Extension
ablation cells are single-seed (marked in T4); E1/E2 are 5 seeds, E3/decay 3.
