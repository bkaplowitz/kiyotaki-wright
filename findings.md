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
since it is the load-bearing fact behind three of the four mismatches.

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

### C -- MISMATCH (br_gain and stationarity both ~5-6x lower than paper)

`br_gain = 0.594` (paper `12.8`), `stationarity = 0.0223` (paper `0.130`).
Deployment aggregate `[0.300, 0.1375, 0.3494, 0.2131]` vs the H1 C-money
SNE aggregate `[0.320, 0.2930, 0.2456, 0.1414]` -- visibly short of the
true Nash point on goods 2-4, consistent with the nonzero `br_gain`. Same
training/deployment decoupling as A1: the training-table `br_gain_series`
sits at ~2.16-2.19 for the last 10 iterations (an order of magnitude
*above* the deployment number), so the deployment PUCT pass (adj 8) is
doing real work here but does not fully close the gap within its fixed
budget (`h_dep=14, n_sims_dep=96, tau_dep=0.35`) the way it does for A1.

**Broken fiat link, deployment (unclipped) policy** -- the three cells
architecture.md §15 names (`α₂(0,2)=0.001, α₂(2,0)=1.000, α₃(1,2)≈0`,
paper 1-indexed types, C's own goods already unshifted since
`economies.py`'s C economy uses `s=[0.0,9.0,14.0,29.0]` over goods `{0
(fiat), 1, 2, 3}` matching the paper 1:1): reading these as `alpha[i-1,
k, j]` under the codebase's own `alpha[i,k,j]` = "type i holds k, meets
counterparty holding j" convention (confirmed at `simulate.py:131-132`
and architecture.md's `F_i(k,j)` formula, §6) gives `alpha[1,0,2]=0.9994`
and `alpha[1,2,0]=0.00498` -- the **opposite** of the paper's stated
`(0.001, 1.000)` pair. Reading them with `k` and `j` swapped instead
(`alpha[1,2,0]` for `α₂(0,2)`, `alpha[1,0,2]` for `α₂(2,0)`) matches
closely: `alpha[1,2,0]=0.00498 ≈ 0.001` and `alpha[1,0,2]=0.9994 ≈ 1.000`.
`α₃(1,2)` is small under either reading (`alpha[2,1,2]=2.24e-6` or
`alpha[2,2,1]=3.84e-3`), so it does not discriminate. Per controller
resolution 5 (do not chase literal cell equality; record actual values,
interpret "broken fiat link" as deployment-probability ~0), the numeric
match is reported as-is rather than silently resolved: whichever argument
order architecture.md's §15 prose intends, our C run reproduces the
qualitative signature exactly -- for type T2, one direction between fiat
(good 0) and good 2 is pinned near 1 and the reverse near 0
(near-deterministic, asymmetric), not the symmetric ~0.5 a naive prior
would give. **Training-table (eps-clipped) values for the same four cells**
are `alpha_train_final[1,0,2]=0.446`, `[1,2,0]=0.927`, `[2,1,2]=0.107`,
`[2,2,1]=0.849` -- compressed into `[eps_floor, 1-eps_floor] = [0.03,
0.97]` as expected (adj 1/5) and less sharply polarized than the unclipped
deployment read, but showing the same asymmetric ordering.

### D -- MISMATCH (br_gain ~2.7x low, stationarity ~7x low)

`br_gain = 0.626` (paper `1.71`) driven almost entirely by T2
(`br_gaps_deployment = [0.0606, 0.6260, 5.3e-15, 0, 0]`); `stationarity =
0.0235` (paper `0.173`). Deployment aggregate `[0.494, 0.247, 0.125,
0.086, 0.048]` vs the H1 D-default SNE aggregate `[0.612, 0.077, 0.192,
0.045, 0.074]` -- good-1 share short by ~0.12, again consistent with a
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
applied.** A1 and B reproduce paper Table 2 to the stated order of
magnitude (both effectively exact zero). A2 mismatches in the "good"
direction -- our seed lands on the paper's harder-to-find speculative
Nash instead of the fundamental basin the paper's `br_gain≈21.3` implies
-- consistent with architecture.md §15's own caveat that single-seed
numbers test seed variance, not the underlying claim; §12 items 1 and 7
are the plausible basin-selection levers, left untouched. C and D
mismatch in the "not yet converged" direction (`br_gain` 0.59 vs 12.8,
0.63 vs 1.71; `stationarity` 0.022 vs 0.130, 0.024 vs 0.173) -- both
economies' training tables reach a ζ-mixing fixed point well short of the
true Nash by n=108, and the deployment-time PUCT refinement pass (adj 8)
narrows but does not close that gap; §12 item 9 (Δ̄ defined on post-clip
stored tables, hence blind to the true residual) is the primary suspect,
with items 3 (memoization) and 11 (deployment exploration floor) as
secondary candidates. C's specific "broken fiat link" cells reproduce
the paper's qualitative signature (asymmetric, near-deterministic T2
policy between fiat and good 2) but only after swapping the `(k,j)`
argument order against architecture.md §15's literal cell labels -- an
interpretation gap in that cross-reference, not a code defect, reported
with both readings rather than silently resolved. No hyperparameter was
changed and no library code was edited to chase any of these three
mismatches; multi-seed campaigns (Task 17) are the right next step to
determine whether C/D's gap is a per-seed artifact of the shared,
economy-independent (D1) stopping point or a systematic shortfall.
