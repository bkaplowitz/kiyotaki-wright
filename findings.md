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
| T2 | good1=1.00 | good1=0.72 | path-dependent split (see below) |
| T3 | good1=1.00 | good1=1.00 | MATCH |
| T4 | good1=1.00 | good1=1.00 | MATCH |
| T5 | good1=1.00 | good1=1.00 | MATCH |

Paper type 1 (code index 0) sits in autarky on its production good 3 with
beliefs `q_1 = 0` while the other four types keep trading toward the
commodity money good 1 -- the paper's "one skeptic in a working economy"
endpoint. T2's paper value 0.72 leaves a 0.28 residual whose placement is
recurrent-class/path dependent (paper p. 19: with T1 absent from trade,
several absorbing splits exist); our run funnels all of T2's mass to good
1, so we pin only the lower bound `mu[T2, good1] > 0.7`
(`test_d_type1_autarky_sce`). `br_gain = 80.0` (T1): the working economy
around the skeptic would objectively buy its good. Verdict: **MATCH** on
T1/T3/T4/T5 (the values Fig. 10 prints as 1.00), **CONSISTENT** on T2
(path-dependent residual).

### H2 verdict

**Confirmed.** All three misdirectional prompts converge to genuine SCE
(subjective optimality gap `s <= 1e-9`, on-path belief error `b <= 1e-9`,
distributional residual `d <= 1e-12`) that complete information reveals to
be far from best responses: `br_gain` of 19.9 (A1), 33.3 (C), 80.0 (D) --
each dominated by the misdirected type -- versus 0 at every H1 Nash point
(`test_nash_is_sce` pins the converse: started from the A1 Nash
equilibrium, `solve_sce` stays there with `br_gain <= 1e-6`). Subjectively
optimal, objectively poor, exactly as the paper's §7 misdirection
experiments report. The A2 variant (paper fn. 10, deviation gain ~21.4)
was not run -- optional in the brief and skipped.
