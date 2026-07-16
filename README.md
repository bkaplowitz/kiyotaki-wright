# kiyotaki-wright

Replication of Sargent & Yang (2026), *"How Should I Trade? Stationary Nash, Self-Confirming,
and Learned Equilibria in Kiyotaki–Wright Exchange Economies"* (PDF at repo root), plus a
Stage-2 study replacing the paper's learner with standard RL baselines and a Stage-3
extensions study (ablations, EMA world-models, per-agent model banks, exploration and
decay-schedule sweeps).

## Layout

- `src/kw/` — core library: economies, Bellman machinery, RVI Nash solver (paper
  Algorithm 1), SCE fixed-point solver (Algorithm 2), simulation harness, tabular PUCT
  self-play learner (Algorithms 3–4), diagnostics. `src/kw/extensions/` holds Stage-2/3
  learners (differential Q-learning, Dyna) and extension hooks; core never imports from
  it (enforced by test).
- `scripts/` — drivers (`run_learning.py`, `run_campaign.py`, `run_extensions.py`) and
  report generators (`make_tables.py`, `make_figures.py`).
- `runs/` — committed campaign artifacts: `campaign-replication` (100 seeds),
  `campaign-extensions` (150 Stage-2 runs), `extensions` (169 Stage-3 runs). Each run
  carries `trace.json` provenance; each campaign a `summary.json`.
- `reports/` — generated tables (`tables.md`) and 12 figures; regenerate byte-identically
  from `runs/`.
- `findings.md` — hypothesis verdicts H1–H9 with pointers into tables, figures, and
  artifacts.
- `architecture.md`, `extensions/architecture.md`, `plan.md` — specs and the 19-task
  implementation plan.
- `docs/certs/` — solver fixed-point certificates for the benchmark economies.

## Reproduction

```sh
uv sync                      # Python >= 3.12
uv run pytest -q             # 98 passed, 3 deselected (slow integration marks)
uv run pytest -q -m ''       # include the slow-marked integration runs

uv run python scripts/make_tables.py    # rebuilds reports/tables.md from runs/
uv run python scripts/make_figures.py   # rebuilds reports/figures/*.png from runs/
```

Both report scripts read `runs/` only and fail loudly, listing every missing artifact.
Single learning runs go through `scripts/run_learning.py --help`; multi-seed campaigns
through `scripts/run_campaign.py` and `scripts/run_extensions.py`.

## Determinism and numerical notes

- All committed tables and figures regenerate byte-identically from the committed
  artifacts. Reproducing the artifacts themselves depends on the exact RNG stream
  construction documented in `src/kw/simulate.py` (two same-seeded generators, kept for
  bit-compatibility with the committed runs); new run-sets should prefer
  `default_rng(seed).spawn(...)`.
- Learned near-deterministic policies can produce Markov chains with spectral gaps of
  order 1e-6 per iteration. Solver iteration caps are sized empirically for this regime
  (invariant/RVI caps at 40M); tolerances (1e-12 distribution, 1e-10 Bellman) are
  contracts and are never loosened. Anyone planning follow-on stages should read the
  post-plan note at the end of `plan.md` first.
