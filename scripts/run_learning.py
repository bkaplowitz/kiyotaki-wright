#!/usr/bin/env python
"""CLI: run a population-learning economy (Algorithms 3-4) to the (D1)
stopping rule, extract a deployment policy, and save the run artifact with
per-iteration best-response-gain / stationarity diagnostics.

Usage:
    uv run python scripts/run_learning.py --economy A1 --learner muzero-emp \
        --seed 0 --out runs/A1/muzero-emp/0

`--dry-run` does not print-and-exit (unlike compute_nash.py/compute_sce.py,
whose solvers have no population-size knob to shrink); it swaps in the
repo's shrunk smoke-test config (n_agents=60, rounds=5, n_iters_max=6,
n_min=2, patience=2 -- the same DRY config the test suite uses) and runs the
*entire* pipeline -- simulation, diagnostics, save -- at that tiny scale.

Learner registry: `--learner muzero-emp` is the paper's fully-empirical
MuZero variant (all three ablation flags off); `--learner muzero-known` is
the known-environment variant (all three flags on -- architecture.md
§7.2/§12.12). Both route through `MuZeroLearner`; muzero-known currently
raises NotImplementedError because `planner.py` has no RulesBackend yet
(arrives with Task 14) -- this is expected and reported as a clean CLI
error, not swallowed. `q`/`dyna-emp`/`dyna-rules` are registry stubs
(Stage-2 learners, Tasks 15-16); each raises the same clean error. Landing
those tasks means replacing exactly one registry entry apiece -- the
muzero-known entry needs no change at all once RulesBackend exists.

Per-iteration diagnostics: `RunResult.trace` (simulate.py) carries no policy
snapshots by design (the fairness contract forbids modifying simulate.py to
add them). `_SnapshotDelegate` instead wraps the learner and forwards every
`Learner`-protocol call unchanged, appending a copy of `learner.alpha` /
`learner.beta` after each `end_iteration(n)` call -- the convention this
script expects of any learner it drives (`MuZeroLearner` already exposes
these; future Stage-2 learners should too). After the run, `br_gain`/
`stationarity` are computed once per snapshot (mu=None -> invariant under
that iteration's policy for br_gain; the iteration's empirical
`agg_holdings` for stationarity) and saved alongside the raw trace.
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

from kw.artifacts import save_run
from kw.diagnostics import br_gain, stationarity
from kw.economies import ECONOMIES, get_economy
from kw.learners.base import LearnerConfig
from kw.learners.muzero import MuZeroLearner
from kw.simulate import run_learning

DRY_OVERRIDES = {"n_agents": 60, "rounds": 5, "n_iters_max": 6, "n_min": 2, "patience": 2}

LEARNER_PRESETS = {
    "muzero-emp": {"known_rules": False, "greedy_backup": False, "centered_values": False},
    "muzero-known": {"known_rules": True, "greedy_backup": True, "centered_values": True},
    "q": {"known_rules": False, "greedy_backup": False, "centered_values": False},
    "dyna-emp": {"known_rules": False, "greedy_backup": False, "centered_values": False},
    "dyna-rules": {"known_rules": False, "greedy_backup": False, "centered_values": False},
}


class _SnapshotDelegate:
    """Forwards every `Learner`-protocol call to `learner` unchanged; after
    each `end_iteration(n)`, records a copy of `learner.alpha`/`.beta` and
    prints one progress line (economy, iteration, delta, elapsed) -- the only
    point in Algorithm 3-4's loop this script can observe without touching
    simulate.py.
    """

    def __init__(self, learner, econ_name: str) -> None:
        self.learner = learner
        self.econ_name = econ_name
        self.alpha_snapshots: list[np.ndarray] = []
        self.beta_snapshots: list[np.ndarray] = []
        self._t0 = time.perf_counter()

    def observe_meeting(self, i: int, j_good: int) -> None:
        self.learner.observe_meeting(i, j_good)

    def trade_action(self, i: int, k: int, j: int, rng: np.random.Generator) -> int:
        return self.learner.trade_action(i, k, j, rng)

    def cons_action(self, i: int, h: int, rng: np.random.Generator) -> int:
        return self.learner.cons_action(i, h, rng)

    def observe_trade(self, rec) -> None:
        self.learner.observe_trade(rec)

    def observe_cons(self, rec) -> None:
        self.learner.observe_cons(rec)

    def end_iteration(self, n: int) -> float:
        delta = self.learner.end_iteration(n)
        self.alpha_snapshots.append(np.array(self.learner.alpha, copy=True))
        self.beta_snapshots.append(np.array(self.learner.beta, copy=True))
        elapsed = time.perf_counter() - self._t0
        print(f"[{self.econ_name}] iter {n:>4}  delta={delta:.5f}  elapsed={elapsed:7.1f}s")
        return delta

    def deployment_policy(self):
        return self.learner.deployment_policy()


def resolve_ablation_flags(learner: str, args: argparse.Namespace) -> dict:
    preset = LEARNER_PRESETS[learner]
    return {
        "known_rules": preset["known_rules"] if args.known_rules is None else args.known_rules,
        "greedy_backup": (
            preset["greedy_backup"] if args.greedy_backup is None else args.greedy_backup
        ),
        "centered_values": (
            preset["centered_values"] if args.centered_values is None else args.centered_values
        ),
    }


def _muzero_factory(econ, cfg, rng, flags, args) -> MuZeroLearner:
    try:
        return MuZeroLearner(
            econ,
            cfg,
            rng,
            known_rules=flags["known_rules"],
            greedy_backup=flags["greedy_backup"],
            centered_values=flags["centered_values"],
            ema_gain=args.ema_gain,
            per_agent=args.per_agent,
        )
    except NotImplementedError as exc:
        if flags["known_rules"]:
            raise NotImplementedError(f"{exc} (RulesBackend arrives with Task 14)") from exc
        raise


def _stub_factory(task: str):
    def factory(econ, cfg, rng, flags, args):
        raise NotImplementedError(f"--learner {args.learner!r} is not implemented until {task}")

    return factory


LEARNER_REGISTRY = {
    "muzero-emp": _muzero_factory,
    "muzero-known": _muzero_factory,
    "q": _stub_factory("Task 15"),
    "dyna-emp": _stub_factory("Task 16"),
    "dyna-rules": _stub_factory("Task 16"),
}


def _converged(cfg: LearnerConfig, deltas: list, n_iters: int) -> bool:
    """Reproduces simulate.py's (D1) predicate from the trace, for reporting only."""
    if n_iters < cfg.n_min + cfg.patience:
        return False
    tail = deltas[n_iters - cfg.patience : n_iters]
    return all(d <= cfg.delta_stop for d in tail)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--economy", required=True, choices=sorted(ECONOMIES))
    parser.add_argument("--learner", required=True, choices=sorted(LEARNER_REGISTRY))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--known-rules", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--greedy-backup", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--centered-values", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ema-gain", type=float, default=0.0)
    parser.add_argument("--per-agent", action="store_true")
    parser.add_argument("--eps-floor", type=float, default=None)
    parser.add_argument("--kappa0", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    econ = get_economy(args.economy)

    cfg = LearnerConfig()
    if args.dry_run:
        cfg = dataclasses.replace(cfg, **DRY_OVERRIDES)
        print(f"dry run -- shrunk smoke-test config {DRY_OVERRIDES}; full pipeline still runs.")
    if args.eps_floor is not None:
        cfg = dataclasses.replace(cfg, eps_floor=args.eps_floor)
    if args.kappa0 is not None:
        cfg = dataclasses.replace(cfg, kappa0=args.kappa0)

    flags = resolve_ablation_flags(args.learner, args)

    if args.out.exists() and any(args.out.iterdir()):
        print(f"error: --out {args.out} already exists and is non-empty; refusing to overwrite",
              file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    try:
        learner = LEARNER_REGISTRY[args.learner](econ, cfg, rng, flags, args)
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    wrapped = _SnapshotDelegate(learner, args.economy)
    t0 = time.perf_counter()
    res = run_learning(econ, wrapped, cfg, seed=args.seed)
    elapsed = time.perf_counter() - t0

    gain_series: list[float] = []
    gain_per_type_series: list[list] = []
    stat_series: list[float] = []
    for idx in range(res.n_iters):
        alpha_n = wrapped.alpha_snapshots[idx]
        beta_n = wrapped.beta_snapshots[idx]
        gap, gaps = br_gain(econ, alpha_n, beta_n, mu=None)
        stat = stationarity(econ, alpha_n, beta_n, res.trace["agg_holdings"][idx])
        gain_series.append(gap)
        gain_per_type_series.append(gaps.tolist())
        stat_series.append(stat)

    mu_emp_terminal = res.trace["agg_holdings"][-1]
    dep_gain, dep_gaps = br_gain(econ, res.alpha_star, res.beta_star, res.mu_star)
    dep_stat = stationarity(econ, res.alpha_star, res.beta_star, mu_emp_terminal)
    agg = (econ.nu[:, None] * res.mu_star).sum(axis=0)
    converged = _converged(cfg, res.trace["delta"], res.n_iters)

    config = {
        "economy": args.economy,
        "learner": args.learner,
        "seed": args.seed,
        "known_rules": flags["known_rules"],
        "greedy_backup": flags["greedy_backup"],
        "centered_values": flags["centered_values"],
        "ema_gain": args.ema_gain,
        "per_agent": args.per_agent,
        "eps_floor": cfg.eps_floor,
        "kappa0": cfg.kappa0,
        "dry_run": args.dry_run,
        "out": str(args.out),
        "learner_cfg": dataclasses.asdict(cfg),
    }
    arrays = {
        "alpha_star": res.alpha_star,
        "beta_star": res.beta_star,
        "mu_star": res.mu_star,
        "aggregate_holdings": agg,
        "br_gaps_deployment": dep_gaps,
        "holdings_final": res.holdings_final,
        "alpha_train_final": wrapped.alpha_snapshots[-1],
        "beta_train_final": wrapped.beta_snapshots[-1],
    }
    trace = {
        "agg_holdings": [a.tolist() for a in res.trace["agg_holdings"]],
        "trade_volume": res.trace["trade_volume"],
        "delta": res.trace["delta"],
        "br_gain_series": gain_series,
        "br_gain_per_type_series": gain_per_type_series,
        "stationarity_series": stat_series,
        "n_iters": res.n_iters,
        "converged": converged,
        "br_gain_deployment": dep_gain,
        "br_gain_deployment_per_type": dep_gaps.tolist(),
        "stationarity_deployment": dep_stat,
        "elapsed_seconds": elapsed,
    }
    save_run(args.out, config, arrays, trace)

    print(f"economy={args.economy} learner={args.learner} seed={args.seed}")
    print(
        f"known_rules={flags['known_rules']} greedy_backup={flags['greedy_backup']} "
        f"centered_values={flags['centered_values']} ema_gain={args.ema_gain} "
        f"per_agent={args.per_agent}"
    )
    print(f"n_iters={res.n_iters} converged={converged} elapsed={elapsed:.1f}s")
    print(f"deployment br_gain={dep_gain:.4f} per_type={np.array2string(dep_gaps, precision=4)}")
    print(f"deployment stationarity={dep_stat:.4e}")
    print(f"aggregate_holdings={np.array2string(agg, precision=4)}")
    print(f"saved to {args.out}")

    return 0 if converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
