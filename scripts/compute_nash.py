#!/usr/bin/env python
"""CLI: run Algorithm 1 (damped best-response RVI) to a stationary Nash
equilibrium for one economy/prompt and save the run artifact.

Usage:
    uv run python scripts/compute_nash.py --economy A1 --prompt default --out runs/A1/nash-default
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from kw.artifacts import save_run
from kw.economies import ECONOMIES, get_economy
from kw.nash_solver import solve_nash
from kw.prompts import nash_prompt

PROMPT_NAMES = ("default", "speculative", "money")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--economy", required=True, choices=sorted(ECONOMIES))
    parser.add_argument("--prompt", required=True, choices=PROMPT_NAMES)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--eps-br", type=float, default=1e-9)
    parser.add_argument("--eps-dist", type=float, default=1e-12)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    econ = get_economy(args.economy)

    config = {
        "economy": args.economy,
        "prompt": args.prompt,
        "rho": args.rho,
        "eps_br": args.eps_br,
        "eps_dist": args.eps_dist,
        "max_iter": args.max_iter,
        "out": str(args.out),
    }

    if args.dry_run:
        print("dry run — resolved config (no solve, no write):")
        for key, value in config.items():
            print(f"  {key} = {value}")
        return 0

    try:
        alpha0, beta0 = nash_prompt(econ, args.prompt)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.out.exists() and any(args.out.iterdir()):
        print(f"error: --out {args.out} already exists and is non-empty; refusing to overwrite",
              file=sys.stderr)
        return 1

    res = solve_nash(
        econ,
        alpha0,
        beta0,
        rho=args.rho,
        eps_br=args.eps_br,
        eps_dist=args.eps_dist,
        max_iter=args.max_iter,
    )
    agg = (econ.nu[:, None] * res.mu).sum(axis=0)

    manifest = {"config": config}
    arrays = {
        "alpha": res.alpha,
        "beta": res.beta,
        "mu": res.mu,
        "Q": res.Q,
        "g": res.g,
        "aggregate_holdings": agg,
    }
    trace = {
        "gain_gap_history": res.gain_gap_history,
        "dist_history": res.dist_history,
        "converged": res.converged,
        "n_iter": len(res.gain_gap_history),
    }
    save_run(args.out, manifest, arrays, trace)

    print(f"economy={args.economy} prompt={args.prompt}")
    print(f"converged={res.converged} iterations={len(res.gain_gap_history)}")
    print(f"final_gain_gap={res.gain_gap_history[-1]:.3e} final_dist={res.dist_history[-1]:.3e}")
    print(f"aggregate_holdings={np.array2string(agg, precision=4)}")
    print(f"saved to {args.out}")

    return 0 if res.converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
