#!/usr/bin/env python
"""CLI: run Algorithm 2 (restricted-information SCE fixed point) for one
economy/prompt and save the run artifact, including the complete-information
best-response gain — the H2 diagnostic separating an SCE from a Nash point.

Usage:
    uv run python scripts/compute_sce.py --economy A1 --prompt skeptical-autarky --out runs/A1/sce-autarky
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from kw.artifacts import save_run
from kw.diagnostics import br_gain
from kw.economies import ECONOMIES, get_economy
from kw.prompts import sce_prompt
from kw.sce_solver import solve_sce

PROMPT_NAMES = ("skeptical-autarky", "skeptical-lowtrade", "type1-autarky")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--economy", required=True, choices=sorted(ECONOMIES))
    parser.add_argument("--prompt", required=True, choices=PROMPT_NAMES)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--eps-sce", type=float, default=1e-9)
    parser.add_argument("--eps-dist", type=float, default=1e-12)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    econ = get_economy(args.economy)

    config = {
        "economy": args.economy,
        "prompt": args.prompt,
        "rho": args.rho,
        "eps_sce": args.eps_sce,
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
        alpha0, beta0, m0, q0 = sce_prompt(econ, args.prompt)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.out.exists() and any(args.out.iterdir()):
        print(f"error: --out {args.out} already exists and is non-empty; refusing to overwrite",
              file=sys.stderr)
        return 1

    res = solve_sce(
        econ,
        alpha0,
        beta0,
        m0,
        q0,
        rho=args.rho,
        eps_sce=args.eps_sce,
        eps_dist=args.eps_dist,
        max_iter=args.max_iter,
    )
    agg = (econ.nu[:, None] * res.mu).sum(axis=0)
    gap, gaps = br_gain(econ, res.alpha, res.beta, res.mu)

    manifest = {"config": config}
    arrays = {
        "alpha": res.alpha,
        "beta": res.beta,
        "mu": res.mu,
        "m": res.m,
        "q": res.q,
        "aggregate_holdings": agg,
        "br_gaps": gaps,
    }
    trace = {
        "s_history": res.s_history,
        "b_history": res.b_history,
        "d_history": res.d_history,
        "converged": res.converged,
        "n_iter": len(res.s_history),
        "br_gain_max": gap,
    }
    save_run(args.out, manifest, arrays, trace)

    print(f"economy={args.economy} prompt={args.prompt}")
    print(f"converged={res.converged} iterations={len(res.s_history)}")
    print(
        f"final_s={res.s_history[-1]:.3e} final_b={res.b_history[-1]:.3e} "
        f"final_d={res.d_history[-1]:.3e}"
    )
    print(f"aggregate_holdings={np.array2string(agg, precision=4)}")
    print(f"br_gain_max={gap:.4f} per_type={np.array2string(gaps, precision=4)}")
    print(f"saved to {args.out}")

    return 0 if res.converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
