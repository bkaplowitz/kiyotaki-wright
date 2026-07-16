#!/usr/bin/env python
"""CLI: the Task 18 extension grids as a thin loop over run_learning subprocesses.

Usage:
    uv run python scripts/run_extensions.py --jobs 8 --out runs/extensions

The grids are fixed (extensions/architecture.md sections 3, 5, 6 + the Task 18
decay addendum) -- this script takes no per-cell knobs, only --jobs/--out/--dry-run:

  - ablation: 2^3 over --known-rules/--greedy-backup/--centered-values x {C, D},
    seed 0, muzero-emp base (the explicit flags override the preset).
  - e1:  --ema-gain in {0.005, 0.02, 0.05} x {C, D} x seeds 0-4, muzero-emp.
  - e2:  --per-agent x {A1, C, D} x seeds 0-4, muzero-emp.
  - e3:  --eps-floor in {0.01, 0.03, 0.1, 0.2} x --kappa0 in {0.5, 2, 8} on C,
    seeds 0-2, muzero-emp.
  - decay (addendum A): --eps-start 0.5 / --q-lr-visit / both, over the stuck
    Stage-2 cells (B,q) (B,dyna-emp) (C,q) (C,dyna-rules) (A2,dyna-rules)
    (D,q) (D,dyna-emp) (D,dyna-rules), seeds 0-2.

Each cell invokes `scripts/run_learning.py --economy E --learner L --seed S
--out <out>/<arm>/<econ>/<cell>/<seed>` plus the cell's flags, with
stdout/stderr in the sibling `<seed>.log` -- run_campaign.py's layout with one
extra <arm>/<cell> level, and its exact completion semantics (imported from
run_campaign, the single source of truth): trace.json is the completion
marker, completed cells are skipped on re-invocation, a non-empty cell dir
without trace.json fails fast naming the dir, and exit codes 0/2 are both
completed runs. `--dry-run` forwards --dry-run to run_learning and shrinks
every multi-seed arm to seed 0.

`<out>/summary.json` is rebuilt from the on-disk artifacts after every
invocation: one record per cell -- {arm, economy, learner, cell, seed,
br_gain, stationarity, classification, n_iters, wall_clock}, NaN as null.
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_campaign import (  # noqa: E402
    OK_RETURNCODES,
    RUN_LEARNING,
    classify,
    none_if_nan,
)

EMA_GAINS = ("0.005", "0.02", "0.05")
EPS_FLOORS = ("0.01", "0.03", "0.1", "0.2")
KAPPA0S = ("0.5", "2", "8")
DECAY_CELLS = (
    ("B", "q"), ("B", "dyna-emp"), ("C", "q"), ("C", "dyna-rules"),
    ("A2", "dyna-rules"), ("D", "q"), ("D", "dyna-emp"), ("D", "dyna-rules"),
)
DECAY_ARMS = (
    ("eps", ("--eps-start", "0.5")),
    ("lr", ("--q-lr-visit",)),
    ("eps-lr", ("--eps-start", "0.5", "--q-lr-visit")),
)


class Cell(NamedTuple):
    arm: str
    econ: str
    learner: str
    label: str
    seed: int
    flags: tuple[str, ...]


def build_cells(dry_run: bool) -> list[Cell]:
    """The full 169-cell grid, or its 61-cell seed-0 shrink under --dry-run."""

    def seeds(n: int) -> range:
        return range(1 if dry_run else n)

    cells: list[Cell] = []
    for econ, (kr, gb, cv) in itertools.product(("C", "D"), itertools.product((0, 1), repeat=3)):
        flags = (
            "--known-rules" if kr else "--no-known-rules",
            "--greedy-backup" if gb else "--no-greedy-backup",
            "--centered-values" if cv else "--no-centered-values",
        )
        cells.append(Cell("ablation", econ, "muzero-emp", f"kr{kr}-gb{gb}-cv{cv}", 0, flags))
    for gain, econ, seed in itertools.product(EMA_GAINS, ("C", "D"), seeds(5)):
        cells.append(Cell("e1", econ, "muzero-emp", f"ema{gain}", seed, ("--ema-gain", gain)))
    for econ, seed in itertools.product(("A1", "C", "D"), seeds(5)):
        cells.append(Cell("e2", econ, "muzero-emp", "per-agent", seed, ("--per-agent",)))
    for eps, k0, seed in itertools.product(EPS_FLOORS, KAPPA0S, seeds(3)):
        flags = ("--eps-floor", eps, "--kappa0", k0)
        cells.append(Cell("e3", "C", "muzero-emp", f"eps{eps}-k{k0}", seed, flags))
    for (econ, learner), (arm_tag, flags), seed in itertools.product(
        DECAY_CELLS, DECAY_ARMS, seeds(3)
    ):
        cells.append(Cell("decay", econ, learner, f"{learner}-{arm_tag}", seed, flags))
    return cells


def _cell_dir(out: Path, cell: Cell) -> Path:
    return out / cell.arm / cell.econ / cell.label / str(cell.seed)


def _scan_cells(out: Path, cells: list[Cell]) -> tuple[list[Cell], list[Cell], list[Path]]:
    """Partition into (todo, skipped, incomplete) by run_campaign's trace.json marker."""
    todo, skipped, incomplete = [], [], []
    for cell in cells:
        run_dir = _cell_dir(out, cell)
        if (run_dir / "trace.json").exists():
            skipped.append(cell)
        elif run_dir.exists() and any(run_dir.iterdir()):
            incomplete.append(run_dir)
        else:
            todo.append(cell)
    return todo, skipped, incomplete


def _run_one(out: Path, cell: Cell, dry_run: bool) -> None:
    run_dir = _cell_dir(out, cell)
    log_path = run_dir.parent / f"{cell.seed}.log"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(RUN_LEARNING),
        "--economy", cell.econ, "--learner", cell.learner, "--seed", str(cell.seed),
        "--out", str(run_dir), *cell.flags,
    ]
    if dry_run:
        cmd.append("--dry-run")
    name = f"{cell.arm}/{cell.econ}/{cell.label}/{cell.seed}"
    print(f"[start] {name}", flush=True)
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    elapsed = time.perf_counter() - t0
    if rc not in OK_RETURNCODES or not (run_dir / "trace.json").exists():
        raise RuntimeError(f"{name} exited {rc}; see {log_path}")
    print(f"[done ] {name} rc={rc} elapsed={elapsed:.1f}s", flush=True)


def _summarize(out: Path, cells: list[Cell]) -> list[dict]:
    records = []
    for cell in cells:
        trace = json.loads((_cell_dir(out, cell) / "trace.json").read_text())
        records.append(
            {
                "arm": cell.arm,
                "economy": cell.econ,
                "learner": cell.learner,
                "cell": cell.label,
                "seed": cell.seed,
                "br_gain": none_if_nan(trace["br_gain_deployment"]),
                "stationarity": none_if_nan(trace["stationarity_deployment"]),
                "classification": classify(trace),
                "n_iters": trace["n_iters"],
                "wall_clock": none_if_nan(trace.get("elapsed_seconds")),
            }
        )
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--jobs", type=int, default=1, help="joblib n_jobs")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="shrink every multi-seed arm to seed 0 and forward --dry-run to run_learning",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cells = build_cells(args.dry_run)
    if args.dry_run:
        print("dry run -- seed 0 only per arm; forwarding --dry-run to run_learning.")

    todo, skipped, incomplete = _scan_cells(args.out, cells)
    for cell in skipped:
        print(f"[skip ] {cell.arm}/{cell.econ}/{cell.label}/{cell.seed} (trace.json present)",
              flush=True)
    if incomplete:
        print(
            "error: non-empty run dir(s) without trace.json (crashed/interrupted runs);\n"
            "not touching them -- move them away (e.g. `trash <dir>`) and re-run:",
            file=sys.stderr,
        )
        for run_dir in incomplete:
            print(f"  {run_dir}", file=sys.stderr)
        return 1

    print(f"extensions: {len(cells)} cells ({len(todo)} to run, {len(skipped)} skipped), "
          f"jobs={args.jobs}, out={args.out}", flush=True)

    t0 = time.perf_counter()
    Parallel(n_jobs=args.jobs)(delayed(_run_one)(args.out, cell, args.dry_run) for cell in todo)
    elapsed = time.perf_counter() - t0

    records = _summarize(args.out, cells)
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(records, indent=2, allow_nan=False))
    print(f"ran {len(todo)}, skipped {len(skipped)}, total {elapsed:.1f}s")
    print(f"summary ({len(records)} records) -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
