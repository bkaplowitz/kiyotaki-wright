#!/usr/bin/env python
"""CLI: multi-seed learning campaign over (economy, learner, seed) via joblib.

Usage:
    uv run python scripts/run_campaign.py --economies A1,A2,B,C,D \
        --learners muzero-emp,muzero-known --seeds 10 --jobs 8 \
        --out runs/campaign-replication

`--seeds N` runs seeds 0..N-1. Any flag this parser does not recognize is
forwarded verbatim to `scripts/run_learning.py` (the single source of truth
for one run), which each cell invokes as a subprocess with `--economy
--learner --seed --out <out>/<econ>/<learner>/<seed>/`; that subprocess's
stdout/stderr goes to the sibling `<out>/<econ>/<learner>/<seed>.log`.
run_learning exits 0 (converged) or 2 (completed, (D1) not reached) -- both
are completed runs here; any other exit code aborts the campaign.

Resume: a cell whose `manifest.json` exists is skipped (run_learning itself
refuses a non-empty --out, so completed work is never overwritten). A cell
directory *without* a manifest is a crashed/interrupted run: the campaign
fails fast, names the dirs, and leaves removal to you (use `trash`).

`<out>/summary.json` is rebuilt from the on-disk artifacts after every
invocation -- a resumed campaign yields a complete summary. One record per
cell: {economy, learner, seed, br_gain, stationarity, classification,
n_iters, wall_clock}, with NaN serialized as null (NaN is not valid JSON).
`classification` is not stored in trace.json, so it is derived from the
stored deployment diagnostics with `classify_run`'s no-beliefs semantics
(kw.diagnostics): NASH iff br_gain <= Thresholds.nash_gain and stationarity
<= Thresholds.dist, else UNSETTLED (SCE_TRAP needs beliefs, which run
artifacts do not store).

`--dry-run` shrinks to 2 seeds and forwards `--dry-run` to run_learning
(its documented shrunk smoke-test config; the full pipeline still runs).
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from joblib import Parallel, delayed

from kw.diagnostics import Thresholds
from kw.economies import ECONOMIES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_learning import LEARNER_REGISTRY  # noqa: E402

RUN_LEARNING = Path(__file__).resolve().parent / "run_learning.py"
OK_RETURNCODES = (0, 2)  # run_learning: 0 = converged, 2 = completed but (D1) not reached


def _cell_dir(out: Path, econ: str, learner: str, seed: int) -> Path:
    return out / econ / learner / str(seed)


def _run_one(out: Path, econ: str, learner: str, seed: int, forwarded: list[str]) -> None:
    run_dir = _cell_dir(out, econ, learner, seed)
    log_path = run_dir.parent / f"{seed}.log"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(RUN_LEARNING),
        "--economy", econ, "--learner", learner, "--seed", str(seed),
        "--out", str(run_dir), *forwarded,
    ]
    print(f"[start] {econ}/{learner}/{seed}", flush=True)
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    elapsed = time.perf_counter() - t0
    # trace.json is save_run's last write; its absence means the run did not
    # complete (argparse errors also exit 2, so rc alone cannot tell).
    if rc not in OK_RETURNCODES or not (run_dir / "trace.json").exists():
        raise RuntimeError(f"{econ}/{learner}/{seed} exited {rc}; see {log_path}")
    print(f"[done ] {econ}/{learner}/{seed} rc={rc} elapsed={elapsed:.1f}s", flush=True)


def _none_if_nan(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def _classification(trace: dict) -> str | None:
    if trace.get("classification") is not None:
        return trace["classification"]
    b = _none_if_nan(trace.get("br_gain_deployment"))
    d = _none_if_nan(trace.get("stationarity_deployment"))
    if b is None or d is None:
        return None
    thr = Thresholds()
    return "NASH" if (b <= thr.nash_gain and d <= thr.dist) else "UNSETTLED"


def _summarize(out: Path, cells: list[tuple[str, str, int]]) -> list[dict]:
    records = []
    for econ, learner, seed in cells:
        trace = json.loads((_cell_dir(out, econ, learner, seed) / "trace.json").read_text())
        records.append(
            {
                "economy": econ,
                "learner": learner,
                "seed": seed,
                "br_gain": _none_if_nan(trace["br_gain_deployment"]),
                "stationarity": _none_if_nan(trace["stationarity_deployment"]),
                "classification": _classification(trace),
                "n_iters": trace["n_iters"],
                "wall_clock": _none_if_nan(trace.get("elapsed_seconds")),
            }
        )
    return records


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--economies", required=True, help="comma-separated, e.g. A1,A2,B,C,D")
    parser.add_argument("--learners", required=True, help="comma-separated, e.g. q,dyna-emp")
    parser.add_argument("--seeds", required=True, type=int, help="number of seeds; runs 0..N-1")
    parser.add_argument("--jobs", type=int, default=1, help="joblib n_jobs")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="shrink to 2 seeds and forward --dry-run to run_learning",
    )
    args, forwarded = parser.parse_known_args(argv)
    return args, forwarded


def main(argv: list[str] | None = None) -> int:
    args, forwarded = parse_args(argv)

    economies = args.economies.split(",")
    learners = args.learners.split(",")
    bad_econ = [e for e in economies if e not in ECONOMIES]
    bad_learner = [name for name in learners if name not in LEARNER_REGISTRY]
    if bad_econ or bad_learner:
        for e in bad_econ:
            print(f"error: unknown economy {e!r} (have {sorted(ECONOMIES)})", file=sys.stderr)
        for name in bad_learner:
            print(
                f"error: unknown learner {name!r} (have {sorted(LEARNER_REGISTRY)})",
                file=sys.stderr,
            )
        return 1

    n_seeds = 2 if args.dry_run else args.seeds
    if args.dry_run:
        forwarded = [*forwarded, "--dry-run"]
        print("dry run -- 2 seeds; forwarding --dry-run to run_learning.")

    cells = [(e, name, s) for e in economies for name in learners for s in range(n_seeds)]

    todo, incomplete = [], []
    for econ, learner, seed in cells:
        run_dir = _cell_dir(args.out, econ, learner, seed)
        if (run_dir / "manifest.json").exists():
            print(f"[skip ] {econ}/{learner}/{seed} (manifest present)", flush=True)
        elif run_dir.exists():
            incomplete.append(run_dir)
        else:
            todo.append((econ, learner, seed))
    if incomplete:
        print(
            "error: run dir(s) exist without manifest.json (crashed/interrupted runs);\n"
            "not touching them -- move them away (e.g. `trash <dir>`) and re-run:",
            file=sys.stderr,
        )
        for run_dir in incomplete:
            print(f"  {run_dir}", file=sys.stderr)
        return 1

    n_skipped = len(cells) - len(todo)
    print(f"campaign: {len(cells)} cells ({len(todo)} to run, {n_skipped} skipped), "
          f"jobs={args.jobs}, out={args.out}", flush=True)
    if forwarded:
        print(f"forwarding to run_learning: {forwarded}", flush=True)

    t0 = time.perf_counter()
    Parallel(n_jobs=args.jobs)(
        delayed(_run_one)(args.out, econ, learner, seed, forwarded)
        for econ, learner, seed in todo
    )
    elapsed = time.perf_counter() - t0

    records = _summarize(args.out, cells)
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(records, indent=2, allow_nan=False))
    print(f"ran {len(todo)}, skipped {n_skipped}, total {elapsed:.1f}s")
    print(f"summary ({len(records)} records) -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
