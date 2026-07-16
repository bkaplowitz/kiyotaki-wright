#!/usr/bin/env python
"""CLI: campaign result tables (T1-T4) from the on-disk campaign summaries.

Usage:
    uv run python scripts/make_tables.py \
        --replication runs/campaign-replication \
        --stage2 runs/campaign-extensions \
        --extensions runs/extensions \
        --out reports/

Read-only over the three campaign roots' `summary.json` artifacts (never the
run dirs themselves, never writing under any `runs/*`). Emits one markdown
report `reports/tables.md` with four tables:

  T1  Table-2 analog: median (IQR) BR gain and stationarity per economy x
      learner over seeds, alongside the paper's single-run reference values
      (architecture.md section 15), with an explicit replication/extension
      divider row inside each economy block.
  T2  Outcome-frequency table with Wilson 95% CIs -- reported at BOTH the
      classification level (NASH: br_gain <= 0.05 AND stationarity <= 1e-3)
      and the gain level (br_gain <= 0.05 only), because the finite-N
      stationarity floor (~1/sqrt(1800) ~ 0.024), not distance from Nash,
      is what blocks most near-Nash cells from classifying.
  T3  Mann-Whitney U (two-sided) p-values on deployment BR gain between every
      learner pair within each economy. Degenerate all-identical inputs (e.g.
      all-zero learner columns) are reported "n/a (identical)" rather than
      crashing scipy.
  T4  Extension tables: the 2^3 ablation (seed 0 only -- marked), E1 (EMA),
      E2 (per-agent), E3 (exploration sweep), and the decay addendum.

Fail-loud: before building any table the script enumerates the *expected* run
set for each campaign (5x2x10 replication, 5x3x10 Stage-2, and the 169-cell
extension grid via run_extensions.build_cells) and exits nonzero listing every
missing record if any expected artifact is absent.

Statistics use scipy.stats.mannwhitneyu; the Wilson interval is the closed-form
score interval (statsmodels is not a dependency). Aggregation is nan-aware
(numpy nan-reducers) although the campaign summaries currently carry no nulls.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_extensions import build_cells  # noqa: E402

from kw.diagnostics import Thresholds  # noqa: E402

ECONS = ("A1", "A2", "B", "C", "D")
REPL_LEARNERS = ("muzero-emp", "muzero-known")
STAGE2_LEARNERS = ("q", "dyna-emp", "dyna-rules")
ALL_LEARNERS = REPL_LEARNERS + STAGE2_LEARNERS
N_AGENTS = 1800  # population size -> finite-N stationarity floor 1/sqrt(N) ~ 0.024
Z95 = 1.959963984540054  # standard normal 0.975 quantile

# Paper single-run reference targets (architecture.md section 15). (br_gain, stationary_dist);
# the paper reports a stationary distance only for economies C and D.
PAPER = {
    ("A1", "muzero-emp"): (1.50e-12, None),
    ("A1", "muzero-known"): (0.0, None),
    ("A2", "muzero-emp"): (21.3, None),
    ("A2", "muzero-known"): (0.0, None),
    ("B", "muzero-emp"): (0.0, None),
    ("B", "muzero-known"): (0.0, None),
    ("C", "muzero-emp"): (12.8, 0.130),
    ("C", "muzero-known"): (0.0, 1.74e-8),
    ("D", "muzero-emp"): (1.71, 0.173),
    ("D", "muzero-known"): (7.11e-15, 8.69e-9),
}


def load_summary(root: Path) -> list[dict]:
    path = root / "summary.json"
    if not path.exists():
        sys.exit(f"error: missing campaign summary {path}")
    return json.loads(path.read_text())


def check_grid(records: list[dict], econs, learners, seeds: int, label: str) -> list[str]:
    """Return the expected (econ, learner, seed) keys absent from a flat campaign."""
    present = {(r["economy"], r["learner"], r["seed"]) for r in records}
    missing = []
    for econ in econs:
        for learner in learners:
            for seed in range(seeds):
                if (econ, learner, seed) not in present:
                    missing.append(f"{label}: {econ}/{learner}/seed{seed}")
    return missing


def check_extensions(records: list[dict]) -> list[str]:
    """Return the expected 169 extension cells (run_extensions.build_cells) absent from disk."""
    present = {(r["arm"], r["economy"], r["cell"], r["seed"]) for r in records}
    missing = []
    for cell in build_cells(dry_run=False):
        key = (cell.arm, cell.econ, cell.label, cell.seed)
        if key not in present:
            missing.append(f"extensions: {cell.arm}/{cell.econ}/{cell.label}/seed{cell.seed}")
    return missing


def _arr(records: list[dict], field: str) -> np.ndarray:
    """Field as a float array, null -> nan (nan-aware reducers downstream)."""
    return np.array([np.nan if r[field] is None else float(r[field]) for r in records])


def agg(values: np.ndarray) -> tuple[int, float, float, float]:
    """(n_finite, median, q1, q3), nan-aware; all-nan -> nan stats."""
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return 0, np.nan, np.nan, np.nan
    q1, med, q3 = np.percentile(finite, [25, 50, 75])
    return finite.size, med, q1, q3


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Closed-form Wilson score interval for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return center - half, center + half


def sig(x: float, n: int = 3) -> str:
    """Round to n significant figures, nan-safe, avoiding scientific dust for readable magnitudes."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if x == 0:
        return "0"
    from math import floor, log10

    digits = n - 1 - floor(log10(abs(x)))
    r = round(x, digits)
    if 1e-4 <= abs(r) < 1e5:
        s = f"{r:.{max(digits, 0)}f}" if digits > 0 else f"{r:.0f}"
        return s
    return f"{r:.{n - 1}e}"


def pct(x: float) -> str:
    return "--" if np.isnan(x) else f"{100 * x:.1f}%"


# --------------------------------------------------------------------------- T1


def table1(repl: list[dict], stage2: list[dict]) -> str:
    lines = [
        "## T1 -- Table-2 analog: BR gain and stationarity, per economy x learner",
        "",
        "Median (IQR) over seeds of the deployment best-response gain and stationarity "
        "(population one-step residual). `NASH n/N` counts seeds classified NASH "
        "(br_gain <= 0.05 and stationarity <= 1e-3). Paper columns are the single-run "
        "reference values (architecture.md section 15); the paper reports a stationary "
        "distance only for C and D. A Nash gain is exactly 0 by construction.",
        "",
        "| Economy | Stage | Learner | seeds | BR gain median [IQR] | Stationarity median "
        "[IQR] | NASH n/N | Paper BR gain | Paper stat |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    by_key = {}
    for stage, recs in (("replication", repl), ("extension", stage2)):
        for econ in ECONS:
            for learner in (REPL_LEARNERS if stage == "replication" else STAGE2_LEARNERS):
                cell = [r for r in recs if r["economy"] == econ and r["learner"] == learner]
                by_key[(econ, stage, learner)] = cell

    for econ in ECONS:
        # replication learners, then an explicit divider, then extension learners
        for learner in REPL_LEARNERS:
            lines.append(_t1_row(econ, "replication", learner, by_key))
        lines.append(f"| {econ} | | *— extension baselines —* | | | | | | |")
        for learner in STAGE2_LEARNERS:
            lines.append(_t1_row(econ, "extension", learner, by_key))
    lines.append("")
    return "\n".join(lines)


def _t1_row(econ: str, stage: str, learner: str, by_key: dict) -> str:
    cell = by_key[(econ, stage, learner)]
    n_b, med_b, q1_b, q3_b = agg(_arr(cell, "br_gain"))
    n_s, med_s, q1_s, q3_s = agg(_arr(cell, "stationarity"))
    n_nash = sum(1 for r in cell if r["classification"] == "NASH")
    br = f"{sig(med_b)} [{sig(q1_b)}, {sig(q3_b)}]"
    st = f"{sig(med_s)} [{sig(q1_s)}, {sig(q3_s)}]"
    paper = PAPER.get((econ, learner))
    p_br = sig(paper[0]) if paper else "--"
    p_st = sig(paper[1]) if (paper and paper[1] is not None) else "--"
    return (
        f"| {econ} | {stage} | {learner} | {n_b} | {br} | {st} | "
        f"{n_nash}/{len(cell)} | {p_br} | {p_st} |"
    )


# --------------------------------------------------------------------------- T2


def _freq_cell(recs: list[dict], predicate) -> str:
    n = len(recs)
    k = sum(1 for r in recs if predicate(r))
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {pct(k / n) if n else '--'} [{pct(lo)}, {pct(hi)}]"


def table2(repl: list[dict], stage2: list[dict], ext: list[dict]) -> str:
    thr = Thresholds()
    is_nash = lambda r: r["classification"] == "NASH"  # noqa: E731
    is_gain = lambda r: r["br_gain"] is not None and r["br_gain"] <= thr.nash_gain  # noqa: E731
    lines = [
        "## T2 -- Outcome frequencies with Wilson 95% CIs",
        "",
        f"Two readings per group. **Classification-level NASH**: br_gain <= {thr.nash_gain} "
        f"AND stationarity <= {thr.dist}. **Gain-level**: br_gain <= {thr.nash_gain} only "
        "(ignores stationarity). The gap between them is the finite-N stationarity floor "
        f"(~1/sqrt({N_AGENTS}) ~ {1 / np.sqrt(N_AGENTS):.3f} empirical-histogram noise): many "
        "cells reach Nash-distance yet stay UNSETTLED on the stationarity residual alone.",
        "",
        "### Campaign totals",
        "",
        "| Campaign | n | NASH [Wilson 95%] | gain<=0.05 [Wilson 95%] |",
        "|---|---|---|---|",
    ]
    for label, recs in (("replication (muzero)", repl), ("Stage-2 (q/dyna)", stage2),
                        ("extensions (Task 18)", ext)):
        lines.append(
            f"| {label} | {len(recs)} | {_freq_cell(recs, is_nash)} | {_freq_cell(recs, is_gain)} |"
        )
    lines += [
        "",
        "### Per economy x learner (replication + Stage-2)",
        "",
        "| Economy | Learner | n | NASH [Wilson 95%] | gain<=0.05 [Wilson 95%] |",
        "|---|---|---|---|---|",
    ]
    pool = repl + stage2
    for econ in ECONS:
        for learner in ALL_LEARNERS:
            cell = [r for r in pool if r["economy"] == econ and r["learner"] == learner]
            if not cell:
                continue
            lines.append(
                f"| {econ} | {learner} | {len(cell)} | {_freq_cell(cell, is_nash)} | "
                f"{_freq_cell(cell, is_gain)} |"
            )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- T3


def _mwu(x: np.ndarray, y: np.ndarray) -> str:
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return "n/a (empty)"
    if np.ptp(np.concatenate([x, y])) == 0:
        return "n/a (identical)"
    try:
        u, p = mannwhitneyu(x, y, alternative="two-sided")
    except ValueError as exc:
        return f"n/a ({exc})"
    return f"{p:.3g}"


def table3(repl: list[dict], stage2: list[dict]) -> str:
    lines = [
        "## T3 -- Mann-Whitney U (two-sided) on BR gain, learner pairs within each economy",
        "",
        "p-values from scipy.stats.mannwhitneyu on the per-seed deployment BR gain "
        "distributions (n=10 vs n=10). Pairs whose pooled values are all identical "
        "(e.g. two all-zero learner columns) are 'n/a (identical)'. Small p means the two "
        "learners' gain distributions differ.",
        "",
    ]
    pool = repl + stage2
    for econ in ECONS:
        cols = {lr: _arr([r for r in pool if r["economy"] == econ and r["learner"] == lr],
                        "br_gain") for lr in ALL_LEARNERS}
        lines.append(f"### Economy {econ}")
        lines.append("")
        lines.append("| pair | median A | median B | MWU p |")
        lines.append("|---|---|---|---|")
        for i, a in enumerate(ALL_LEARNERS):
            for b in ALL_LEARNERS[i + 1:]:
                xa, xb = cols[a], cols[b]
                ma = np.nanmedian(xa) if np.any(~np.isnan(xa)) else np.nan
                mb = np.nanmedian(xb) if np.any(~np.isnan(xb)) else np.nan
                lines.append(f"| {a} vs {b} | {sig(ma)} | {sig(mb)} | {_mwu(xa, xb)} |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- T4


def _arm(ext: list[dict], arm: str) -> list[dict]:
    return [r for r in ext if r["arm"] == arm]


def _decay_converged(ext_root: Path, r: dict) -> bool:
    """Read the trace.json (D1) `converged` flag (not stored in summary.json)."""
    trace = json.loads(
        (ext_root / "decay" / r["economy"] / r["cell"] / str(r["seed"]) / "trace.json").read_text()
    )
    return bool(trace.get("converged"))


def table4(ext: list[dict], ext_root: Path) -> str:
    lines = ["## T4 -- Extension grids (Task 18)", ""]

    # ablation: 2^3 x {C,D}, seed 0 only
    lines += [
        "### 2^3 known-environment ablation (seed 0 only -- single run per cell)",
        "",
        "Flags kr=known-rules, gb=greedy-backup, cv=centered-values on muzero-emp. "
        "**Single-seed** cells: no IQR.",
        "",
        "| Economy | kr-gb-cv | BR gain | Stationarity | Class |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(_arm(ext, "ablation"), key=lambda r: (r["economy"], r["cell"])):
        lines.append(
            f"| {r['economy']} | {r['cell']} | {sig(r['br_gain'])} | "
            f"{sig(r['stationarity'])} | {r['classification']} |"
        )

    # e1: EMA gains x {C,D}, 5 seeds
    lines += [
        "",
        "### E1 -- EMA / constant-gain empirical model (5 seeds/cell)",
        "",
        "| Economy | ema-gain | seeds | BR gain median [IQR] | NASH n/N |",
        "|---|---|---|---|---|",
    ]
    for r in _agg_rows(_arm(ext, "e1"), keys=("economy", "cell")):
        lines.append(r)

    # e2: per-agent x {A1,C,D}, 5 seeds
    lines += [
        "",
        "### E2 -- per-agent vs population-shared tables (5 seeds/cell)",
        "",
        "**Caveat:** e2 artifacts are the MEAN of the per-agent pi_dep(1) policies, not the "
        "policy of a mean model; individual agents' tables were not persisted, so per-agent "
        "trap status is not testable post-hoc.",
        "",
        "| Economy | seeds | BR gain median [IQR] | NASH n/N |",
        "|---|---|---|---|",
    ]
    for r in _agg_rows(_arm(ext, "e2"), keys=("economy",)):
        lines.append(r)

    # e3: eps x kappa0 on C, 3 seeds
    lines += [
        "",
        "### E3 -- exploration sweep on Economy C (3 seeds/cell)",
        "",
        "| Cell (eps-floor, kappa0) | seeds | BR gain median [IQR] | NASH n/N |",
        "|---|---|---|---|",
    ]
    for r in _agg_rows(_arm(ext, "e3"), keys=("cell",), econ_col=False):
        lines.append(r)

    # decay: 8 stuck cells x 3 arms, 3 seeds
    lines += [
        "",
        "### Decay addendum -- eps-decay / lr-decay on stuck Stage-2 cells (3 seeds/cell)",
        "",
        "Arm suffix: `-eps` = eps-start 0.5 decay, `-lr` = 1/visit Q learning rate, "
        "`-eps-lr` = both. `(D1) conv` counts seeds whose training loop hit the (D1) "
        "convergence flag (trace.json `converged`) -- the first such convergences ever "
        "recorded for a Stage-2 learner (0/150 in the Stage-2 campaign).",
        "",
        "| Economy | cell | seeds | BR gain median [IQR] | NASH n/N | (D1) conv n/N |",
        "|---|---|---|---|---|---|",
    ]
    decay = _arm(ext, "decay")
    groups = {}
    for r in decay:
        groups.setdefault((r["economy"], r["cell"]), []).append(r)
    for (econ, cell), recs in sorted(groups.items()):
        n, med, q1, q3 = agg(_arr(recs, "br_gain"))
        n_nash = sum(1 for r in recs if r["classification"] == "NASH")
        n_conv = sum(1 for r in recs if _decay_converged(ext_root, r))
        lines.append(
            f"| {econ} | {cell} | {len(recs)} | {sig(med)} [{sig(q1)}, {sig(q3)}] | "
            f"{n_nash}/{len(recs)} | {n_conv}/{len(recs)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _agg_rows(recs: list[dict], keys: tuple[str, ...], econ_col: bool = True) -> list[str]:
    groups = {}
    for r in recs:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, group in sorted(groups.items()):
        n, med, q1, q3 = agg(_arr(group, "br_gain"))
        n_nash = sum(1 for r in group if r["classification"] == "NASH")
        label = " | ".join(str(k) for k in key)
        out.append(
            f"| {label} | {len(group)} | {sig(med)} [{sig(q1)}, {sig(q3)}] | {n_nash}/{len(group)} |"
        )
    return out


# --------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replication", type=Path, default=Path("runs/campaign-replication"))
    p.add_argument("--stage2", type=Path, default=Path("runs/campaign-extensions"))
    p.add_argument("--extensions", type=Path, default=Path("runs/extensions"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repl = load_summary(args.replication)
    stage2 = load_summary(args.stage2)
    ext = load_summary(args.extensions)

    missing = (
        check_grid(repl, ECONS, REPL_LEARNERS, 10, "replication")
        + check_grid(stage2, ECONS, STAGE2_LEARNERS, 10, "stage2")
        + check_extensions(ext)
    )
    if missing:
        print(f"error: {len(missing)} expected run record(s) absent from the summaries:",
              file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 2

    report = "\n".join([
        "# Campaign result tables",
        "",
        f"Generated by `scripts/make_tables.py` from {args.replication}/summary.json "
        f"({len(repl)} records), {args.stage2}/summary.json ({len(stage2)}), and "
        f"{args.extensions}/summary.json ({len(ext)}). Read-only over the campaign roots.",
        "",
        table1(repl, stage2),
        table2(repl, stage2, ext),
        table3(repl, stage2),
        table4(ext, args.extensions),
    ])
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "tables.md"
    out_path.write_text(report)
    print(f"wrote {out_path} (T1-T4, {len(repl) + len(stage2) + len(ext)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
