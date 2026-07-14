#!/usr/bin/env python
"""CLI: campaign figures (one PNG per figure) from the on-disk run artifacts.

Usage:
    uv run python scripts/make_figures.py \
        --replication runs/campaign-replication \
        --stage2 runs/campaign-extensions \
        --reference runs \
        --out reports/figures

Read-only over the run artifacts (arrays.npz / trace.json); never writes under
any `runs/*`. Matplotlib Agg backend, one PNG per figure:

  holdings_<econ>.png    stationary holdings (aggregate_holdings, the good-
                         marginal of mu_star) as grouped bars: the Nash/SCE
                         benchmarks vs each learner's seed-0 mu_star.
  learning_<econ>.png    learning paths: per-snapshot BR gain (top) and
                         stationarity (bottom) over training, overlaying the
                         learners' seed-0 runs -- the learner analogue of the
                         SCE solver's b_t / d_t residual histories.
  trade_policy_C.png     Economy C final trade policy alpha as heatmaps: the
                         Nash fiat SNE vs muzero-emp seed 0 (trap band,
                         br_gain 0.59, policy structurally near Nash) and
                         seed 8 (trapped) -- the broken-fiat-links visual.
  outcome_frequency.png  campaign NASH and gain-level (br_gain<=0.05)
                         frequencies with Wilson 95% error bars.

Fail-loud: every artifact each figure needs is checked up front; if any is
absent the script prints the full missing list and exits nonzero before
plotting.

Seed choices are fixed and disclosed in the figure captions/titles: learner
mu_star and learning paths use seed 0; the C trade-policy figure contrasts the
trap-band seed-0 attractor (br_gain 0.59, policy structurally near Nash) with
the trapped seed-8 attractor (the two poles of the seed-dependent
stopping-time lottery documented in findings.md).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_tables import (  # noqa: E402
    ALL_LEARNERS,
    ECONS,
    REPL_LEARNERS,
    load_summary,
    wilson,
)

# Per-economy benchmark reference dirs (label -> dir under <reference>/<econ>/).
REF = {
    "A1": [("Nash", "nash-default"), ("SCE-autarky", "sce-autarky")],
    "A2": [("Nash-fund", "nash-default"), ("Nash-spec", "nash-spec")],
    "B": [("Nash-fund", "nash-default"), ("Nash-spec", "nash-spec")],
    "C": [("Nash", "nash-money"), ("SCE-lowtrade", "sce-lowtrade")],
    "D": [("Nash", "nash-default"), ("SCE-t1autarky", "sce-t1autarky")],
}
FIAT_IDX = {"C": 0}  # economies with a fiat good (index into the goods axis)
LEARNER_COLORS = {
    "muzero-emp": "#1f77b4", "muzero-known": "#2ca02c", "q": "#d62728",
    "dyna-emp": "#9467bd", "dyna-rules": "#ff7f0e",
}
# Benchmarks get grayscale so they never collide with the learner palette above.
REF_COLORS = {
    "Nash": "black", "Nash-fund": "black", "Nash-spec": "dimgray",
    "SCE-autarky": "silver", "SCE-lowtrade": "silver", "SCE-t1autarky": "silver",
}
C_TRADE_SEEDS = (0, 8)  # trap band (near-Nash policy) vs trapped attractor (seed lottery)


def campaign_root(replication: Path, stage2: Path, learner: str) -> Path:
    return replication if learner in REPL_LEARNERS else stage2


def run_dir(replication: Path, stage2: Path, econ: str, learner: str, seed: int) -> Path:
    return campaign_root(replication, stage2, learner) / econ / learner / str(seed)


def required_paths(args: argparse.Namespace) -> list[Path]:
    paths = []
    for econ in ECONS:
        for _, ref in REF[econ]:
            paths.append(args.reference / econ / ref / "arrays.npz")
        for learner in ALL_LEARNERS:
            d = run_dir(args.replication, args.stage2, econ, learner, 0)
            paths += [d / "arrays.npz", d / "trace.json"]
    for seed in C_TRADE_SEEDS:
        paths.append(run_dir(args.replication, args.stage2, "C", "muzero-emp", seed) / "arrays.npz")
    return paths


def fail_loud_missing(paths: list[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"error: {len(missing)} artifact(s) required for figures are absent:",
              file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        sys.exit(2)


def _nan_series(values) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values])


# --------------------------------------------------------------------------- holdings


def fig_holdings(args: argparse.Namespace, econ: str) -> Path:
    sources, values = [], []
    for label, ref in REF[econ]:
        z = np.load(args.reference / econ / ref / "arrays.npz")
        sources.append(label)
        values.append(z["aggregate_holdings"])
    for learner in ALL_LEARNERS:
        z = np.load(run_dir(args.replication, args.stage2, econ, learner, 0) / "arrays.npz")
        sources.append(learner)
        values.append(z["aggregate_holdings"])
    values = np.array(values)  # (n_sources, n_goods)
    n_src, n_goods = values.shape

    fig, ax = plt.subplots(figsize=(1.4 * n_goods + 3, 4.2))
    x = np.arange(n_goods)
    width = 0.8 / n_src
    for i, (src, row) in enumerate(zip(sources, values)):
        color = LEARNER_COLORS.get(src) or REF_COLORS.get(src)
        ax.bar(x + (i - (n_src - 1) / 2) * width, row, width, label=src, color=color)
    labels = [f"good {g}" + (" (fiat)" if g == FIAT_IDX.get(econ) else "") for g in range(n_goods)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("aggregate stationary holdings")
    ax.set_title(f"Economy {econ}: stationary holdings (benchmarks vs learner mu* @ seed 0)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = args.out / f"holdings_{econ}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- learning paths


def fig_learning(args: argparse.Namespace, econ: str) -> Path:
    fig, (ax_b, ax_d) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for learner in ALL_LEARNERS:
        trace = json.loads(
            (run_dir(args.replication, args.stage2, econ, learner, 0) / "trace.json").read_text()
        )
        b = _nan_series(trace["br_gain_series"])
        d = _nan_series(trace["stationarity_series"])
        color = LEARNER_COLORS.get(learner)
        ax_b.plot(np.arange(len(b)), b, label=learner, color=color, lw=1.3)
        ax_d.plot(np.arange(len(d)), d, label=learner, color=color, lw=1.3)
    ax_b.set_ylabel("BR gain b_t")
    ax_b.set_title(f"Economy {econ}: learning paths @ seed 0 "
                  "(learner analogue of the SCE b_t / d_t residuals)")
    ax_b.set_yscale("symlog", linthresh=1e-3)
    ax_b.legend(fontsize=7, ncol=3)
    ax_d.set_ylabel("stationarity d_t")
    ax_d.set_xlabel("training snapshot")
    ax_d.axhline(1 / np.sqrt(1800), ls="--", c="grey", lw=0.8, label="finite-N floor")
    ax_d.legend(fontsize=7)
    fig.tight_layout()
    out = args.out / f"learning_{econ}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- C trade policy


def fig_trade_policy_C(args: argparse.Namespace) -> Path:
    nash = np.load(args.reference / "C" / "nash-money" / "arrays.npz")["alpha"]  # (I,G,G)
    emp = {
        s: np.load(run_dir(args.replication, args.stage2, "C", "muzero-emp", s)
                  / "arrays.npz")["alpha_star"]
        for s in C_TRADE_SEEDS
    }
    cols = [("Nash fiat SNE", nash),
            (f"muzero-emp seed {C_TRADE_SEEDS[0]} (trap band, near-Nash policy)",
             emp[C_TRADE_SEEDS[0]]),
            (f"muzero-emp seed {C_TRADE_SEEDS[1]} (trapped)", emp[C_TRADE_SEEDS[1]])]
    n_types = nash.shape[0]
    fig, axes = plt.subplots(n_types, len(cols), figsize=(3.2 * len(cols), 3.0 * n_types))
    for j, (title, alpha) in enumerate(cols):
        for i in range(n_types):
            ax = axes[i, j]
            im = ax.imshow(alpha[i], vmin=0, vmax=1, cmap="viridis")
            for (r, c), v in np.ndenumerate(alpha[i]):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=7)
            if i == 0:
                ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"type {i}\nholding (good)", fontsize=8)
            if i == n_types - 1:
                ax.set_xlabel("offered good (0=fiat)", fontsize=8)
            ax.set_xticks(range(alpha[i].shape[1]))
            ax.set_yticks(range(alpha[i].shape[0]))
    fig.suptitle("Economy C final trade policy alpha[type](holding, offered): "
                 "broken fiat links in the trapped attractor", fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="accept probability")
    out = args.out / "trade_policy_C.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- outcome frequency


def fig_outcome_frequency(args: argparse.Namespace) -> Path:
    repl = load_summary(args.replication)
    stage2 = load_summary(args.stage2)
    ext = load_summary(args.extensions)
    groups = [("replication\n(muzero)", repl), ("Stage-2\n(q/dyna)", stage2),
              ("extensions\n(Task 18)", ext)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(groups))
    width = 0.38
    for off, (metric, pred) in enumerate([
        ("NASH", lambda r: r["classification"] == "NASH"),
        ("gain<=0.05", lambda r: r["br_gain"] is not None and r["br_gain"] <= 0.05),
    ]):
        fracs, errs = [], [[], []]
        for _, recs in groups:
            n = len(recs)
            k = sum(1 for r in recs if pred(r))
            lo, hi = wilson(k, n)
            fracs.append(k / n)
            errs[0].append(max(0.0, k / n - lo))  # clamp float dust when phat == 0
            errs[1].append(max(0.0, hi - k / n))
        ax.bar(x + (off - 0.5) * width, fracs, width, yerr=errs, capsize=4, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("fraction of runs")
    ax.set_ylim(0, 1.05)
    ax.set_title("Campaign outcomes: NASH vs gain-level frequency (Wilson 95% CI)")
    ax.legend()
    fig.tight_layout()
    out = args.out / "outcome_frequency.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replication", type=Path, default=Path("runs/campaign-replication"))
    p.add_argument("--stage2", type=Path, default=Path("runs/campaign-extensions"))
    p.add_argument("--extensions", type=Path, default=Path("runs/extensions"))
    p.add_argument("--reference", type=Path, default=Path("runs"),
                  help="root holding the per-economy benchmark dirs <econ>/<nash|sce>")
    p.add_argument("--out", type=Path, default=Path("reports/figures"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fail_loud_missing(required_paths(args))
    args.out.mkdir(parents=True, exist_ok=True)

    written = []
    for econ in ECONS:
        written.append(fig_holdings(args, econ))
        written.append(fig_learning(args, econ))
    written.append(fig_trade_policy_C(args))
    written.append(fig_outcome_frequency(args))

    print(f"wrote {len(written)} figures to {args.out}:")
    for p in written:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
