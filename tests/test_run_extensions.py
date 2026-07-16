"""Task 18: scripts/run_extensions.py runs the fixed ablation/E1/E2/E3/decay
grids (extensions/architecture.md sections 3, 5, 6 + the Task 18 decay
addendum) as a thin loop over run_learning subprocesses with run_campaign's
artifact and completion semantics.

Fast tests only, mirroring tests/test_campaign.py's post-fix pattern: the
grid is inspected via `build_cells`, and resume/fail-fast/summary behavior is
exercised by calling `run_extensions.main` directly against fabricated cell
dirs in tmp_path -- no subprocesses, no learning work.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from kw.economies import get_economy
from kw.learners.base import LearnerConfig

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import run_extensions  # noqa: E402
import run_learning  # noqa: E402

DECAY_CELLS = {
    ("B", "q"), ("B", "dyna-emp"), ("C", "q"), ("C", "dyna-rules"),
    ("A2", "dyna-rules"), ("D", "q"), ("D", "dyna-emp"), ("D", "dyna-rules"),
}


def test_full_grid_counts_and_contents():
    cells = run_extensions.build_cells(dry_run=False)
    assert Counter(c.arm for c in cells) == {
        "ablation": 16, "e1": 30, "e2": 15, "e3": 36, "decay": 72,
    }
    assert len(cells) == 169
    assert len({(c.arm, c.econ, c.label, c.seed) for c in cells}) == 169  # unique run dirs

    abl = {(c.econ, c.label): c for c in cells if c.arm == "ablation"}
    assert {e for e, _ in abl} == {"C", "D"}
    cell = abl[("C", "kr1-gb0-cv1")]
    assert cell.learner == "muzero-emp" and cell.seed == 0
    assert list(cell.flags) == ["--known-rules", "--no-greedy-backup", "--centered-values"]

    e1 = [c for c in cells if c.arm == "e1"]
    assert {c.econ for c in e1} == {"C", "D"} and {c.seed for c in e1} == set(range(5))
    assert {c.label for c in e1} == {"ema0.005", "ema0.02", "ema0.05"}
    some = next(c for c in e1 if c.label == "ema0.02")
    assert some.learner == "muzero-emp" and list(some.flags) == ["--ema-gain", "0.02"]

    e2 = [c for c in cells if c.arm == "e2"]
    assert {c.econ for c in e2} == {"A1", "C", "D"} and {c.seed for c in e2} == set(range(5))
    assert all(
        c.learner == "muzero-emp" and c.label == "per-agent" and list(c.flags) == ["--per-agent"]
        for c in e2
    )

    e3 = [c for c in cells if c.arm == "e3"]
    assert {c.econ for c in e3} == {"C"} and {c.seed for c in e3} == set(range(3))
    assert len({c.label for c in e3}) == 12
    picked = next(c for c in e3 if c.label == "eps0.1-k8")
    assert list(picked.flags) == ["--eps-floor", "0.1", "--kappa0", "8"]

    dec = [c for c in cells if c.arm == "decay"]
    assert {c.seed for c in dec} == set(range(3))
    assert {(c.econ, c.learner) for c in dec} == DECAY_CELLS
    both = next(c for c in dec if c.econ == "A2" and c.label == "dyna-rules-eps-lr")
    assert list(both.flags) == ["--eps-start", "0.5", "--q-lr-visit"]
    eps_only = next(c for c in dec if c.econ == "B" and c.label == "q-eps")
    assert list(eps_only.flags) == ["--eps-start", "0.5"]
    lr_only = next(c for c in dec if c.econ == "D" and c.label == "dyna-emp-lr")
    assert list(lr_only.flags) == ["--q-lr-visit"]


def test_dry_run_grid_shrinks_multi_seed_arms_to_seed_zero():
    cells = run_extensions.build_cells(dry_run=True)
    assert Counter(c.arm for c in cells) == {
        "ablation": 16, "e1": 6, "e2": 3, "e3": 12, "decay": 24,
    }
    assert len(cells) == 61
    assert {c.seed for c in cells} == {0}
    full = {(c.arm, c.econ, c.label, c.seed) for c in run_extensions.build_cells(dry_run=False)}
    assert {(c.arm, c.econ, c.label, c.seed) for c in cells} <= full


def _fake_cell(out: Path, cell, with_trace: bool) -> Path:
    run_dir = out / cell.arm / cell.econ / cell.label / str(cell.seed)
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}")
    if with_trace:
        trace = {
            "br_gain_deployment": 0.01,
            "stationarity_deployment": 1e-4,
            "n_iters": 5,
            "elapsed_seconds": 1.0,
        }
        (run_dir / "trace.json").write_text(json.dumps(trace))
    return run_dir


def test_incomplete_dir_fails_fast_naming_dir(tmp_path, capsys):
    cells = run_extensions.build_cells(dry_run=True)
    bad = _fake_cell(tmp_path, cells[0], with_trace=False)
    assert run_extensions.main(["--out", str(tmp_path), "--dry-run"]) == 1
    err = capsys.readouterr().err
    assert str(bad) in err
    assert "trace.json" in err
    assert (bad / "manifest.json").exists()  # nothing was deleted


def test_completed_cells_skipped_and_summarized(tmp_path, capsys):
    cells = run_extensions.build_cells(dry_run=True)
    for cell in cells:
        _fake_cell(tmp_path, cell, with_trace=True)
    assert run_extensions.main(["--out", str(tmp_path), "--dry-run"]) == 0
    assert capsys.readouterr().out.count("[skip ]") == len(cells)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary) == len(cells)
    rec = next(r for r in summary if r["arm"] == "e2" and r["economy"] == "A1")
    assert rec == {
        "arm": "e2",
        "economy": "A1",
        "learner": "muzero-emp",
        "cell": "per-agent",
        "seed": 0,
        "br_gain": 0.01,
        "stationarity": 1e-4,
        "classification": "NASH",
        "n_iters": 5,
        "wall_clock": 1.0,
    }


def test_unknown_arg_rejected(tmp_path):
    with pytest.raises(SystemExit):
        run_extensions.main(["--out", str(tmp_path), "--seed", "3"])
    assert not any(tmp_path.iterdir())  # nothing was launched


def test_run_learning_decay_flag_plumbing(tmp_path):
    """The decay arms consume run_learning's --eps-start/--q-lr-visit: parse_args
    must surface both and the q/dyna factories must hand them to QLearner."""
    args = run_learning.parse_args(
        ["--economy", "B", "--learner", "q", "--out", str(tmp_path),
         "--eps-start", "0.5", "--q-lr-visit"]
    )
    assert args.eps_start == 0.5 and args.q_lr_visit is True
    flags = run_learning.resolve_ablation_flags("q", args)
    lr = run_learning.LEARNER_REGISTRY["q"](
        get_economy("B"), LearnerConfig(), np.random.default_rng(0), flags, args
    )
    assert lr.eps == 0.5
    assert lr.q_lr_visit is True and lr._qt_visits is not None
