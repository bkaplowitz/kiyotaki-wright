"""Task 17 Step 1: a dry-run campaign over (A1) x (q) x 2 seeds creates 2
manifests and a summary with 2 entries; re-invocation skips completed runs
(manifests untouched, no learning work redone) and rebuilds the same summary
from the on-disk artifacts.

Fix round 1 adds fast, unmarked tests (no subprocesses; they call
run_campaign.main directly) for the trace.json completion marker -- a cell
dir with manifest.json but no trace.json fails fast naming the dir; a cell
with both is skipped -- and for the rejection of forwarded args that collide
with the campaign-managed per-run flags."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "run_campaign.py"

sys.path.insert(0, str(REPO / "scripts"))
import run_campaign  # noqa: E402


def _invoke(out: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, str(SCRIPT),
        "--economies", "A1", "--learners", "q", "--seeds", "2",
        "--jobs", "2", "--out", str(out), "--dry-run",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)


@pytest.mark.slow
def test_dry_run_campaign_creates_artifacts_and_resumes(tmp_path):
    out = tmp_path / "campaign"
    first = _invoke(out)
    assert first.returncode == 0, first.stdout + first.stderr

    manifests = [out / "A1" / "q" / str(seed) / "manifest.json" for seed in (0, 1)]
    assert all(m.exists() for m in manifests)

    summary_text = (out / "summary.json").read_text()
    assert "NaN" not in summary_text  # NaN is not valid JSON; missing values are null
    summary = json.loads(summary_text)
    assert len(summary) == 2
    assert {(r["economy"], r["learner"], r["seed"]) for r in summary} == {
        ("A1", "q", 0),
        ("A1", "q", 1),
    }
    for rec in summary:
        assert {"br_gain", "stationarity", "classification", "n_iters", "wall_clock"} <= set(rec)
        assert rec["n_iters"] > 0
        assert rec["wall_clock"] is None or rec["wall_clock"] >= 0

    mtimes = [m.stat().st_mtime for m in manifests]
    t0 = time.perf_counter()
    second = _invoke(out)
    resume_elapsed = time.perf_counter() - t0
    assert second.returncode == 0, second.stdout + second.stderr
    assert [m.stat().st_mtime for m in manifests] == mtimes  # completed runs were skipped
    assert second.stdout.count("[skip") == 2
    assert resume_elapsed < 60  # wall-clock ~ 0 for completed: no learning work redone
    assert json.loads((out / "summary.json").read_text()) == summary


def _fake_cell(out: Path, econ: str, learner: str, seed: int, with_trace: bool) -> Path:
    run_dir = out / econ / learner / str(seed)
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


def _main(out: Path, *extra: str) -> int:
    return run_campaign.main(
        ["--economies", "A1", "--learners", "q", "--seeds", "1", "--out", str(out), *extra]
    )


def test_manifest_without_trace_fails_fast_naming_dir(tmp_path, capsys):
    run_dir = _fake_cell(tmp_path, "A1", "q", 0, with_trace=False)
    assert _main(tmp_path) == 1
    err = capsys.readouterr().err
    assert str(run_dir) in err
    assert "trace.json" in err
    assert (run_dir / "manifest.json").exists()  # nothing was deleted


def test_completed_cell_with_both_files_is_skipped(tmp_path, capsys):
    _fake_cell(tmp_path, "A1", "q", 0, with_trace=True)
    cells = [("A1", "q", 0)]
    assert run_campaign._scan_cells(tmp_path, cells) == ([], cells, [])
    assert _main(tmp_path) == 0
    assert "[skip ] A1/q/0" in capsys.readouterr().out
    assert json.loads((tmp_path / "summary.json").read_text()) == [
        {
            "economy": "A1",
            "learner": "q",
            "seed": 0,
            "br_gain": 0.01,
            "stationarity": 1e-4,
            "classification": "NASH",
            "n_iters": 5,
            "wall_clock": 1.0,
        }
    ]


def test_forwarded_reserved_flag_rejected(tmp_path, capsys):
    assert _main(tmp_path, "--seed", "3") == 1
    assert "--seed" in capsys.readouterr().err
    assert not (tmp_path / "A1").exists()  # nothing was launched
