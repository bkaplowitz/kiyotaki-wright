"""Task 17 Step 1: a dry-run campaign over (A1) x (q) x 2 seeds creates 2
manifests and a summary with 2 entries; re-invocation skips completed runs
(manifests untouched, no learning work redone) and rebuilds the same summary
from the on-disk artifacts."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "run_campaign.py"


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
