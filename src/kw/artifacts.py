"""Run artifact IO: `manifest.json` + `arrays.npz` + `trace.json` per run directory.

One directory per run. `save_run` auto-adds provenance (git SHA, ISO timestamp,
Python/NumPy versions) to the caller-supplied manifest so every artifact is
traceable to the code that produced it (architecture.md §11).
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_run(path: Path, manifest: dict, arrays: dict[str, np.ndarray], trace: dict) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    full_manifest = dict(manifest)
    full_manifest["git_sha"] = _git_sha()
    full_manifest["timestamp"] = datetime.now(timezone.utc).isoformat()
    full_manifest["python_version"] = sys.version.split()[0]
    full_manifest["numpy_version"] = np.__version__

    (path / "manifest.json").write_text(json.dumps(full_manifest, indent=2, sort_keys=True))
    np.savez_compressed(path / "arrays.npz", **arrays)
    (path / "trace.json").write_text(json.dumps(trace, indent=2))


def load_run(path: Path) -> tuple[dict, dict[str, np.ndarray], dict]:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    with np.load(path / "arrays.npz") as npz:
        arrays = {key: npz[key] for key in npz.files}
    trace = json.loads((path / "trace.json").read_text())
    return manifest, arrays, trace
