import numpy as np

from kw.artifacts import load_run, save_run


def test_round_trip_manifest_and_arrays(tmp_path):
    out = tmp_path / "A1" / "nash-default"
    manifest = {"config": {"economy": "A1", "prompt": "default", "rho": 0.3}}
    arrays = {"alpha": np.arange(8.0).reshape(2, 2, 2), "mu": np.eye(2)}
    trace = {"gain_gap_history": [1.0, 0.5, 0.0], "converged": True}

    save_run(out, manifest, arrays, trace)
    loaded_manifest, loaded_arrays, loaded_trace = load_run(out)

    assert loaded_manifest["config"] == manifest["config"]
    assert "git_sha" in loaded_manifest and loaded_manifest["git_sha"]
    assert "timestamp" in loaded_manifest
    assert "python_version" in loaded_manifest
    assert "numpy_version" in loaded_manifest
    assert np.allclose(loaded_arrays["alpha"], arrays["alpha"])
    assert np.allclose(loaded_arrays["mu"], arrays["mu"])
    assert loaded_trace == trace


def test_save_run_writes_expected_files(tmp_path):
    out = tmp_path / "run"
    save_run(out, {"config": {}}, {"x": np.array([1.0, 2.0])}, {"h": [1, 2, 3]})

    assert (out / "manifest.json").exists()
    assert (out / "arrays.npz").exists()
    assert (out / "trace.json").exists()


def test_save_run_creates_parent_directories(tmp_path):
    out = tmp_path / "nested" / "dirs" / "run"
    save_run(out, {"config": {}}, {"x": np.array([1.0])}, {})
    manifest, arrays, trace = load_run(out)
    assert np.allclose(arrays["x"], [1.0])
