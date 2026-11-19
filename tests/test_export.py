"""Results bundles: exported shapes match the live computation, and uploads are extracted safely."""
import zipfile

import numpy as np
import pytest

from physprobe.export import export_bundle, safe_extract
from physprobe.gui import data as D
from physprobe.selftest import make_demo_cache


def test_bundle_round_trip(tmp_path):
    make_demo_cache(tmp_path / "runs" / "demo" / "acts")
    z = export_bundle(tmp_path / "runs", tmp_path / "bundle", zip_it=True, progress=lambda *_: None)
    root = safe_extract(z, tmp_path / "unzipped")
    f = D.discover(root)
    entry = next(c for c in f.caches if c["kind"] == "bundle")
    b = D.BundleSet(entry["path"])
    assert "direction_deg" in b.shape_labels()
    live = D.manifold_trajectory(D.load_cache(tmp_path / "runs" / "demo" / "acts"), "direction_deg", with_videos=True)
    got = b.trajectory("direction_deg")
    assert np.allclose(got["scores"], live["scores"], atol=1e-5)
    assert np.allclose(got["videos"], live["videos"], atol=1e-2 * np.abs(live["videos"]).max())
    assert len(b.metrics("direction_deg")) == 24


def test_safe_extract_rejects_traversal(tmp_path):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("bundle.json", '{"format": "physprobe-bundle"}')
        zf.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="unsafe path"):
        safe_extract(z, tmp_path / "out")


def test_safe_extract_requires_bundle(tmp_path):
    z = tmp_path / "plain.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("notes.txt", "x")
    with pytest.raises(ValueError, match="no bundle.json"):
        safe_extract(z, tmp_path / "out")
