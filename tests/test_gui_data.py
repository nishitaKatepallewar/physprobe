"""The explorer's data layer (no browser needed): discovery, shape trajectories and metrics
on the simulated demo cache, where a ring is planted from about one-third depth."""
import numpy as np

from physprobe.gui import data as D
from physprobe.selftest import make_demo_cache


def test_explorer_data(tmp_path):
    make_demo_cache(tmp_path / "runs" / "demo" / "acts")
    f = D.discover(tmp_path / "runs", tmp_path / "data")
    assert [c["name"] for c in f.caches if not c["missing"]] == ["demo/acts"]
    c = D.load_cache(f.caches[0]["path"])
    assert "direction_deg" in D.label_columns(c.labels)
    assert D.is_circular(c.labels, "direction_deg", c.targets)
    tr = D.manifold_trajectory(c, "direction_deg", None, 3, with_videos=True)
    assert tr["scores"].shape == (24, 8, 3) and tr["videos"].shape == (24, 392, 3)
    m = D.manifold_metrics(c, "direction_deg", None, circular=True)
    late = m[m.layer >= 12]
    assert (late["reliability"] > 0.3).all() and (late["ring order"] > 0.9).all()
    assert late["harmonic 1"].median() > 0.9  # single layers can be noisy (8 split-halves)


def test_figures_build(tmp_path):
    from physprobe.gui import figures as F
    make_demo_cache(tmp_path / "acts")
    c = D.load_cache(tmp_path / "acts")
    tr = D.manifold_trajectory(c, "direction_deg", None, 3, with_videos=True)
    for dims in (2, 3):
        assert F.manifold_view(tr, 8, True, dims, True, True, "direction_deg").data
        assert len(F.manifold_animation(tr, True, dims).frames) == 24
    assert F.filmstrip(tr, True, current=8).data
    s = D.quick_probe(c, ["direction_deg:circular", "speed:scalar"])
    assert F.probe_curves({"demo": s}, ["direction_deg", "speed"], 8) is not None


def test_shape_views_with_three_levels():
    """A variable with three values has only two principal axes; 3-D views fall back to 2-D."""
    from physprobe.gui import figures as F
    L = 4
    traj = {"scores": np.random.default_rng(0).normal(size=(L, 3, 2)), "var": np.full((L, 2), 0.5),
            "levels": np.array([2.0, 4.0, 6.0]), "videos": None, "video_levels": None}
    F.manifold_view(traj, 1, False, dims=3)
    F.manifold_animation(traj, False, dims=3)
