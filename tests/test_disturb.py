import numpy as np
import pandas as pd
import torch

from physprobe import disturb


def test_condition_shifts_wraps_angles():
    X = np.eye(4)
    vals = np.array([0, 90, 180, 350])
    delta, near = disturb.condition_shifts(X, vals, 5)
    assert near == 0  # 350 and 0 are 10 and 5 degrees from 5
    np.testing.assert_allclose(delta[3], X[0] - X[3])
    np.testing.assert_allclose(delta[0], 0)


def test_controls_keep_norm():
    rng = np.random.default_rng(0)
    d = rng.standard_normal((5, 32))
    for kind in ("permuted", "random"):
        c = disturb.controls_for(d, kind, rng)
        np.testing.assert_allclose(np.linalg.norm(c, axis=1), np.linalg.norm(d, axis=1))
    p = disturb.controls_for(d, "permuted", rng)
    np.testing.assert_allclose(np.sort(p, 1), np.sort(d, 1))


def test_matched_ids():
    lab = pd.DataFrame({"direction_deg": [0, 90, 0, 90], "speed": [1, 1, 2, 2], "start_id": [0, 0, 0, 0]})
    m = disturb.matched_ids(lab, "direction_deg", 90, ["speed", "start_id"])
    assert list(m) == [1, 1, 3, 3]


class _Cache:
    def __init__(self, X, labels):
        self.X, self.labels, self.meta = X, labels, {"frames": 16}

    def layer(self, l):
        return self.X


class _DS:
    def get(self, i, frames):
        return np.array([i])


class _Enc:
    """Planted model: the predicted future is the projection of the layer activation on
    one axis u, which is also the axis that separates the condition means. A shift
    along u (the informative steer) changes the prediction; a same-norm permuted
    control mostly does not."""
    device, tubelet = "cpu", 2

    def __init__(self, X, u):
        self.X, self.u = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(u, dtype=torch.float32)

    def predict(self, videos, context, edits=None):
        ids = torch.as_tensor(np.asarray(videos)[:, 0])
        h = self.X[ids][:, None, :]
        for fn in (edits or {}).values():
            h = fn(h)
        s = h[:, 0] @ self.u
        n, n_ctx = 8, context
        tokens = (self.X[ids] @ self.u)[:, None, None].repeat(1, n, 1)
        pred = s[:, None, None].repeat(1, n - n_ctx, 1) + 0.5  # imperfect predictor
        return {"tokens": tokens, "pred": pred, "target_ids": torch.arange(n_ctx, n)}


def test_planted_importance_and_pull():
    rng = np.random.default_rng(0)
    d = 64
    u = np.zeros(d)
    u[0] = 1.0
    vals = np.repeat([0, 90, 180, 270], 6)
    labels = pd.DataFrame({"direction_deg": vals, "speed": np.tile(np.arange(6), 4), "start_id": 0})
    X = rng.standard_normal((len(vals), d)) * 0.1
    X[:, 0] += np.cos(np.deg2rad(vals)) * 3  # the condition means differ along u
    ids = np.where(vals != 90)[0]
    df = disturb.run(_Enc(X, u), _DS(), _Cache(X, labels), 0, "direction_deg", 90, ids, context=4,
                     n_controls=3, batch_size=5, target_norm=False)
    s = disturb.summarize(df).set_index("metric")
    assert s.loc["l2", "importance_vs_permuted"] > 1.0
    assert s.loc["l2", "importance_vs_random"] > 1.0
    assert s.loc["l2", "pull_vs_permuted"] > 0.5  # the prediction moves to the target future
    np.testing.assert_allclose(df[df.condition == "unperturbed"].l2, 0.25, atol=1e-6)


def test_disturb_command_on_tiny_model(tmp_path):
    from physprobe.cli import main
    main(["make", "velocity", "--out", str(tmp_path / "data"), "--grid", "direction_deg=0,90,180,270",
          "speed=2", "--n-starts", "2"])
    main(["extract", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "acts"), "--random-init",
          "--tiny", "--res", "64", "--device", "cpu"])
    main(["disturb", "--acts", str(tmp_path / "acts"), "--out", str(tmp_path / "d"), "--tiny",
          "--device", "cpu", "--n", "4", "--controls", "1", "--layers", "1", "--scales", "1,8"])
    s = pd.read_csv(tmp_path / "d" / "disturb_summary.csv")
    assert set(s.metric) == {"l2", "l1"} and set(s.scale) == {1.0, 8.0} and s.videos.iloc[0] == 4
    assert np.isfinite(s.importance_vs_permuted).all() and np.isfinite(s.pull_vs_random).all()
    rows = pd.read_csv(tmp_path / "d" / "disturb_L1.csv")
    assert set(rows.kind) == {"unperturbed", "steer", "permuted", "random"}
