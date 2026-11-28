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


def test_ring_shift_moves_along_the_fitted_ring():
    rng = np.random.default_rng(0)
    ang = np.repeat(np.arange(0, 360, 45), 5).astype(float)
    th = np.deg2rad(ang)
    a, b = rng.standard_normal(16), rng.standard_normal(16)
    resid = rng.standard_normal((len(ang), 16)) * 0.01
    X = np.cos(th)[:, None] * a + np.sin(th)[:, None] * b + resid
    D = disturb.ring_shifts(X, ang, 90.0, harmonics=1)
    moved = X + D
    np.testing.assert_allclose(moved - resid, np.tile(b, (len(ang), 1)), atol=0.02)  # all at 90 deg
    np.testing.assert_allclose(D[ang == 90], 0, atol=0.02)


def test_verdict_rule():
    row = {"layer": 8, "scale": 0.4, "metric": "l2", "reader_err_deg": 20.0,
           "pull_vs_permuted": 0.03, "pull_vs_permuted_se": 0.01,
           "pull_vs_shuffled": 0.01, "pull_vs_shuffled_se": 0.01}
    v = disturb.verdict(pd.DataFrame([row]))
    assert v.verdict.iloc[0] == "FAIL"  # does not beat the shuffled-label control
    row["pull_vs_shuffled"] = 0.025
    assert disturb.verdict(pd.DataFrame([row])).verdict.iloc[0] == "PASS"
    row["reader_err_deg"] = 70.0
    assert disturb.verdict(pd.DataFrame([row])).verdict.iloc[0].startswith("NO VERDICT")


def test_ring_from_another_set():
    rng = np.random.default_rng(1)
    ang = np.repeat(np.arange(0, 360, 45), 4).astype(float)
    th = np.deg2rad(ang)
    a, b = rng.standard_normal(12), rng.standard_normal(12)
    X_src = np.cos(th)[:, None] * a + np.sin(th)[:, None] * b
    X = X_src * 1.0 + rng.standard_normal(X_src.shape) * 0.01
    D_self = disturb.ring_shifts(X, ang, 0.0, 1)
    D_src = disturb.ring_shifts(X, ang, 0.0, 1, source=(X_src, ang))
    np.testing.assert_allclose(D_self, D_src, atol=0.05)


def test_recheck_readers():
    rng = np.random.default_rng(2)
    rows, P = [], []
    for v, ang in enumerate(np.repeat(np.arange(0, 360, 45), 3)):
        for kind in ("unperturbed", "steer", "permuted"):
            shown = 90 if kind == "steer" else ang
            P.append(np.r_[np.cos(np.deg2rad(shown)), np.sin(np.deg2rad(shown)), rng.normal(0, .05, 6)])
            rows.append({"video": v, "direction_deg": ang, "kind": kind, "scale": 0.0 if kind == "unperturbed" else 0.4})
    out = disturb.recheck(pd.DataFrame(rows), np.array(P), "direction_deg", 90.0, k=3)
    for _, r in out.iterrows():
        assert r.reader_err_deg < 20 and r.err_to_target_steer < 10 and r.err_to_target_permuted > 60


def test_flip_gap():
    rows = []
    for v in range(4):
        rows += [{"video": v, "kind": "unperturbed", "scale": 0.0, "l2": 1.0, "l2_to_target": 3.0},
                 {"video": v, "kind": "steer", "scale": 0.4, "l2": 2.5, "l2_to_target": 1.5},
                 {"video": v, "kind": "permuted", "scale": 0.4, "l2": 1.2, "l2_to_target": 3.0}]
    r = disturb.flip(pd.DataFrame(rows)).iloc[0]
    assert r.flip_unperturbed == 0 and r.flip_steer == 1 and r.flip_permuted == 0
    assert abs(r.gap_closed_steer - 1.5) < 1e-9 and abs(r.gap_closed_permuted - 0.1) < 1e-9


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
    df, _ = disturb.run(_Enc(X, u), _DS(), _Cache(X, labels), 0, "direction_deg", 90, ids, context=4,
                     n_controls=3, batch_size=5, target_norm=False)
    s = disturb.summarize(df).set_index("metric")
    assert s.loc["l2", "importance_vs_permuted"] > 1.0
    assert s.loc["l2", "importance_vs_random"] > 1.0
    assert s.loc["l2", "pull_vs_permuted"] > 0.5  # the prediction moves to the target future
    np.testing.assert_allclose(df[df.condition == "unperturbed"].l2, 0.25, atol=1e-6)


def test_disturb_command_on_tiny_model(tmp_path):
    from physprobe.cli import main
    main(["make", "velocity", "--out", str(tmp_path / "data"), "--grid", "direction_deg=0,90,180,270",
          "speed=2", "--n-starts", "3"])
    main(["extract", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "acts"), "--random-init",
          "--tiny", "--res", "64", "--device", "cpu"])
    main(["disturb", "--acts", str(tmp_path / "acts"), "--out", str(tmp_path / "d"), "--tiny",
          "--device", "cpu", "--n", "8", "--controls", "1", "--layers", "1", "--scales", "1,8"])
    s = pd.read_csv(tmp_path / "d" / "disturb_summary.csv")
    assert set(s.metric) == {"l2", "l1"} and set(s.scale) == {1.0, 8.0} and s.videos.iloc[0] == 8
    assert np.isfinite(s.importance_vs_permuted).all() and np.isfinite(s.pull_vs_random).all()
    rows = pd.read_csv(tmp_path / "d" / "disturb_L1.csv")
    assert set(rows.kind) == {"unperturbed", "steer", "permuted", "random", "shuffled"}
    st = rows[rows.kind == "steer"].set_index(["video", "scale"]).shift_rel
    sh = rows[rows.kind == "shuffled"].groupby(["video", "scale"]).shift_rel.mean()
    np.testing.assert_allclose(sh.reindex(st.index), st, rtol=1e-4)  # same size as the steer
    v = pd.read_csv(tmp_path / "d" / "disturb_verdict.csv")
    assert set(v.verdict) <= {"PASS", "FAIL", "NO VERDICT (reader too weak)"}
    assert "pred_dir_deg" in rows.columns
    main(["disturb", "--acts", str(tmp_path / "acts"), "--out", str(tmp_path / "r"), "--tiny",
          "--device", "cpu", "--n", "8", "--controls", "1", "--layers", "1", "--method", "ring",
          "--sizes", "0.3"])
    r = pd.read_csv(tmp_path / "r" / "disturb_L1.csv")
    np.testing.assert_allclose(r[r.kind == "steer"].shift_rel, 0.3, rtol=1e-4)
    assert np.load(tmp_path / "r" / "disturb_L1_predictions.npz")["pooled"].shape[0] == len(r)
    assert "pred_speed_change" not in r.columns or r.pred_speed_change.notna().any()
    main(["disturb-recheck", "--run", str(tmp_path / "r")])
    assert (tmp_path / "r" / "disturb_recheck.csv").exists()
    assert (tmp_path / "r" / "disturb_flip.csv").exists()
    main(["disturb", "--acts", str(tmp_path / "acts"), "--out", str(tmp_path / "x"), "--tiny",
          "--device", "cpu", "--n", "8", "--controls", "1", "--layers", "1", "--method", "ring",
          "--sizes", "0.3", "--ring-from", str(tmp_path / "acts")])
    main(["disturb-figure", "--run", str(tmp_path / "d"), "--data", str(tmp_path / "data"), "--layer", "1",
          "--scale", "8", "--examples", "3"])
    assert (tmp_path / "d" / "disturb_L1_x8.png").stat().st_size > 10_000
    # explorer: the run is discovered, exported and drawn
    from physprobe.export import export_bundle
    from physprobe.gui import data as D
    from physprobe.gui import figures as F
    found = D.discover(tmp_path, tmp_path / "data")
    assert [e["name"] for e in found.steering] == ["d", "r", "x"]
    bundle = export_bundle(tmp_path, tmp_path.parent / f"{tmp_path.name}_bundle", data=tmp_path / "data",
                           progress=lambda *_: None)
    res = D.read_disturb(D.discover(bundle).steering[0]["path"])
    assert (1, 8.0) in res["figures"] and set(res["summary"].scale) == {1.0, 8.0}
    assert len(F.steering_doses(res["summary"], 1).data) >= 4
