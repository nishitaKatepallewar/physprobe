"""Self-test: does the analysis code recover structure we planted ourselves, and does
the whole pipeline run on a model? No pretrained weights or downloads needed.

Part A plants known geometry in fake activations and checks each analysis against
the known answer. Part B pushes a tiny randomly initialised V-JEPA 2 through every
CLI command. `--full` adds a full-size ViT-L forward pass.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok)))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def planted_labels():
    return pd.DataFrame([(a, s, p) for a in range(0, 360, 45) for s in range(1, 8) for p in range(7)],
                        columns=["direction_deg", "speed", "start_id"])


def planted_ring(lab, d=256, gain=3.0, aniso=1.0, noise=1.0, seed=0):
    """Direction lives in exactly TWO dimensions (a planar ring); speed in one; noise
    has a power-law spectrum in a random basis."""
    r = np.random.default_rng(seed)
    th = np.deg2rad(lab.direction_deg.to_numpy())
    U = np.linalg.qr(r.standard_normal((d, d)))[0]
    spec = np.arange(1, d + 1) ** -aniso
    eps = noise * (r.standard_normal((len(lab), d)) * spec / np.sqrt((spec ** 2).mean())) @ U.T
    S = np.linalg.qr(r.standard_normal((d, 3)))[0]
    X = gain * (np.cos(th)[:, None] * S[:, 0] + np.sin(th)[:, None] * S[:, 1]) \
        + 0.6 * (lab.speed.to_numpy() - 4)[:, None] * S[:, 2] + eps
    return X.astype(np.float32), S


def planted_population(lab, units=240, kappa=2.0, seed=0):
    """Units with bell-shaped direction tuning whose preferred directions tile the circle."""
    r = np.random.default_rng(seed)
    th = np.deg2rad(lab.direction_deg.to_numpy())
    pref = np.linspace(0, 2 * np.pi, units, endpoint=False)
    X = np.exp(kappa * (np.cos(th[:, None] - pref[None]) - 1)) + 0.15 * r.standard_normal((len(lab), units))
    return X.astype(np.float32), np.rad2deg(pref)


def part_a():
    from . import manifold as mf
    from . import probes, subspace, tuning
    print("\nA. planted structure")
    lab = planted_labels()
    X, S = planted_ring(lab)
    t_dir = probes.make_target(lab, "direction_deg", "circular")
    t_spd = probes.make_target(lab, "speed", "scalar")
    groups = probes.resolve_groups(lab, "start_id")

    for solver in ("ridge", "adamw"):
        df = probes.cv_probe(X, t_dir, groups, solver=solver)
        check(f"direction probe ({solver}) decodes a planted ring", df.r2.mean() > 0.9,
              f"R2 {df.r2.mean():.3f}, MAE {df.circ_mae_deg.mean():.1f} deg")
    df = probes.cv_probe(X, t_spd, groups, solver="adamw")
    check("speed probe decodes planted speed", df.r2.mean() > 0.8, f"R2 {df.r2.mean():.3f}")
    r = np.random.default_rng(1)
    shuf = probes.Target("shuffled", "circular", t_dir.Y[r.permutation(len(lab))], t_dir.angle)
    df = probes.cv_probe(X, shuf, groups, solver="ridge")
    check("probe is at chance on shuffled labels", df.r2.mean() < 0.1, f"R2 {df.r2.mean():.3f}")
    yb = (lab.direction_deg < 180).astype(int)
    df = probes.cv_probe(X, probes.make_target(lab.assign(up=yb), "up", "binary"), groups, solver="adamw")
    check("binary probe works", df.accuracy.mean() > 0.9, f"acc {df.accuracy.mean():.3f}")

    # geometry of condition means (milder noise spectrum: these estimators are
    # unbiased, but their sampling error grows when noise piles into a few axes)
    X_hard = X
    X, _ = planted_ring(lab, aniso=0.5, noise=0.3)
    keys, M, _ = mf.condition_means(X, lab, ["direction_deg"])
    h = mf.harmonics_1d_cv(X, lab, "direction_deg", "start_id")
    cv = mf.cv_spectrum(X, lab, ["direction_deg"], "start_id")
    check("planar ring: noise-corrected dimensionality is 2", cv["dims90"] == 2 and cv["pr"] < 2.5,
          f"dims90 {cv['dims90']}, participation ratio {cv['pr']:.2f}")
    P = mf.pca(M, 4)
    rf = mf.ring_fit(P["scores"][:, :2], keys.direction_deg.to_numpy(float))
    check("planar ring: variance sits in harmonic 1", h[0] > 0.9, f"harmonic 1 = {h[0]:.2f}")
    check("planar ring: PC1-2 form an ordered circle", rf["circ_corr"] > 0.95 and rf["radius_cv"] < 0.25,
          f"circ_corr {rf['circ_corr']:.2f}, radius_cv {rf['radius_cv']:.2f}")
    vp = mf.variance_partition(X, lab, ["direction_deg", "speed", "start_id"], "start_id")
    tot = float(((X - X.mean(0)) ** 2).sum() / len(X))
    want = {"direction_deg": 9.0 / tot, "speed": 0.36 * 4.0 / tot}  # planted signal variances
    ok = all(abs(vp[k] - w) < 0.2 * w for k, w in want.items()) and vp["start_id"] < 0.004 \
        and max(v for k, v in vp.items() if " x " in k) < 0.01
    check("variance partition recovers the planted shares and ignores start position", ok,
          ", ".join(f"{k} {v:.4f}" for k, v in vp.items() if k != "residual")
          + f" | planted: direction {want['direction_deg']:.4f}, speed {want['speed']:.4f}")

    # subspaces (back to the strongly anisotropic noise)
    X = X_hard
    perm = np.random.default_rng(0).permutation(len(lab))
    va, tr = perm[:78], perm[78:]
    sub = lambda t, i: probes.Target(t.name, t.kind, t.Y[i], None if t.angle is None else t.angle[i])
    sd = subspace.orthogonal_probe_sequence(X[tr], sub(t_dir, tr), X[va], sub(t_dir, va))
    ss = subspace.orthogonal_probe_sequence(X[tr], sub(t_spd, tr), X[va], sub(t_spd, va))
    check("orthogonal probe sequence terminates and finds the ring", 1 <= sd.n_probes < 60,
          f"{sd.n_probes} probes = {sd.dim} dims for a ring planted in 2 dims")
    Q = sd.Q
    check("probe-sequence basis is orthonormal", np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-6))
    cap = np.linalg.norm(Q.T @ S[:, :2]) ** 2 / 2
    check("probe-sequence subspace contains the planted plane", cap > 0.8, f"captures {cap:.2f} of it")
    Z = sd.scaler(X).astype(np.float64)
    er = subspace.erasure_check(Z[tr], t_dir.Y[tr], Z[va], t_dir.Y[va], "circular")
    check("rank-2 covariance erasure removes direction", er["before"] > 0.9 and er["after"] < 0.1,
          f"R2 {er['before']:.2f} -> {er['after']:.2f} after removing 2 dims")
    check("speed probe sequence finds the speed axis", ss.n_probes >= 1, f"{ss.n_probes} probes")
    ov = subspace.overlap(sd.Q, ss.Q)
    check("direction and speed subspaces (planted orthogonal) are near 90 deg",
          ov["min_angle_deg"] > 60, f"min principal angle {ov['min_angle_deg']:.0f} deg, overlap {ov['overlap']:.3f}")
    ov_self = subspace.overlap(sd.Q, sd.Q)
    check("a subspace overlaps itself completely", abs(ov_self["overlap"] - 1) < 1e-6 and ov_self["mean_angle_deg"] < 1e-3)

    # steering
    res = subspace.steering_experiment(X, t_dir, 90.0, strict=True)
    full = res["curve"][-1]["mae_to_target_deg"]
    ctl = res["controls"]["mean shift, top-2 axes"]["mae_to_target_deg"]
    check("steering with every probe reaches the target", full < 15,
          f"{res['baseline_mae_to_target_deg']:.0f} deg -> {full:.1f} deg with {res['n_probes']} probes")
    check("2-dim difference-of-means steering reaches the target", ctl < 15, f"{ctl:.1f} deg with 2 dims")
    print(f"      note: the ring was planted in 2 dims; the probe sequence reported {sd.dim}. "
          "Treat that count as method-dependent.")

    # stimuli: the stepped control really hides motion inside each tubelet
    from . import stimuli
    row = {"direction_deg": 45, "speed": 4, "start_x": 0.0, "start_y": 0.0}
    v = stimuli.render_row("velocity_stepped", row, 16, 64)
    w = stimuli.render_row("velocity", row, 16, 64)
    check("stepped videos: frozen inside each frame pair, moving between pairs",
          all((v[2 * i] == v[2 * i + 1]).all() for i in range(8)) and all((v[2 * i + 1] != v[2 * i + 2]).any() for i in range(7))
          and all((w[2 * i] != w[2 * i + 1]).any() for i in range(8)))

    # tuning
    Xp, pref = planted_population(lab)
    tun = tuning.direction_tuning(Xp, lab.direction_deg.to_numpy(float))
    err = np.abs((tun.pref_deg - pref + 180) % 360 - 180)
    check("direction tuning recovers preferred directions", np.median(err) < 5 and (tun.delta_r2 > 0.3).mean() > 0.9,
          f"median error {np.median(err):.1f} deg, {100 * (tun.delta_r2 > 0.3).mean():.0f}% units tuned")
    check("tuned units tile the circle", np.isfinite(tuning.tiling_row(tun)).all())
    spd = lab.speed.to_numpy(float)
    Xs = np.stack([-(spd - p) ** 2 for p in (2.0, 4.0, 6.0)], 1) + 0.05 * np.random.default_rng(0).standard_normal((len(lab), 3))
    st = tuning.speed_tuning(Xs, spd)
    check("speed tuning recovers preferred speeds", np.allclose(st.pref_speed, [2, 4, 6], atol=0.2),
          f"{np.round(st.pref_speed.to_numpy(), 2)}")
    hp = mf.harmonics_1d_cv(Xp, lab, "direction_deg", "start_id")
    check("population code spreads over several harmonics", hp[0] < 0.9 and hp[1] > 0.05,
          f"harmonics {np.round(hp, 2)}")

    # torus
    two = pd.DataFrame([(a, b, k) for a in range(0, 360, 45) for b in range(0, 360, 45) for k in range(3)],
                       columns=["a", "b", "rep"])
    ta, tb = np.deg2rad(two.a.to_numpy()), np.deg2rad(two.b.to_numpy())
    r = np.random.default_rng(2)
    B = np.linalg.qr(r.standard_normal((64, 6)))[0]
    fac = np.stack([np.cos(ta), np.sin(ta), np.cos(tb), np.sin(tb)], 1) @ B[:, :4].T
    conj = np.stack([np.cos(ta + tb), np.sin(ta + tb)], 1) @ B[:, 4:].T
    for name, Xt, want in (("factorised", fac, "(1,0)+(0,1)"), ("conjunctive", conj, "(1,1)")):
        _, M2, _ = mf.condition_means(Xt + 0.05 * r.standard_normal(Xt.shape), two, ["a", "b"])
        h2 = mf.harmonics_2d(M2.reshape(8, 8, -1))
        g = lambda ka, kb: float(h2[(h2.ka == ka) & (h2.kb == kb)].var_frac.iloc[0])
        val = g(1, 0) + g(0, 1) if name == "factorised" else g(1, 1)
        check(f"torus harmonics identify a {name} two-angle code", val > 0.9, f"{want} = {val:.2f}")


def part_b():
    import torch
    from . import models
    from .cli import main
    from .models import AttnEdit
    print("\nB. tiny random-weight V-JEPA 2 through every command")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    enc = models.load_encoder("vjepa2-l", res=64, random_init=True, tiny=True, device="cpu")
    v = np.random.default_rng(0).integers(0, 255, (2, 16, 64, 64, 3), dtype=np.uint8)
    base = enc.run(v)["pooled"]
    tapped = enc.run(v, attn_stats=True)
    check("explicit attention forward matches the stock module", np.abs(base - tapped["pooled"]).max() < 1e-4,
          f"max abs diff {np.abs(base - tapped['pooled']).max():.1e}")
    check("stock attention is restored afterwards", "forward" not in enc.blocks[0].attention.__dict__)
    abl = enc.run(v, attn_edit=AttnEdit((1,), spatial=1.0, temporal=1))["pooled"]
    check("ablation leaves earlier layers untouched and changes later ones",
          np.abs(base[:, 0] - abl[:, 0]).max() == 0 and np.abs(base[:, 1:] - abl[:, 1:]).max() > 0)
    ed = enc.run(v, edits={1: lambda h: h + 1.0})["pooled"]
    check("residual edit lands at its layer only", np.abs(ed[:, 0] - base[:, 0]).max() == 0
          and abs((ed[:, 1] - base[:, 1]).mean() - 1) < 1e-4)
    t, h, w = enc.token_coords(16)
    check("token grid matches input", (int(t.max()) + 1, int(h.max()) + 1, int(w.max()) + 1) == enc.grid(16))
    del enc
    vm = models.load_encoder("videomae-l", res=224, random_init=True, tiny=True, device="cpu")
    r = vm.run(np.random.default_rng(0).integers(0, 255, (1, 16, 64, 64, 3), dtype=np.uint8),
               models.Capture(True, (1,), (1,), (1,), (1,)))
    check("VideoMAE backend: residual and MLP hooks", r["pooled"].shape == (1, 4, 96)
          and r["fc1_L1"].shape == (1, 384) and r["patch_L1"].shape == (1, 196, 96), str(r["pooled"].shape))
    del vm

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        m = ["--model", "vjepa2-l", "--random-init", "--tiny", "--res", "64", "--device", "cpu"]
        steps = [
            ("make velocity", ["make", "velocity", "--out", f"{tmp}/vel", "--grid", "speed=2,4,6", "--n-starts", "5",
                               "--render-res", "64"]),
            ("make permanence", ["make", "permanence", "--out", f"{tmp}/perm", "--grid", "scene_id=0:10:1",
                                 "--render-res", "64"]),
            ("make two_balls", ["make", "two_balls", "--out", f"{tmp}/two", "--grid", "rep=0,1", "speed=2",
                                "--render-res", "64", "--no-preview"]),
            ("make-shuffled (tubelet blocks)", ["make-shuffled", "--data", f"{tmp}/perm", "--out", f"{tmp}/shuf",
                                                "--block", "2"]),
            ("make velocity_stepped", ["make", "velocity_stepped", "--out", f"{tmp}/step", "--grid", "speed=2,4",
                                       "--n-starts", "3", "--render-res", "64", "--no-preview"]),
            ("make-intphys (mock layout)", ["make-intphys", "--root", f"{tmp}/intphys_mock", "--out", f"{tmp}/intphys"]),
            ("extract intphys (mock layout)", ["extract", "--data", f"{tmp}/intphys", "--out", f"{tmp}/acts_intphys", *m]),
            ("make-folder", ["make-folder", "--csv", f"{tmp}/folder/videos.csv", "--out", f"{tmp}/folder_ds",
                             "--target", "moving_right:binary", "--frames", "8"]),
            ("extract shuffled", ["extract", "--data", f"{tmp}/shuf", "--out", f"{tmp}/acts_shuf", *m, "--limit", "24"]),
            ("extract folder (npy, frames dir, mp4)", ["extract", "--data", f"{tmp}/folder_ds", "--out",
                                                       f"{tmp}/acts_folder", *m]),
            ("extract velocity", ["extract", "--data", f"{tmp}/vel", "--out", f"{tmp}/acts_vel", *m,
                                  "--patch-layers", "1,2", "--time-layers", "1", "--fc1-layers", "all",
                                  "--fc2-layers", "all", "--attn-stats"]),
            ("extract permanence", ["extract", "--data", f"{tmp}/perm", "--out", f"{tmp}/acts_perm", *m]),
            ("extract two_balls", ["extract", "--data", f"{tmp}/two", "--out", f"{tmp}/acts_two", *m]),
            ("probe (adamw sweep)", ["probe", "--acts", f"{tmp}/acts_vel", "--out", f"{tmp}/probe", "--epochs", "20"]),
            ("probe (ridge, holdout)", ["probe", "--acts", f"{tmp}/acts_vel", "--out", f"{tmp}/probe_gen",
                                        "--solver", "ridge", "--holdout", "speed>=6",
                                        "--compare", f"cv={tmp}/probe"]),
            ("probe binary", ["probe", "--acts", f"{tmp}/acts_perm", "--out", f"{tmp}/probe_perm", "--solver", "ridge"]),
            ("subspace", ["subspace", "--task", f"direction={tmp}/acts_vel:direction_deg:circular",
                          f"speed={tmp}/acts_vel:speed:scalar", f"possible={tmp}/acts_perm:possible:binary",
                          "--out", f"{tmp}/subspace", "--layers", "1,3", "--max-probes", "6"]),
            ("tuning", ["tuning", "--acts", f"{tmp}/acts_vel", "--out", f"{tmp}/tuning", "--thr", "0.02"]),
            ("steer + propagate", ["steer", "--acts", f"{tmp}/acts_vel", "--out", f"{tmp}/steer", "--strict",
                                   "--max-probes", "6", "--min-probes", "3", "--propagate", "--propagate-n", "8", *m]),
            ("ablate", ["ablate", "--data", f"{tmp}/vel", "--out", f"{tmp}/ablate", *m, "--solver", "ridge",
                        "--base-acts", f"{tmp}/acts_vel", "--limit", "60"]),
            ("ablate (resume, one condition)", ["ablate", "--data", f"{tmp}/vel", "--out", f"{tmp}/ablate", *m,
                                                "--solver", "ridge", "--base-acts", f"{tmp}/acts_vel", "--limit", "60",
                                                "--conditions", "combined"]),
            ("attn", ["attn", "--acts", f"{tmp}/acts_vel", "--out", f"{tmp}/attn"]),
            ("patch", ["patch", "--acts", f"{tmp}/acts_vel", "--out", f"{tmp}/patch"]),
            ("manifold (ring)", ["manifold", "--acts", f"{tmp}/acts_vel", "--out", f"{tmp}/mf", "--by",
                                 "direction_deg", "--embed", "isomap", "--plot-layers", "1"]),
            ("manifold (torus)", ["manifold", "--acts", f"{tmp}/acts_two", "--out", f"{tmp}/mf2", "--by",
                                  "dir_a_deg", "--by2", "dir_b_deg"]),
            ("manifold (fc1 units, scalar)", ["manifold", "--acts", f"{tmp}/acts_vel", "--out", f"{tmp}/mf3",
                                              "--by", "speed", "--kind", "fc1"]),
            ("report", ["report", "--run", str(tmp)]),
        ]
        import contextlib
        import io
        _write_folder_fixture(tmp / "folder")
        for name, argv in steps:
            buf = io.StringIO()
            t0 = time.time()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                    main(argv)
                check(f"cli: {name}", True, f"{time.time() - t0:.1f}s")
            except BaseException as e:  # noqa: BLE001 - report and continue
                check(f"cli: {name}", False, f"{type(e).__name__}: {e}")
                print(buf.getvalue()[-800:])
        figs = list(tmp.rglob("*.png"))
        check("figures were written", len(figs) >= 20, f"{len(figs)} PNG files")
        acts = np.load(tmp / "acts_vel" / "pooled.npy")
        check("activation cache is finite and complete", np.isfinite(acts).all() and acts.shape[:2] == (120, 4),
              str(acts.shape))


def _write_folder_fixture(root: Path):
    """Three tiny 'user videos' in the three supported containers."""
    import cv2
    from PIL import Image
    from . import stimuli
    root.mkdir(parents=True)
    rows = []
    for i, deg in enumerate((0, 180, 0, 180, 0, 180)):
        v = stimuli.render_row("velocity", {"direction_deg": deg, "speed": 3, "start_x": 0.3 * i, "start_y": 0}, 12, 64)
        kind = ("npy", "dir", "mp4")[i % 3]
        if kind == "npy":
            np.save(root / f"v{i}.npy", v)
            path = f"v{i}.npy"
        elif kind == "dir":
            (root / f"v{i}").mkdir()
            for t, fr in enumerate(v):
                Image.fromarray(fr).save(root / f"v{i}" / f"{t:03d}.png")
            path = f"v{i}"
        else:
            w = cv2.VideoWriter(str(root / f"v{i}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 64))
            for fr in v:
                w.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            w.release()
            path = f"v{i}.mp4"
        rows.append({"path": path, "moving_right": int(deg == 0)})
    pd.DataFrame(rows).to_csv(root / "videos.csv", index=False)
    # IntPhys-style tree: <block>/<scene>/<1..4>/{scene/*.png, status.json}
    import json
    mock = root.parent / "intphys_mock"
    for block in ("O1", "O2"):
        for scene in ("001", "002"):
            for k in range(1, 5):
                d = mock / block / scene / str(k) / "scene"
                d.mkdir(parents=True)
                v = stimuli.render_row("velocity", {"direction_deg": 45 * k, "speed": 2, "start_x": 0, "start_y": 0}, 20, 64)
                for t, fr in enumerate(v):
                    Image.fromarray(fr).save(d / f"scene_{t + 1:03d}.png")
                (d.parent / "status.json").write_text(json.dumps({"header": {"is_possible": k <= 2}, "frames": []}))


def part_c():
    import torch
    from . import models
    print("\nC. full-size ViT-L architecture (random weights)")
    enc = models.load_encoder("vjepa2-l", res=224, random_init=True, device="cpu")
    v = np.random.default_rng(0).integers(0, 255, (1, 16, 256, 256, 3), dtype=np.uint8)
    t0 = time.time()
    r = enc.run(v)
    check("ViT-L at 16 x 224 x 224 gives 24 layers x 1024 dims from 1568 tokens",
          r["pooled"].shape == (1, 24, 1024) and int(np.prod(enc.grid(16))) == 1568,
          f"{time.time() - t0:.1f}s per video on {torch.get_num_threads()} CPU threads")
    check("rotary grid follows the input resolution", enc.blocks[0].attention.grid_size == 14)


def run(full=False) -> bool:
    _results.clear()
    part_a()
    part_b()
    if full:
        part_c()
    bad = [n for n, ok in _results if not ok]
    print(f"\n{len(_results) - len(bad)}/{len(_results)} checks passed" + (f"; FAILED: {bad}" if bad else ""))
    return not bad


def make_demo_cache(out, d=256, n_layers=24, units=384, seed=0):
    """Write a SIMULATED activation cache so every analysis command can be tried
    without a GPU or weights. Direction is planted as a planar ring that switches on
    around one-third depth and fades toward the output; speed is present from the
    first layer; MLP units become direction-tuned after the transition. These are
    made-up numbers that mimic the paper's qualitative story; they say nothing
    about V-JEPA."""
    import json

    from . import stimuli
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lab = stimuli.make_manifest("velocity")
    r = np.random.default_rng(seed)
    th = np.deg2rad(lab.direction_deg.to_numpy())
    N = len(lab)
    U = np.linalg.qr(r.standard_normal((d, d)))[0]
    spec = np.arange(1, d + 1) ** -1.0
    spec = spec / np.sqrt((spec ** 2).mean())
    S = np.linalg.qr(r.standard_normal((d, 5)))[0]
    pref = np.linspace(0, 2 * np.pi, units, endpoint=False)
    pooled = np.zeros((N, n_layers, d), dtype=np.float32)
    start = np.stack([lab.start_x, lab.start_y], 1)
    for l in range(n_layers):
        on = 1 / (1 + np.exp(-(l - n_layers / 3 + 0.5) * 2.2))
        fade = 1 - 0.45 * max(0, l - n_layers / 2) / (n_layers / 2)
        g = 3.0 * on * fade
        pooled[:, l] = (g * (np.cos(th)[:, None] * S[:, 0] + np.sin(th)[:, None] * S[:, 1])
                        + 0.9 * (lab.speed.to_numpy() - 4)[:, None] * S[:, 2]
                        + 1.2 * start @ S[:, 3:5].T
                        + 0.5 * (r.standard_normal((N, d)) * spec) @ U.T)
        tuned = np.exp(2.0 * (np.cos(th[:, None] - pref[None]) - 1)) * on * fade
        fc1 = tuned + 0.25 * r.standard_normal((N, units))
        np.save(out / f"fc1_L{l}.npy", fc1.astype(np.float16))
    np.save(out / "pooled.npy", pooled)
    lab.to_csv(out / "labels.csv", index=False)
    fam = stimuli.FAMILIES["velocity"]
    meta = {"model": "SIMULATED (planted structure, not V-JEPA)", "kind": "simulated", "res": 0,
            "frames": 16, "n": N, "n_layers": n_layers, "d_model": d, "d_mlp": units, "n_heads": 0,
            "grid": [8, 14, 14], "patch_layers": [], "time_layers": [],
            "fc1_layers": list(range(n_layers)), "fc2_layers": [], "attn_edit": None, "dataset": "",
            "dataset_meta": {"type": "synthetic", "family": "velocity", "targets": fam.targets,
                             "group": fam.group}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out
