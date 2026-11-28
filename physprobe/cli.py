"""Command line: `python -m physprobe <command> ...`  (see README for the workflow)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _out(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _dump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=2, default=conv))


def _add_model_args(p):
    p.add_argument("--model", default="vjepa2-l",
                   help="alias (vjepa2-l/h/g, vjepa2-g-384, videomae-l), HF repo id, or local dir")
    p.add_argument("--res", type=int, default=224, help="input resolution (paper: 224)")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--frames", type=int, default=None, help="frames per clip (default: dataset's, 16)")
    p.add_argument("--random-init", action="store_true",
                   help="untrained weights: pipeline test and 'no learning' control")
    p.add_argument("--tiny", action="store_true", help="with --random-init: 4-layer toy model")


def _load_enc(a):
    from .models import load_encoder
    return load_encoder(a.model, a.res, a.device, a.dtype, a.random_init, a.tiny)


def _fit_kw(a):
    return {"solver": a.solver, "epochs": a.epochs, "device": a.probe_device}


def _add_probe_args(p):
    p.add_argument("--solver", default="adamw", choices=["adamw", "ridge"],
                   help="adamw = paper's 20-config sweep; ridge = fast closed form")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--probe-device", default="cpu")
    p.add_argument("--group", default=None, help="label column that defines CV groups")


# --------------------------------------------------------------------------- data


def cmd_families(a):
    from .stimuli import FAMILIES
    for f in FAMILIES.values():
        n = int(np.prod([len(v) for v in f.grid.values()])) * max(f.n_starts, 1)
        print(f"\n{f.name}  ({n} videos by default)\n  {f.doc}")
        for k, v in f.grid.items():
            print(f"    {k}: {v if len(v) <= 12 else str(v[:4])[:-1] + ', ... ' + str(v[-1]) + ']'}")
        if f.n_starts:
            print(f"    start positions: {f.n_starts} random in [-{f.start_range}, {f.start_range}]^2 m")
        print(f"  probe targets: {f.targets}   CV groups: {f.group}")


def cmd_make(a):
    from . import datasets, stimuli
    for plug in a.plugin:
        datasets.load_plugin(plug)
    grid = {}
    for g in a.grid:
        k, v = g.split("=", 1)
        grid[k] = stimuli.parse_levels(v)
    ds = datasets.write_synthetic(a.out, a.family, grid, a.n_starts, a.seed, a.frames, a.render_res,
                                  a.floor, a.view_m, a.plugin)
    print(f"{a.family}: {len(ds)} videos -> {a.out}/manifest.csv (frames are rendered on demand)")
    if not a.no_preview:
        datasets.write_contact_sheet(ds, Path(a.out) / "preview.png")
        datasets.write_gif(ds, len(ds) // 2, Path(a.out) / "example.gif")
        print(f"preview: {a.out}/preview.png, {a.out}/example.gif")


def cmd_make_intphys(a):
    from . import datasets
    df = datasets.index_intphys(a.root)
    ds = datasets.write_folder(a.out, df, {"possible": "binary"}, "scene", a.frames, a.sampling,
                               a.stride, {"source": "intphys"})
    print(f"IntPhys: {len(ds)} videos, {df.scene.nunique()} scenes, "
          f"{int(df.possible.sum())} possible / {int((1 - df.possible).sum())} impossible -> {a.out}")
    print(df.groupby('block').possible.agg(['count', 'mean']).rename(columns={'mean': 'frac_possible'}))


def cmd_make_shuffled(a):
    from . import datasets
    ds = datasets.write_shuffled(a.out, a.data, a.block)
    print(f"{len(ds)} videos ({len(ds) // 2} intact + {len(ds) // 2} shuffled in blocks of "
          f"{a.block} frame(s)) -> {a.out}")


def cmd_make_folder(a):
    from . import datasets
    df = pd.read_csv(a.csv)
    if "path" not in df.columns:
        sys.exit("the csv needs a `path` column (video file, frame folder, or .npy)")
    base = Path(a.csv).parent
    df["path"] = [str(p if Path(p).is_absolute() else (base / p).resolve()) for p in df["path"]]
    targets = dict(t.rsplit(":", 1) for t in a.target)
    ds = datasets.write_folder(a.out, df, targets, a.group, a.frames, a.sampling, a.stride)
    print(f"{len(ds)} videos -> {a.out}; targets {targets}; groups {a.group}")


# ------------------------------------------------------------------------ extract


def cmd_extract(a):
    from .datasets import VideoDataset
    from .extract import extract, parse_layers
    from .models import AttnEdit
    ds = VideoDataset(a.data)
    enc = _load_enc(a)
    L = enc.n_layers
    edit = None
    if a.ablate_layers:
        edit = AttnEdit(parse_layers(a.ablate_layers, L), a.ablate_spatial, a.ablate_temporal,
                        a.ablate_combine)
    c = extract(ds, enc, a.out, a.batch_size, a.frames, parse_layers(a.patch_layers, L),
                parse_layers(a.time_layers, L), parse_layers(a.fc1_layers, L),
                parse_layers(a.fc2_layers, L), edit, a.attn_stats, a.limit)
    print(f"activations: {c.pooled.shape} (videos, layers, dim) -> {a.out}")


# -------------------------------------------------------------------------- probe


def _masks(labels, where, holdout):
    row = labels.eval(where).to_numpy(bool) if where else None
    test = labels.eval(holdout).to_numpy(bool) if holdout else None
    return row, test


def _probe_cache(cache, a, layer_spec=None):
    from . import probes
    from .extract import parse_layers
    tg = probes.parse_targets(a.target, cache.labels, cache.targets)
    row, test = _masks(cache.labels, getattr(a, "where", None), getattr(a, "holdout", None))
    layers = parse_layers(layer_spec, cache.n_layers) if layer_spec else None
    df = probes.layerwise(cache, tg, a.group, layers, test, row,
                          paired_col=getattr(a, "paired", None), **_fit_kw(a))
    return df, probes.summarize(df)


def cmd_probe(a):
    from . import plots, probes
    from .extract import ActivationCache
    cache = ActivationCache(a.acts)
    out = _out(a.out)
    df, summ = _probe_cache(cache, a, a.layers)
    df.to_csv(out / "probe_folds.csv", index=False)
    summ.to_csv(out / "probe_summary.csv", index=False)
    extra = {}
    for spec in a.compare:
        label, path = spec.split("=", 1)
        extra[label] = pd.read_csv(Path(path) / "probe_summary.csv")
    plots.layer_curves(summ, out / "layer_curves.png", f"Layerwise probes: {cache.meta['model']}",
                       cache.pez, extra)
    print(f"\n{cache.meta['model']}  ({len(cache)} videos, {cache.n_layers} layers, 1/3 depth = layer {cache.pez})")
    for tgt, g in summ.groupby("target", sort=False):
        col = "accuracy_mean" if g["kind"].iloc[0] == "binary" else "r2_mean"
        g = g.sort_values("layer")
        best = g.loc[g[col].idxmax()]
        em = probes.emergence_layer(summ, tgt)
        pick = sorted({0, cache.pez, cache.n_layers // 2, cache.n_layers - 1})
        vals = "  ".join(f"L{l}={g[g.layer == l][col].iloc[0]:.2f}" for l in pick if (g.layer == l).any())
        print(f"  {tgt:16s} {col[:-5]:8s} {vals}   peak L{int(best.layer)}={best[col]:.2f}   "
              + (f"half-way-to-peak at L{em}" if em is not None else "never decodable"))
        if "paired_accuracy_mean" in g and g["paired_accuracy_mean"].notna().any():
            pc = "paired_accuracy_mean"
            pb = g.loc[g[pc].idxmax()]
            pv = "  ".join(f"L{l}={g[g.layer == l][pc].iloc[0]:.2f}" for l in pick if (g.layer == l).any())
            print(f"  {'':16s} {'paired':8s} {pv}   peak L{int(pb.layer)}={pb[pc]:.2f}")
    print(f"-> {out}/layer_curves.png, probe_summary.csv")


# ----------------------------------------------------------------------- subspace


def _split_target(t, idx):
    from .probes import Target
    return Target(t.name, t.kind, t.Y[idx], None if t.angle is None else t.angle[idx])


def cmd_subspace(a):
    from . import plots, probes, subspace
    from .extract import ActivationCache, parse_layers
    out = _out(a.out)
    tasks = {}
    for spec in a.task:  # name=acts_dir:column:kind
        name, rest = spec.split("=", 1)
        path, col, kind = rest.rsplit(":", 2)
        c = ActivationCache(path)
        tasks[name] = (c, probes.make_target(c.labels, col, kind))
    first = next(iter(tasks.values()))[0]
    layers = parse_layers(a.layers, first.n_layers)
    rows, seq_metrics, task_rows = [], {}, []
    for l in layers:
        seqs = {}
        for name, (c, t) in tasks.items():
            # 80/20 split that keeps CV groups (start positions, matched scenes) intact
            g = probes.resolve_groups(c.labels, c.group)
            m_tr, m_va = probes._inner_split(g, a.seed)
            tr, va = np.where(m_tr)[0], np.where(m_va)[0]
            X = c.layer(l)
            seqs[name] = subspace.orthogonal_probe_sequence(
                X[tr], _split_target(t, tr), X[va], _split_target(t, va),
                max_probes=a.max_probes, seed=a.seed, device=a.probe_device)
            seq_metrics[(l, name)] = seqs[name].metrics
            Z = seqs[name].scaler(X).astype(np.float64)
            er = subspace.erasure_check(Z[tr], t.Y[tr], Z[va], t.Y[va], t.kind)
            task_rows.append({"layer": l, "task": name, "n_probes": seqs[name].n_probes,
                              "probe_sequence_dim": seqs[name].dim,
                              "first_probe_score": seqs[name].metrics[0],
                              "score_before_erasure": er["before"],
                              "score_after_rank_k_erasure": er["after"],
                              "rank_k": er["dims_removed"]})
            print(f"layer {l:2d}  {name:12s} probe sequence: {seqs[name].n_probes:3d} probes / "
                  f"{seqs[name].dim:3d} dims | rank-{er['dims_removed']} covariance erasure: "
                  f"score {er['before']:.2f} -> {er['after']:.2f}")
        dims = {f"dim[{n}]": s.dim for n, s in seqs.items()}
        names = list(seqs)
        for i, na in enumerate(names):
            for nb in names[i + 1:]:
                if seqs[na].dim == 0 or seqs[nb].dim == 0:
                    continue
                ov = subspace.overlap(seqs[na].Q, seqs[nb].Q)
                rows.append({"layer": l, "pair": f"{na} vs {nb}", **dims, **ov})
        if len(names) == 1 or not rows or rows[-1]["layer"] != l:
            rows.append({"layer": l, "pair": "-", **dims})
    df = pd.DataFrame(rows)
    df.to_csv(out / "subspace_by_layer.csv", index=False)
    pd.DataFrame(task_rows).to_csv(out / "dimension_by_layer.csv", index=False)
    _dump({f"L{l}:{n}": m for (l, n), m in seq_metrics.items()}, out / "probe_sequences.json")
    show = first.pez if first.pez in layers else layers[len(layers) // 2]
    plots.probe_sequence({n: seq_metrics[(show, n)] for n in tasks},
                         {n: subspace.THRESHOLDS[t.kind] for n, (c, t) in tasks.items()},
                         out / f"probe_sequence_L{show}.png",
                         f"Orthogonal probe sequence, layer {show}")
    if "mean_angle_deg" in df.columns and df["mean_angle_deg"].notna().any():
        plots.subspace_layers(df[df.pair != "-"], out / "subspace_by_layer.png", first.pez)
    print(f"-> {out}/subspace_by_layer.csv (+ figures)")


# ------------------------------------------------------------------------- tuning


def cmd_tuning(a):
    from . import plots, tuning
    from .extract import ActivationCache
    c = ActivationCache(a.acts)
    out = _out(a.out)
    layers = c.meta.get(f"{a.kind}_layers", [])
    if not layers:
        sys.exit(f"no {a.kind} activations in {a.acts}; re-run extract with --{a.kind}-layers all")
    ang = c.labels[a.angle].to_numpy(float) if a.angle in c.labels else None
    spd = c.labels[a.speed].to_numpy(float) if a.speed in c.labels else None
    Md, Ms, fd, fs, rows = [], [], [], [], []
    for l in layers:
        A = np.asarray(c.get(a.kind, l), dtype=np.float32)
        if ang is not None:
            t = tuning.direction_tuning(A, ang)
            Md.append(tuning.tiling_row(t))
            fd.append(float((t.delta_r2 > a.thr).mean()))
            t.insert(0, "layer", l)
            t.to_csv(out / f"direction_tuning_{a.kind}_L{l}.csv", index=False)
            sel = t[t.delta_r2 > a.thr].sort_values("pref_deg")
            if l == (a.layer if a.layer is not None else min(layers, key=lambda x: abs(x - c.pez))) and len(sel):
                units = sel.unit.to_numpy()
                if len(units) > 400:
                    units = units[np.linspace(0, len(units) - 1, 400).astype(int)]
                plots.tuning_overview(t, tuning.tuning_curves(A, ang, units),
                                      out / f"direction_tuning_{a.kind}_L{l}.png", l, a.kind, a.thr)
            # coverage of the circle by well-tuned units (24 bins of 15 degrees)
            cov = float(np.isfinite(np.where(Md[-1] > a.thr, Md[-1], np.nan)).mean())
            rows.append({"layer": l, "frac_direction_tuned": fd[-1], "circle_coverage": cov,
                         "max_delta_r2": float(t.delta_r2.max())})
        if spd is not None:
            s = tuning.speed_tuning(A, spd)
            lo, hi = np.percentile(spd, [1, 99])
            Ms.append(tuning.tiling_row(s, "pref_speed", 24, lo, hi))
            fs.append(float((s.delta_r2 > a.thr).mean()))
            if ang is None:
                rows.append({"layer": l})
            rows[-1]["frac_speed_tuned"] = fs[-1]
    pd.DataFrame(rows).to_csv(out / f"tuning_by_layer_{a.kind}.csv", index=False)
    if Md:
        plots.tiling_heatmap(np.stack(Md), out / f"direction_tiling_{a.kind}.png",
                             f"Direction tiling by layer ({a.kind} units)", pez=c.pez)
    if Ms:
        plots.tiling_heatmap(np.stack(Ms), out / f"speed_tiling_{a.kind}.png",
                             f"Speed tiling by layer ({a.kind} units)", "preferred speed", lo, hi, c.pez)
    fr = {k: v for k, v in (("direction", fd), ("speed", fs)) if v}
    if fr and len(layers) > 1:
        plots.fraction_tuned(layers, fr, out / f"fraction_tuned_{a.kind}.png", c.pez, a.thr)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"-> {out}")


# -------------------------------------------------------------------------- steer


def cmd_steer(a):
    from . import plots, probes, subspace
    from .extract import ActivationCache
    c = ActivationCache(a.acts)
    out = _out(a.out)
    l = c.pez if a.layer is None else a.layer
    t = probes.make_target(c.labels, a.col, "circular")
    X = c.layer(l)
    res = subspace.steering_experiment(X, t, a.target_deg, strict=a.strict, seed=a.seed,
                                       device=a.probe_device, max_probes=a.max_probes,
                                       min_probes=a.min_probes)
    seq = res.pop("seq")
    levels, mus = res.pop("levels"), res.pop("class_means")
    print(f"layer {l}: {res['n_probes']} orthogonal direction probes ({res['dim']} dims)")
    print(f"evaluation probe on held-out split: R2 {res['eval_probe_r2']:.3f}, "
          f"MAE {res['eval_probe_mae_deg']:.1f} deg")
    print(f"error to {a.target_deg:g} deg without steering: {res['baseline_mae_to_target_deg']:.1f} deg")
    for r in res["curve"]:
        print(f"  steer {r['n_probes']:3d} probes ({r['dims']:3d} dims): error to target "
              f"{r['mae_to_target_deg']:6.1f} deg   |delta|/|x| = {r['relative_shift']:.3f}")
    for name, r in res.get("controls", {}).items():
        print(f"  control, {name} ({r['dims']} dims) -> {res['control_target_deg']:g} deg: error to target "
              f"{r['mae_to_target_deg']:6.1f} deg   |delta|/|x| = {r['relative_shift']:.3f}")
    _dump(res, out / f"steering_L{l}.json")
    if res["curve"]:
        plots.steering_curve(res, out / f"steering_L{l}.png",
                             f"Steering to {a.target_deg:g}° at layer {l} ({c.meta['model']})")
    if a.propagate:
        _steer_propagate(a, c, l, t, seq, levels, mus, out)
    print(f"-> {out}")


def _steer_propagate(a, c, l, t, seq, levels, mus, out):
    """Add a steering shift to every token at layer l, run the rest of the network,
    and read direction at every later layer with that layer's own probe. Done for
    the probe-subspace steer (all probes) and for the difference-of-means shift."""
    import torch
    from . import plots, subspace
    from .datasets import VideoDataset
    from .probes import circ_mae
    ds = VideoDataset(a.data or c.meta["dataset"])
    enc = _load_enc(a)
    rng = np.random.default_rng(a.seed)
    te = rng.permutation(len(c))[int(round(0.7 * len(c))):]  # the held-out split
    te = np.sort(te[:a.propagate_n])  # a random subset (not the lowest ids: the manifest is
    # ordered by condition, so taking the first rows would cover only a few directions)
    Z = seq.scaler(c.layer(l)[te]).astype(np.float64)
    near = int(np.argmin(np.abs((levels - a.target_deg + 180) % 360 - 180)))
    methods = {"difference of means": subspace.steer_mean_shift(
        Z, np.searchsorted(levels, t.angle[te]), mus, near)}
    if seq.n_probes:
        methods[f"probe subspace ({seq.n_probes} probes)"] = subspace.steer(Z, seq, a.target_deg)
    rest = np.setdiff1d(np.arange(len(c)), te)
    readers = []
    for k in range(c.n_layers):  # one read-out probe per layer, fitted on the other videos
        sc = subspace.IsoScaler(c.layer(k)[rest])
        W, b = subspace.ridge_readout(sc(c.layer(k)[rest]).astype(np.float64), t.Y[rest])
        readers.append(lambda Xk, sc=sc, W=W, b=b: sc(Xk).astype(np.float64) @ W + b)
    tgt = np.full(len(te), float(levels[near]))
    df = pd.DataFrame({"layer": np.arange(c.n_layers)})
    df["not steered"] = [circ_mae(tgt, readers[k](c.layer(k)[te])) for k in range(c.n_layers)]
    for name, Zs in methods.items():
        delta = (Zs - Z) * seq.scaler.s  # back to raw activation units
        pooled = []
        for s in range(0, len(te), a.batch_size):
            d = torch.as_tensor(delta[s:s + a.batch_size], dtype=torch.float32, device=enc.device)
            vids = np.stack([ds.get(int(i), c.meta["frames"]) for i in te[s:s + a.batch_size]])
            pooled.append(enc.run(vids, edits={l: lambda h, d=d: h + d[:, None, :].to(h.dtype)})["pooled"])
        P = np.concatenate(pooled)
        df[name] = [circ_mae(tgt, readers[k](P[:, k])) for k in range(c.n_layers)]
    df.to_csv(out / f"steering_propagation_L{l}.csv", index=False)
    plots.lines_by_layer(df.layer, {k: df[k] for k in df.columns if k != "layer"},
                         out / f"steering_propagation_L{l}.png",
                         f"Does steering at layer {l} carry downstream?",
                         f"decoded error to {levels[near]:g}° (deg)", l)
    print(f"\nsteer at layer {l}, read every layer (error to {levels[near]:g} deg):")
    print(df.round(1).to_string(index=False))


# ------------------------------------------------------------------------ disturb


def cmd_disturb(a):
    """Does shifting a variable change the predictor's view of the future more than a
    same-size uninformative disturbance? (see physprobe/disturb.py)"""
    from tqdm import tqdm
    from . import disturb
    from .datasets import VideoDataset
    from .extract import ActivationCache, parse_layers
    from .models import load_encoder
    c = ActivationCache(a.acts)
    out = _out(a.out)
    ds = VideoDataset(a.data or c.meta["dataset"])
    res = a.res or c.meta.get("res", 224)
    name = a.model or c.meta.get("model", "vjepa2-l")
    random_init = a.random_init or name.endswith("-random")  # an untrained-network cache
    enc = load_encoder(name.removesuffix("-random"), res, a.device, a.dtype, random_init, a.tiny,
                       keep_predictor=True)
    if enc.n_layers != c.n_layers:
        raise SystemExit(f"cache has {c.n_layers} layers, model has {enc.n_layers}: use the same model")
    layers = parse_layers(a.layers, c.n_layers) if a.layers else [c.pez]
    vals = c.labels[a.col].to_numpy()
    _, near = disturb.condition_shifts(c.layer(layers[0]), vals, a.target)
    pool = np.where(vals != near)[0]  # videos already at the target value get no shift
    ids = np.sort(np.random.default_rng(a.seed).permutation(pool)[:a.n])
    scales = [float(x) for x in str(a.scales).split(",")]
    summaries = []
    for l in layers:
        bar = tqdm(total=-(-len(ids) // a.batch_size), desc=f"disturb L{l}", unit="batch")
        sizes = [float(x) for x in a.sizes.split(",")] if a.sizes else None
        ring_source = None
        if a.ring_from:
            src = ActivationCache(a.ring_from)
            ring_source = (src.layer(l).astype(np.float64), src.labels[a.col].to_numpy())
        df, P = disturb.run(enc, ds, c, l, a.col, a.target, ids, a.context, a.controls,
                            batch_size=a.batch_size, seed=a.seed, frames=a.frames, match_cols=a.match,
                            scales=scales, target_norm=not a.raw_targets, method=a.method, sizes=sizes,
                            harmonics=a.harmonics, ring_source=ring_source,
                            progress=lambda i, n: bar.update(1))
        bar.close()
        df.to_csv(out / f"disturb_L{l}.csv", index=False)
        np.savez_compressed(out / f"disturb_L{l}_predictions.npz", pooled=P.astype(np.float32))
        sm = disturb.summarize(df)
        sm.insert(0, "layer", l)
        summaries.append(sm)
        print(f"\nlayer {l} [{a.method} steer]: shift towards {a.col} = {df.attrs['target']}, {len(ids)} videos, context "
              f"{df.attrs['context_tubelets']} of {df.attrs['tubelets']} tubelets; L2 prediction error, "
              f"importance = (steer - control) / unperturbed, pull = towards the target video's future")
        for _, r in sm[sm.metric == "l2"].iterrows():
            tag = f"size {r.scale:<4g}" if sizes else f"x{r.scale:<4g}"
            head = (f"  {tag} |shift|/|act| {r.shift_rel:.3f}  error: unperturbed {r.unperturbed:.4f} "
                    f"steer {r.steer:.4f}" + "".join(f" {k} {r[k]:.4f}" for k in ("permuted", "random", "shuffled") if k in r))
            print(head)
            for k in ("permuted", "random", "shuffled"):
                if f"importance_vs_{k}" not in r:
                    continue
                line = (f"         vs {k:8s} importance {r[f'importance_vs_{k}']:+.4f} "
                        f"(s.e. {r[f'importance_vs_{k}_se']:.4f})")
                if f"pull_vs_{k}" in r and pd.notna(r[f"pull_vs_{k}"]):
                    line += f"  pull {r[f'pull_vs_{k}']:+.4f} (s.e. {r[f'pull_vs_{k}_se']:.4f})"
                print(line)
            if "reader_err_deg" in r and pd.notna(r["reader_err_deg"]):
                print(f"         reader check: direction of held-out unperturbed predictions read to within "
                      f"{r.reader_err_deg:.1f} deg of the truth (0 = perfect, 90 = chance)")
            if "speed_change_steer" in r and pd.notna(r["speed_change_steer"]):
                print(f"         predicted speed (reader error on held-out unperturbed predictions "
                      f"{r.get('speed_reader_err', float('nan')):.2f}), median change: steer {r.speed_change_steer:.2f}"
                      + "".join(f", {k} {r[f'speed_change_{k}']:.2f}" for k in ("permuted", "random", "shuffled")
                                if f"speed_change_{k}" in r))
            if "pred_err_to_target_deg_steer" in r and pd.notna(r["pred_err_to_target_deg_steer"]):
                print(f"         predicted future, decoded direction: error to {df.attrs['target']} deg "
                      f"unperturbed {r.pred_err_to_target_deg_unperturbed:.1f}, steer "
                      f"{r.pred_err_to_target_deg_steer:.1f}"
                      + "".join(f", {k} {r[f'pred_err_to_target_deg_{k}']:.1f}"
                                for k in ("permuted", "random", "shuffled") if f"pred_err_to_target_deg_{k}" in r))
    allsum = pd.concat(summaries)
    allsum.to_csv(out / "disturb_summary.csv", index=False)
    v = disturb.verdict(allsum)
    v.to_csv(out / "disturb_verdict.csv", index=False)
    print("\nDecision rule (fixed in advance; primary measure = pull towards the real future of a video "
          "moving in the target direction): PASS if the reader reads held-out predictions to within 45 deg "
          "AND the steer's pull beats every control by >= 2 standard errors.")
    for _, r in v.iterrows():
        zs = "  ".join(f"{c[len('pull_z_vs_'):]} z={r[c]:+.1f}" for c in v.columns if c.startswith("pull_z_vs_"))
        print(f"  layer {int(r.layer)} size {r['size']:g}: reader {r.reader_err_deg:.1f} deg  {zs}  -> {r.verdict}")
    print(f"-> {out}/disturb_summary.csv, disturb_L*.csv")


def cmd_disturb_recheck(a):
    from . import disturb
    run = Path(a.run)
    rows, flips = [], []
    for f in sorted(run.glob("disturb_L*.csv")):
        m = re.match(r"disturb_L(\d+)\.csv", f.name)
        npz = run / f"disturb_L{m.group(1)}_predictions.npz" if m else None
        if not m or not npz.exists():
            continue
        df = pd.read_csv(f)
        P = np.load(npz)["pooled"].astype(np.float64)
        target = float(df.target.iloc[0]) if "target" in df else a.target
        r = disturb.recheck(df, P, a.col, target, a.k)
        r.insert(0, "layer", int(m.group(1)))
        rows.append(r)
        fl = disturb.flip(df)
        if len(fl):
            fl.insert(0, "layer", int(m.group(1)))
            flips.append(fl)
    if not rows:
        raise SystemExit(f"no disturb_L*.csv with saved predictions in {run}")
    out = pd.concat(rows)
    out.to_csv(run / "disturb_recheck.csv", index=False)
    kinds = [c[len("err_to_target_"):] for c in out.columns if c.startswith("err_to_target_")]
    print("Predicted direction, mean error to the target (deg), read by two held-out readers:")
    for _, r in out.iterrows():
        print(f"  layer {int(r.layer)} size {r.scale:g} {r.reader:7s} reader error {r.reader_err_deg:5.1f} | "
              + "  ".join(f"{k} {r[f'err_to_target_{k}']:.1f}" for k in kinds))
    print(f"-> {run}/disturb_recheck.csv")
    if flips:
        fl = pd.concat(flips)
        fl.to_csv(run / "disturb_flip.csv", index=False)
        print("No reader: is the predicted future closer to the matched target-direction video's real future "
              "than to its own?")
        for _, r in fl.iterrows():
            ctr = sorted(c[len("flip_z_vs_"):] for c in fl.columns if c.startswith("flip_z_vs_"))
            print(f"  layer {int(r.layer)} size {r.scale:g} ({int(r.videos)} videos) flipped: "
                  + "  ".join(f"{k} {r[f'flip_{k}']:.0%}" for k in ["unperturbed", "steer"] + ctr))
            print("      steer vs control, z: " + "  ".join(f"{k} {r[f'flip_z_vs_{k}']:.1f}" for k in ctr)
                  + "   | share of mean gap closed: "
                  + "  ".join(f"{k} {r[f'gap_closed_{k}']:+.2f}" for k in ["steer"] + ctr))
        print(f"-> {run}/disturb_flip.csv")


def cmd_disturb_figure(a):
    from . import disturb
    from .datasets import VideoDataset
    run = Path(a.run)
    df = pd.read_csv(run / f"disturb_L{a.layer}.csv")
    ds = VideoDataset(a.data)
    df.attrs["target"] = a.target
    path = run / f"disturb_L{a.layer}_x{a.scale:g}.png"
    pick = disturb.figure(df, ds, path, a.scale, a.col, a.examples, seed=a.seed,
                          title=a.title or f"Steering direction at layer {a.layer} (x{a.scale:g})")
    print(f"examples: videos {pick}\n-> {path}")


# ------------------------------------------------------------------------- ablate

PAPER_ABLATIONS = {"spatial s=7": (7.0, None), "temporal t=3": (None, 3), "combined s=3,t=1": (3.0, 1)}


def cmd_ablate(a):
    from . import plots
    from .datasets import VideoDataset
    from .extract import ActivationCache, extract, parse_layers
    from .models import AttnEdit
    out = _out(a.out)
    ds = VideoDataset(a.data)
    enc = _load_enc(a)
    L = enc.n_layers
    layers = parse_layers(a.layers, L) if a.layers else (L // 3 - 1, L // 3)
    read = parse_layers(a.read_layers, L) if a.read_layers else (L // 3 + 2, L // 2, L - 1)
    from dataclasses import asdict
    conds = {"base": None}
    wanted = {"spatial": "spatial s=7", "temporal": "temporal t=3", "combined": "combined s=3,t=1"}
    for key in a.conditions:
        s, t = PAPER_ABLATIONS[wanted[key]]
        conds[wanted[key]] = AttnEdit(tuple(layers), s, t, a.combine)
    summaries, table = {}, {}
    for name, edit in conds.items():
        sub = out / name.replace(" ", "_").replace(",", "_").replace("=", "")
        meta = sub / "meta.json"
        if name == "base" and a.base_acts:
            c = ActivationCache(a.base_acts)
        elif meta.exists() and json.loads(meta.read_text()).get("attn_edit") == \
                json.loads(json.dumps(asdict(edit) if edit else None)) and \
                json.loads(meta.read_text()).get("n") == (min(a.limit, len(ds)) if a.limit else len(ds)):
            print(f"{name}: reusing finished activations in {sub}")  # resume after an interruption
            c = ActivationCache(sub)
        else:
            c = extract(ds, enc, sub, a.batch_size, a.frames, attn_edit=edit, limit=a.limit)
        _, summ = _probe_cache(c, a)
        summ.to_csv(_out(sub) / "probe_summary.csv", index=False)
        summaries[name] = summ
        for tgt, g in summ.groupby("target", sort=False):
            col = "accuracy_mean" if g["kind"].iloc[0] == "binary" else "r2_mean"
            for l in read:
                table.setdefault(f"{tgt} @L{l}", {})[name] = float(g[g.layer == l][col].iloc[0])
    tab = pd.DataFrame(table)
    tab.to_csv(out / "ablation_table.csv")
    plots.ablation_bars(tab, out / "ablation_bars.png",
                        f"Removing short-range attention at layers {list(layers)}")
    plots.condition_curves(summaries, out / "ablation_layer_curves.png",
                           "Layerwise probes with and without the attention ablation", L // 3, layers)
    print(f"\nablated layers {list(layers)}; rows = condition, columns = probe target @ read-out layer")
    print(tab.round(3).to_string())
    print(f"-> {out}")


def cmd_attn(a):
    from . import plots
    from .extract import ActivationCache
    c = ActivationCache(a.acts)
    f = Path(a.acts) / "attn_distance.npz"
    if not f.exists():
        sys.exit("no attention statistics in this cache; re-run extract with --attn-stats")
    z = np.load(f)
    out = _out(a.out)
    plots.attention_distance(z["spatial"], z["temporal"], out / "attention_distance.png", c.pez)
    df = pd.DataFrame({"layer": np.arange(c.n_layers), "spatial_mean": z["spatial"].mean(1),
                       "spatial_min_head": z["spatial"].min(1), "spatial_max_head": z["spatial"].max(1),
                       "spatial_head_sd": z["spatial"].std(1), "temporal_mean": z["temporal"].mean(1),
                       "temporal_min_head": z["temporal"].min(1)})
    df.to_csv(out / "attention_distance.csv", index=False)
    print(df.round(2).to_string(index=False))
    print(f"-> {out}/attention_distance.png")


# -------------------------------------------------------------------------- patch


def _ridge_primal(X, Y, alpha):
    mu, ymu = X.mean(0, keepdims=True), Y.mean(0, keepdims=True)
    Xc = X - mu
    G = Xc.T @ Xc
    W = np.linalg.solve(G + alpha * np.trace(G) / len(G) * np.eye(len(G)), Xc.T @ (Y - ymu))
    return W, (ymu - mu @ W)[0]


def cmd_patch(a):
    from . import plots, probes
    from .extract import ActivationCache
    c = ActivationCache(a.acts)
    out = _out(a.out)
    layers = c.meta["patch_layers"]
    if not layers:
        sys.exit("no patch activations; re-run extract with --patch-layers thirds")
    col, kind = a.target.rsplit(":", 1) if a.target else next(iter(c.targets.items()))
    t = probes.make_target(c.labels, col, kind)
    if t.kind == "binary":
        sys.exit("patch probes are implemented for regression targets")
    groups = probes.resolve_groups(c.labels, a.group or c.group)
    tr, te = probes.make_folds(groups, None, 5)[0]
    G = c.meta["grid"][1]
    cols_ = np.arange(G * G) % G
    left, right = np.where(cols_ < G // 2)[0], np.where(cols_ >= G // 2)[0]
    rows = []
    for l in layers:
        P = np.asarray(c.get("patch", l), dtype=np.float32)  # [N, G*G, d]
        sd = P[tr].std() + 1e-8
        P = (P - P[tr].mean((0, 1), keepdims=True)) / sd
        d = P.shape[-1]
        rep = lambda idx, pp: np.repeat(t.Y[idx], len(pp), axis=0)
        # (1) one probe shared by every patch; scored per patch position on held-out videos
        W, b = _ridge_primal(P[tr].reshape(-1, d).astype(np.float64), rep(tr, np.arange(G * G)), a.alpha)
        pred = P[te] @ W + b  # [n_te, G*G, k]
        r2map = np.array([probes.r2_score(t.Y[te], pred[:, p]) for p in range(G * G)]).reshape(G, G)
        plots.patch_map(r2map, out / f"patch_r2_L{l}.png",
                        f"Per-patch R² for {col}, layer {l}\n(held-out videos)")
        # (2) train on left-half patches, test on right-half patches of held-out videos
        W2, b2 = _ridge_primal(P[tr][:, left].reshape(-1, d).astype(np.float64), rep(tr, left), a.alpha)
        p2 = (P[te][:, right] @ W2 + b2).reshape(-1, t.Y.shape[1])
        row = {"layer": l, "r2_all_patches": probes.r2_score(rep(te, np.arange(G * G)), pred.reshape(-1, t.Y.shape[1])),
               "r2_left_to_right": probes.r2_score(rep(te, right), p2),
               "frac_patches_r2>0.5": float((r2map > 0.5).mean())}
        if a.attentive:
            df = probes.cv_attentive(c.get("patch", l), t, groups, device=a.probe_device)
            row["attentive_r2"] = float(df["r2"].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "patch_probe.csv", index=False)
    print(df.round(3).to_string(index=False))
    print(f"-> {out}")


# ----------------------------------------------------------------------- manifold


def cmd_manifold(a):
    from . import manifold as mf
    from . import plots
    from .extract import ActivationCache, parse_layers
    c = ActivationCache(a.acts)
    out = _out(a.out)
    labels = c.labels
    keep = labels.eval(a.where).to_numpy(bool) if a.where else np.ones(len(labels), bool)
    labels = labels[keep].reset_index(drop=True)
    if a.kind == "pooled":
        layers = list(parse_layers(a.layers, c.n_layers))
        getX = lambda l: c.layer(l)[keep]
    else:
        layers = c.meta[f"{a.kind}_layers"]
        getX = lambda l: np.asarray(c.get(a.kind, l), dtype=np.float32)[keep]
    vals = np.sort(labels[a.by].unique())
    circular = a.circular or (c.targets.get(a.by) == "circular" and mf.is_circular_grid(vals))
    if a.factors:
        factors = a.factors.split(",")
    else:  # the design's crossed factors, for synthetic sets
        from .stimuli import FAMILIES
        fam = FAMILIES.get(c.meta["dataset_meta"].get("family"))
        factors = [f for f in (list(fam.grid) if fam else []) + ["start_id"]
                   if f in labels.columns and labels[f].nunique() > 1]
    if a.by not in factors:
        factors = [a.by] + factors
    plot_layers = set(parse_layers(a.plot_layers, c.n_layers)) & set(layers) or {layers[len(layers) // 2]}
    rows, H, parts = [], [], {}
    two = a.by2 is not None
    for l in layers:
        X = getX(l)
        keys, M, _ = mf.condition_means(X, labels, [a.by])
        P = mf.pca(M, min(6, len(M) - 1))
        cv = mf.cv_spectrum(X, labels, [a.by], c.group)
        rel = mf.reliability(X, labels, [a.by], c.group)
        ok = rel >= a.min_reliability  # below this the manifold is noise: blank its shape stats
        row = {"layer": l, "reliability": rel, "signal_dims90": cv["dims90"] if ok else np.nan,
               "signal_pr": cv["pr"] if ok else np.nan,
               "pc12_var": float(P["var_ratio"][:2].sum())}
        vp = mf.variance_partition(X, labels, factors, c.group)
        for k, v in vp.items():
            parts.setdefault(k, []).append(v)
        row["signal_fraction"] = vp[a.by]
        if circular:
            h = mf.harmonics_1d_cv(X, labels, a.by, c.group)[:4]
            h = h if ok else np.full(len(h), np.nan)
            H.append(h)
            row.update({f"harmonic{k + 1}": float(h[k]) for k in range(len(h))})
            shape = {**mf.harmonic_plane_dim(M), **mf.ring_fit(P["scores"][:, :2], keys[a.by].to_numpy(float))}
            row.update(shape if ok else {k: np.nan for k in shape})
        else:
            from scipy.stats import spearmanr
            row["pc1_rank_corr"] = float(abs(spearmanr(P["scores"][:, 0], keys[a.by]).statistic)) if ok else np.nan
        if two:
            k2, M2, _ = mf.condition_means(X, labels, [a.by, a.by2])
            na, nb = k2[a.by].nunique(), k2[a.by2].nunique()
            h2 = mf.harmonics_2d(M2.reshape(na, nb, -1))
            g = lambda ka, kb: float(h2[(h2.ka == ka) & (h2.kb == kb)].var_frac.iloc[0])
            row.update({"h(1,0)": g(1, 0), "h(0,1)": g(0, 1), "h(1,1)": g(1, 1), "h(1,-1)": g(1, -1),
                        "reliability_2d": mf.reliability(X, labels, [a.by, a.by2], c.group)})
        ph = mf.persistent_h1(M) if ok else None  # only if `ripser` is installed
        if ph:
            row.update({"loop_persistence": ph["top"], "second_loop": ph["second"]})
        rows.append(row)
        if l in plot_layers:
            plots.manifold_scatter(P["scores"], keys[a.by].to_numpy(float),
                                   out / f"means_{a.by}_L{l}.png",
                                   f"{a.by}: condition means, layer {l} ({a.kind})", circular,
                                   P["var_ratio"], a.by)
            plots.rdm_plot(mf.rdm(M), list(keys[a.by]), out / f"rdm_{a.by}_L{l}.png",
                           f"Distances between {a.by} levels, layer {l}")
            if two:
                P2 = mf.pca(M2, 4)
                plots.manifold_scatter(P2["scores"], k2[a.by].to_numpy(float),
                                       out / f"means_{a.by}_x_{a.by2}_L{l}.png",
                                       f"{a.by} x {a.by2}: condition means, layer {l}", circular,
                                       P2["var_ratio"], f"{a.by}; lines join points sharing {a.by2}",
                                       second=k2[a.by2].to_numpy())
            if a.embed:
                E = mf.embed(M, a.embed, min(3, len(M) - 2))
                plots.manifold_scatter(E, keys[a.by].to_numpy(float),
                                       out / f"{a.embed}_{a.by}_L{l}.png",
                                       f"{a.by}: {a.embed} of condition means, layer {l}", circular,
                                       None, a.by)
            if l in c.meta.get("time_layers", []) and a.kind == "pooled":
                Tk = np.asarray(c.get("time", l), dtype=np.float32)[keep]
                _, TM, _ = mf.condition_means(Tk.reshape(len(Tk), -1), labels, [a.by])
                TM = TM.reshape(len(TM), Tk.shape[1], -1)
                flat = TM.reshape(-1, TM.shape[-1])
                comp = mf.pca(flat, 2)
                sc = ((flat - comp["mean"]) @ comp["components"].T).reshape(len(TM), Tk.shape[1], 2)
                v = keys[a.by].to_numpy(float)
                plots.trajectories(sc, (v - v.min()) / max(v.max() - v.min(), 1e-9) * (0.94 if circular else 1),
                                   out / f"time_trajectories_{a.by}_L{l}.png",
                                   f"{a.by}: temporal-token trajectories, layer {l}",
                                   [f"{x:g}" for x in v], circular)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"manifold_{a.by}_by_layer.csv", index=False)
    L = [r["layer"] for r in rows]
    plots.stacked_by_layer(L, parts, out / "variance_partition.png",
                           "What the activation variance is about", "fraction of total variance", c.pez)
    plots.lines_by_layer(L, {"split-half reliability": df.reliability}, out / f"reliability_{a.by}.png",
                         f"Is there a reproducible {a.by} manifold?", "split-half reliability", c.pez)
    plots.lines_by_layer(L, {"axes for 90% of signal variance": df.signal_dims90,
                             "participation ratio": df.signal_pr}, out / f"dimensionality_{a.by}.png",
                         f"Dimensionality of the {a.by} manifold (noise-corrected)",
                         f"dimensions (max {len(vals) - 1} for {len(vals)} levels)", c.pez)
    if circular and H:
        plots.heat_by_layer(np.stack(H), out / f"harmonics_{a.by}.png", f"Fourier content of {a.by} code",
                            [str(k + 1) for k in range(np.stack(H).shape[1])], "harmonic of the angle",
                            "fraction of across-angle variance")
    if two:
        plots.lines_by_layer(L, {"(1,0) " + a.by: df["h(1,0)"], "(0,1) " + a.by2: df["h(0,1)"],
                                 "(1,1) conjunctive": df["h(1,1)"], "(1,-1) conjunctive": df["h(1,-1)"]},
                             out / "torus_harmonics.png", "Factorised vs conjunctive code for two angles",
                             "fraction of condition-mean variance", c.pez)
    pd.set_option("display.width", 200)
    print(f"{a.by}: {len(vals)} levels, treated as {'circular' if circular else 'linear'}; "
          f"variance factors: {factors}")
    print(df.round(3).to_string(index=False))
    print(f"-> {out}")


# ----------------------------------------------------------------- report / tests


def cmd_report(a):
    import html
    root = Path(a.run)
    parts = ["<!doctype html><meta charset='utf-8'><title>physprobe report</title>",
             "<style>body{font:14px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;"
             "padding:0 1rem;color:#0b0b0b;background:#f9f9f7}img{max-width:100%;border:1px solid #e1e0d9;"
             "margin:.4rem 0}h2{margin-top:2.2rem;border-bottom:1px solid #e1e0d9}pre{background:#fff;"
             "padding:.6rem;overflow:auto;border:1px solid #e1e0d9;font-size:12px}</style>",
             f"<h1>physprobe report: {html.escape(root.name)}</h1>"]
    for d in sorted(p for p in root.rglob("*") if p.is_dir() or p == root):
        pngs = sorted(d.glob("*.png"))
        csvs = [f for f in sorted(d.glob("*.csv")) if f.stat().st_size < 20000 and f.name != "labels.csv"
                and f.name != "manifest.csv"]
        if not pngs and not csvs:
            continue
        parts.append(f"<h2>{html.escape(str(d.relative_to(root)) or '.')}</h2>")
        for f in csvs:
            parts.append(f"<h4>{f.name}</h4><pre>{html.escape(pd.read_csv(f).round(3).to_string(index=False))}</pre>")
        for f in pngs:
            parts.append(f"<h4>{f.name}</h4><img src='{f.relative_to(root).as_posix()}'>")
    (root / "report.html").write_text("\n".join(parts))
    print(f"-> {root}/report.html")


def cmd_demo(a):
    from .selftest import make_demo_cache
    make_demo_cache(a.out)
    print(f"SIMULATED activations (planted structure, not V-JEPA) -> {a.out}\n"
          f"try:  python -m physprobe probe --acts {a.out} --out {a.out}/probe --solver ridge\n"
          f"      python -m physprobe manifold --acts {a.out} --out {a.out}/manifold --by direction_deg\n"
          f"      python -m physprobe steer --acts {a.out} --out {a.out}/steer")


def cmd_export(a):
    from .export import export_bundle
    out = export_bundle(a.runs, a.out, a.data, a.labels or None, a.zip)
    print(f"-> {out}")


def cmd_gui(a):
    import subprocess
    try:
        import plotly  # noqa: F401
        import streamlit  # noqa: F401
    except ImportError:
        sys.exit("The explorer needs two extra packages:  pip install streamlit plotly")
    app = Path(__file__).parent / "gui" / "app.py"
    theme = ["--theme.base", "light", "--theme.primaryColor", "#0e6f78", "--theme.backgroundColor", "#f6f8f9",
             "--theme.secondaryBackgroundColor", "#e8eef1", "--theme.textColor", "#17212b"]
    argv = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port),
            "--browser.gatherUsageStats", "false", "--client.toolbarMode", "viewer", *theme]
    if a.no_browser:
        argv += ["--server.headless", "true"]
    argv += ["--", "--runs", str(Path(a.runs).resolve()), "--data", str(Path(a.data).resolve())]
    if a.hosted:
        argv.append("--hosted")
    print(f"physprobe explorer: http://localhost:{a.port}  (Ctrl+C to stop)")
    subprocess.run(argv)


def cmd_selftest(a):
    from .selftest import run
    sys.exit(0 if run(full=a.full) else 1)


# --------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(prog="physprobe", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("families", help="list built-in stimulus families").set_defaults(fn=cmd_families)

    p = sub.add_parser("make", help="create a synthetic dataset")
    p.add_argument("family")
    p.add_argument("--out", required=True)
    p.add_argument("--grid", nargs="*", default=[], metavar="FACTOR=LEVELS",
                   help="override factor levels, e.g. direction_deg=0:360:15 speed=2,4,6")
    p.add_argument("--n-starts", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--render-res", type=int, default=256)
    p.add_argument("--floor", default="noise", choices=["noise", "plain", "checker"])
    p.add_argument("--view-m", type=float, default=8.0, help="side of the visible area in metres")
    p.add_argument("--plugin", nargs="*", default=[], help="python files registering extra families")
    p.add_argument("--no-preview", action="store_true")
    p.set_defaults(fn=cmd_make)

    p = sub.add_parser("make-intphys", help="index an IntPhys dev download")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--sampling", default="uniform", choices=["uniform", "center", "first"])
    p.add_argument("--stride", type=int, default=1)
    p.set_defaults(fn=cmd_make_intphys)

    p = sub.add_parser("make-shuffled", help="intact vs frame-shuffled copies of a dataset (control)")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--block", type=int, default=1,
                   help="shuffle blocks of this many frames; 2 = whole tubelets (motion within "
                        "each token stays intact)")
    p.set_defaults(fn=cmd_make_shuffled)

    p = sub.add_parser("make-folder", help="wrap your own videos (csv with a `path` column + labels)")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target", nargs="+", required=True, metavar="COLUMN:KIND")
    p.add_argument("--group", default=None)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--sampling", default="uniform", choices=["uniform", "center", "first"])
    p.add_argument("--stride", type=int, default=1)
    p.set_defaults(fn=cmd_make_folder)

    p = sub.add_parser("extract", help="run the encoder and cache activations")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    _add_model_args(p)
    p.add_argument("--patch-layers", default="none", help="all | thirds | 4,8,12 | none")
    p.add_argument("--time-layers", default="none")
    p.add_argument("--fc1-layers", default="none")
    p.add_argument("--fc2-layers", default="none")
    p.add_argument("--attn-stats", action="store_true", help="record per-head attention distance")
    p.add_argument("--ablate-layers", default=None)
    p.add_argument("--ablate-spatial", type=float, default=None)
    p.add_argument("--ablate-temporal", type=int, default=None)
    p.add_argument("--ablate-combine", default="or", choices=["and", "or"])
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("probe", help="layerwise linear probes")
    p.add_argument("--acts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target", nargs="*", default=[], metavar="COLUMN:KIND",
                   help="default: the dataset's own targets")
    p.add_argument("--layers", default=None)
    p.add_argument("--where", default=None, help="pandas expression: rows to use")
    p.add_argument("--holdout", default=None,
                   help="pandas expression: test on these rows, train on the rest (e.g. 'speed>=6')")
    p.add_argument("--compare", nargs="*", default=[], metavar="LABEL=PROBE_DIR",
                   help="overlay other probe runs (e.g. random=runs/vel_random/probe)")
    p.add_argument("--paired", default=None, metavar="COLUMN",
                   help="label column of matched sets (IntPhys, permanence: scene); binary targets "
                        "then also report paired accuracy within each set")
    _add_probe_args(p)
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("subspace", help="orthogonal probe sequences and subspace overlap")
    p.add_argument("--task", nargs="+", required=True, metavar="NAME=ACTS:COLUMN:KIND")
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="all")
    p.add_argument("--max-probes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-device", default="cpu")
    p.set_defaults(fn=cmd_subspace)

    p = sub.add_parser("tuning", help="direction / speed tuning of MLP units")
    p.add_argument("--acts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kind", default="fc1", choices=["fc1", "fc2"])
    p.add_argument("--angle", default="direction_deg")
    p.add_argument("--speed", default="speed")
    p.add_argument("--layer", type=int, default=None, help="layer for the detailed figure")
    p.add_argument("--thr", type=float, default=0.1, help="delta R^2 that counts as tuned")
    p.set_defaults(fn=cmd_tuning)

    p = sub.add_parser("disturb", help="functional importance: does a shift of a variable change "
                                       "the predictor's prediction more than a same-size control?")
    p.add_argument("--acts", required=True, help="activation cache of the dataset (pooled.npy)")
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=None, help="dataset folder (default: the cache's)")
    p.add_argument("--col", default="direction_deg", help="variable to shift")
    p.add_argument("--target", type=float, default=90.0, help="value to shift towards (nearest level)")
    p.add_argument("--layers", default=None, help="injection layers, e.g. 4,8,12 (default: one-third depth)")
    p.add_argument("--context", type=int, default=None,
                   help="tubelets given to the predictor as context (default: first half)")
    p.add_argument("--n", type=int, default=48, help="videos (random subset)")
    p.add_argument("--controls", type=int, default=3, help="repeats of each control per video")
    p.add_argument("--scales", default="1,4,16",
                   help="multiples of the difference-of-means shift (controls use the same norms)")
    p.add_argument("--method", default="mean", choices=["mean", "ring"],
                   help="mean: difference of condition means; ring: move along a fitted ring")
    p.add_argument("--harmonics", type=int, default=2, help="harmonics of the ring model (--method ring)")
    p.add_argument("--ring-from", default=None, metavar="ACTS",
                   help="fit the ring on another activation cache (same model), e.g. stepped videos, "
                        "and apply it to this dataset (--method ring)")
    p.add_argument("--sizes", default=None,
                   help="shift sizes as fractions of each activation's norm, e.g. 0.2,0.4 (replaces --scales)")
    p.add_argument("--raw-targets", action="store_true",
                   help="compare with raw encoder outputs instead of layer-normalised ones "
                        "(V-JEPA's training target is layer-normalised)")
    p.add_argument("--match", nargs="*", default=["speed", "start_id"],
                   help="factors held fixed to find the steered video's real future (pull)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=None, help="default: the cache's model")
    p.add_argument("--res", type=int, default=None, help="default: the cache's resolution")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.set_defaults(fn=cmd_disturb)

    p = sub.add_parser("disturb-recheck", help="re-read a disturb run's predictions with a linear and a "
                                               "non-linear (nearest-neighbour) held-out reader")
    p.add_argument("--run", required=True, help="output folder of disturb (needs the saved predictions)")
    p.add_argument("--col", default="direction_deg")
    p.add_argument("--target", type=float, default=90.0, help="used only for runs without a target column")
    p.add_argument("--k", type=int, default=5, help="neighbours of the non-linear reader")
    p.set_defaults(fn=cmd_disturb_recheck)

    p = sub.add_parser("disturb-figure", help="picture of a disturb run: predicted directions before "
                                              "and after the steer")
    p.add_argument("--run", required=True, help="output folder of disturb")
    p.add_argument("--data", required=True, help="dataset folder (for the video frames)")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--scale", type=float, default=16.0)
    p.add_argument("--target", type=float, default=90.0, help="the --target used for disturb")
    p.add_argument("--col", default="direction_deg")
    p.add_argument("--examples", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--title", default=None)
    p.set_defaults(fn=cmd_disturb_figure)

    p = sub.add_parser("steer", help="subspace steering of a circular variable")
    p.add_argument("--acts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--col", default="direction_deg")
    p.add_argument("--layer", type=int, default=None, help="default: one-third depth")
    p.add_argument("--target-deg", type=float, default=90.0)
    p.add_argument("--strict", action="store_true",
                   help="evaluation probe never sees the activations it scores")
    p.add_argument("--max-probes", type=int, default=60)
    p.add_argument("--min-probes", type=int, default=0,
                   help="keep at least this many probes even below the stop threshold")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-device", default="cpu")
    p.add_argument("--propagate", action="store_true",
                   help="also inject the shift into the network and read later layers")
    p.add_argument("--propagate-n", type=int, default=64)
    p.add_argument("--data", default=None)
    _add_model_args(p)
    p.set_defaults(fn=cmd_steer)

    p = sub.add_parser("ablate", help="attention ablations at the emergence layers (Table 2)")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    _add_model_args(p)
    p.add_argument("--layers", default=None, help="layers to ablate (default: L/3-1, L/3)")
    p.add_argument("--read-layers", default=None, help="layers to report (default: L/3+2, L/2, last)")
    p.add_argument("--combine", default="or", choices=["and", "or"],
                   help="combined condition: remove keys near in space OR time (default) or AND")
    p.add_argument("--base-acts", default=None, help="reuse an existing un-ablated cache")
    p.add_argument("--conditions", nargs="+", default=["spatial", "temporal", "combined"],
                   choices=["spatial", "temporal", "combined"],
                   help="which of the paper's three ablations to run (each is one full extraction)")
    p.add_argument("--target", nargs="*", default=[])
    p.add_argument("--limit", type=int, default=None)
    _add_probe_args(p)
    p.set_defaults(fn=cmd_ablate)

    p = sub.add_parser("attn", help="attention-distance figure (needs extract --attn-stats)")
    p.add_argument("--acts", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_attn)

    p = sub.add_parser("patch", help="per-patch probes (needs extract --patch-layers)")
    p.add_argument("--acts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target", default=None, metavar="COLUMN:KIND")
    p.add_argument("--group", default=None)
    p.add_argument("--alpha", type=float, default=1.0, help="ridge strength (relative)")
    p.add_argument("--attentive", action="store_true", help="also train attentive probes")
    p.add_argument("--probe-device", default="cpu")
    p.set_defaults(fn=cmd_patch)

    p = sub.add_parser("manifold", help="geometry of a variable's condition means, by layer")
    p.add_argument("--acts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--by", required=True, help="label column to trace")
    p.add_argument("--by2", default=None, help="second circular column (torus analysis)")
    p.add_argument("--circular", action="store_true", help="force circular treatment")
    p.add_argument("--kind", default="pooled", choices=["pooled", "fc1", "fc2"])
    p.add_argument("--layers", default="all")
    p.add_argument("--plot-layers", default="thirds")
    p.add_argument("--factors", default=None, help="comma list for the variance partition")
    p.add_argument("--where", default=None)
    p.add_argument("--embed", default=None, choices=["isomap", "mds", "umap"])
    p.add_argument("--min-reliability", type=float, default=0.3,
                   help="blank the shape statistics of layers whose split-half reliability is lower")
    p.set_defaults(fn=cmd_manifold)

    p = sub.add_parser("report", help="collect a run folder's figures and tables into report.html")
    p.add_argument("--run", required=True)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("export", help="bundle results for sharing or hosting, without activation arrays")
    p.add_argument("--runs", default="runs", help="results folder")
    p.add_argument("--data", default="data", help="datasets folder (for preview images)")
    p.add_argument("--out", required=True, help="bundle folder to write")
    p.add_argument("--labels", nargs="*", default=[], help="labels to export shapes for (default: design factors)")
    p.add_argument("--zip", action="store_true", help="also write <out>.zip")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("gui", help="open the interactive explorer in a browser")
    p.add_argument("--runs", default="runs", help="results folder to explore")
    p.add_argument("--data", default="data", help="datasets folder")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--no-browser", action="store_true", help="don't open a browser window (e.g. on a server)")
    p.add_argument("--hosted", action="store_true", help="read-only mode, as on a public server")
    p.set_defaults(fn=cmd_gui)

    p = sub.add_parser("demo", help="write a SIMULATED activation cache to try the analyses on")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("selftest", help="check the analyses on planted structure + a toy model")
    p.add_argument("--full", action="store_true", help="also build a full-size ViT-L (slow on CPU)")
    p.set_defaults(fn=cmd_selftest)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
