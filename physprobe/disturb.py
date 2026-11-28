"""Functional importance of a decodable variable: does shifting it change what the
model predicts, more than a disturbance of the same size that carries no information?

Follows the logic of Thorat, Aldegheri & Kietzmann (arXiv:2111.07898): a systematic
perturbation that signals a different value of the variable is compared with a
control of the same norm whose elements are randomly permuted, so that on average it
aligns with no variable. Here the network is V-JEPA 2 and its "task" is its own
training objective: predicting the representations of the second half of a clip from
the first half.

For each video v and injection layer l:
  delta_v   = mu_l(target value) - mu_l(value of v)   (difference of condition means of
              the mean-pooled residual stream at layer l), added to every token at l
  controls  = delta_v with its elements randomly permuted ("permuted", as in the paper),
              and a Gaussian vector rescaled to |delta_v| ("random")
  error     = mean squared (and absolute) difference between the predictor's output for
              the future tokens and the *unperturbed* encoder's outputs for those tokens

functional importance = (E_steer - E_control) / E_unperturbed
  > 0: the informative shift hurts prediction more than an uninformative one of the
       same size, i.e. the variable is used for predicting the future;
  ~ 0: the shift is no more disruptive than noise.
Where a matched video exists (same values of the other factors, target value of the
variable), the error is also computed against that video's future ("pull"): a steer
that the predictor uses should move its prediction towards the steered future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def condition_shifts(X: np.ndarray, values: np.ndarray, target) -> tuple[np.ndarray, float]:
    """Per-video shift from the video's own condition mean to the mean of the level
    nearest to `target`. X [N, d] pooled activations at one layer."""
    levels = np.unique(values)
    if np.issubdtype(levels.dtype, np.number):
        diff = np.abs(levels.astype(float) - float(target))
        if np.ptp(levels) > 0 and levels.max() - levels.min() >= 180:  # angles: wrap
            diff = np.abs((levels.astype(float) - float(target) + 180) % 360 - 180)
        near = levels[int(np.argmin(diff))]
    else:
        near = target
    mus = {lev: X[values == lev].mean(0) for lev in levels}
    delta = np.stack([mus[near] - mus[v] for v in values])
    return delta, near


def controls_for(delta: np.ndarray, kind: str, rng) -> np.ndarray:
    """Same-norm, uninformative versions of each row of delta."""
    if kind == "permuted":
        return np.stack([d[rng.permutation(d.size)] for d in delta])
    if kind == "random":
        g = rng.standard_normal(delta.shape)
        return g * (np.linalg.norm(delta, axis=1, keepdims=True) / np.linalg.norm(g, axis=1, keepdims=True))
    raise ValueError(kind)


def matched_ids(labels: pd.DataFrame, col: str, near, match_cols) -> np.ndarray:
    """For each video, the index of the video with the same `match_cols` values and
    `col == near` (or -1)."""
    cols = [c for c in match_cols if c in labels.columns]
    if not cols:
        return np.full(len(labels), -1)
    key = labels[cols].astype(str).agg("|".join, axis=1)
    hit = labels[col] == near
    lut = dict(zip(key[hit], np.where(hit)[0]))
    return np.array([lut.get(k, -1) for k in key])


def _err(pred, ref):
    d = pred - ref
    return (d ** 2).mean(dim=(1, 2)).numpy(), d.abs().mean(dim=(1, 2)).numpy()


def _ln(x):
    """V-JEPA's prediction target: encoder outputs layer-normalised without affine
    parameters (the training loss compares the predictor with F.layer_norm(h))."""
    import torch.nn.functional as F
    return F.layer_norm(x, (x.shape[-1],))


def ring_shifts(X: np.ndarray, angles_deg: np.ndarray, target_deg: float, harmonics: int = 2,
                source=None):
    """Shift along a fitted ring instead of between condition means. A smooth model
    x ~ mu + sum_k A_k cos(k th) + B_k sin(k th) is fitted by least squares; each video
    is moved from its own angle to the target angle on that model, keeping everything
    the model does not explain (its residual) unchanged. The shift lies in the
    2 * harmonics dimensions of the ring, so it carries no other factor.
    source: (X_src, angles_src) fits the ring on another set of videos (e.g. stepped
    videos) and applies it to these ones; default: fit on X itself."""
    def basis(t):
        return np.stack([f(k * t) for k in range(1, harmonics + 1) for f in (np.cos, np.sin)], -1)
    Xs, angs = (X, angles_deg) if source is None else source
    Fs = basis(np.deg2rad(np.asarray(angs, dtype=float)))
    coef, *_ = np.linalg.lstsq(Fs - Fs.mean(0), Xs - Xs.mean(0), rcond=None)  # [2K, d]
    F = basis(np.deg2rad(np.asarray(angles_deg, dtype=float)))
    tgt = basis(np.full(len(F), np.deg2rad(float(target_deg))))
    return (tgt - F) @ coef


def subspace_shifts(X: np.ndarray, angles_deg, target_deg: float, steered=None, probes=None,
                    max_probes: int = 60, seed: int = 0, device: str = "cpu"):
    """The paper's steering (App. C.11-C.12) as a shift of each activation.

    The orthogonal probe sequence for direction is built on the videos NOT in
    `steered` (80/20 split for its stop rule). Each activation is split into V c + x_perp
    on the span V of the first `probes` probes (all by default), c is replaced by the
    least-squares c* for which every one of those probes reads target_deg, and the
    shift is returned in the activations' own units.

    Also returns the paper's own success measure on the steered videos: a ridge probe
    fitted on their unsteered activations reads the steered ones at the same layer
    (error to the target, before and after)."""
    from .probes import Target
    from .subspace import orthogonal_probe_sequence, ridge_readout, steer
    ang = np.asarray(angles_deg, float)
    th = np.deg2rad(ang)
    Y = np.stack([np.sin(th), np.cos(th)], 1)
    n = len(X)
    steered = np.arange(n) if steered is None else np.asarray(steered)
    fit = np.setdiff1d(np.arange(n), steered)
    if len(fit) < 10:
        fit = np.arange(n)
    perm = np.random.default_rng(seed).permutation(fit)
    nva = max(2, int(round(0.2 * len(perm))))
    sub = lambda i: Target("direction", "circular", Y[i], ang[i])
    seq = orthogonal_probe_sequence(X[perm[nva:]], sub(perm[nva:]), X[perm[:nva]], sub(perm[:nva]),
                                    max_probes=max_probes, min_probes=1, seed=seed, device=device)
    m = seq.n_probes if probes is None else min(int(probes), seq.n_probes)
    Z = seq.scaler(X).astype(np.float64)
    Zs = steer(Z, seq, float(target_deg), m)
    delta = (Zs - Z) * seq.scaler.s
    We, be = ridge_readout(Z[steered], Y[steered])
    read = lambda A: np.rad2deg(np.arctan2(*(A @ We + be).T)) % 360
    err = lambda a: float(np.mean(np.abs((a - target_deg + 180) % 360 - 180)))
    info = {"probes_found": seq.n_probes, "probes_used": m, "dims": 2 * m,
            "layer_read_err_to_target_before": err(read(Z[steered])),
            "layer_read_err_to_target_after": err(read(Zs[steered])),
            "shift_rel_mean": float((np.linalg.norm(delta[steered], axis=1)
                                     / np.linalg.norm(X[steered], axis=1)).mean())}
    return delta, info


def _fit_reader(P, Y):
    from .subspace import IsoScaler, ridge_readout
    sc = IsoScaler(P)
    W, b0 = ridge_readout(sc(P), Y)
    return lambda Q: sc(Q) @ W + b0


def run(enc, ds, cache, layer: int, col: str, target, ids, context: int | None = None,
        n_controls: int = 3, control_kinds=("permuted", "random", "shuffled"),
        match_cols=("speed", "start_id"), batch_size: int = 4, seed: int = 0, frames: int | None = None,
        scales=(1.0,), target_norm: bool = True, method: str = "mean", sizes=None,
        speed_col: str | None = "speed", harmonics: int = 2, ring_source=None,
        probes=None, device: str = "cpu", progress=None) -> pd.DataFrame:
    """One row per (video, condition). Conditions: unperturbed, and for every scale s
    steer@s and <kind>_<k>@s (controls with the norm of the scaled steer).

    method: "mean" (difference of condition means, all dimensions) or "ring" (move
            along a fitted ring of `harmonics` harmonics, see ring_shifts) or
            "subspace" (the paper's many-probe subspace steer, see subspace_shifts; its
            "shuffled" control is not defined and is dropped).
    sizes:  if given, each video's shift is rescaled to this fraction of the norm of
            its own activation (equal relative size across videos, layers and sets);
            the values then replace `scales`.
    controls (each with the norm of the steer, per video):
      permuted  the steer's elements randomly permuted (Thorat et al., arXiv:2111.07898)
      random    a Gaussian direction
      shuffled  a steer built exactly like the real one (same method, same target) but
                from randomly reassigned direction labels: as structured as the steer,
                carrying no information about the video's actual direction"""
    import torch
    rng = np.random.default_rng(seed)
    frames = frames or cache.meta.get("frames", 16)
    Tt = frames // enc.tubelet
    context = context or Tt // 2
    labels = cache.labels
    X = cache.layer(layer).astype(np.float64)
    vals = labels[col].to_numpy()
    delta_all, near = condition_shifts(X, vals, target)
    sub_info = None
    if method == "ring":
        delta_all = ring_shifts(X, vals, float(near), harmonics, ring_source)
    elif method == "subspace":
        delta_all, sub_info = subspace_shifts(X, vals, float(near), steered=ids, probes=probes,
                                              seed=seed, device=device)
        control_kinds = tuple(k for k in control_kinds if k != "shuffled")
    elif method != "mean":
        raise ValueError(f"method {method!r}")
    norms = np.linalg.norm(X, axis=1)

    def build(v):
        if method == "ring":
            return ring_shifts(X, v, float(near), harmonics, ring_source)
        return condition_shifts(X, v, near)[0]
    def build_shuffled():
        S = build(rng.permutation(vals))
        for _ in range(50):  # a video whose shuffled label is the target gets no shift: redraw it
            dead = np.linalg.norm(S, axis=1) < 1e-9
            if not dead.any():
                break
            S[dead] = build(rng.permutation(vals))[dead]
        return S
    shuffled = [build_shuffled() for _ in range(n_controls)] if "shuffled" in control_kinds else []
    if sizes:
        unit = delta_all / np.maximum(np.linalg.norm(delta_all, axis=1, keepdims=True), 1e-12)
        delta_all = unit * norms[:, None]
        scales = sizes
    match = matched_ids(labels, col, near, match_cols)
    ids = np.asarray(ids)
    norm = _ln if target_norm else (lambda x: x)
    rows, pooled = [], []
    future_cache: dict = {}

    def future_of(j, tgt_ids):
        if j not in future_cache:
            v = np.stack([ds.get(int(j), frames)])
            future_cache[j] = norm(enc.predict(v, context)["tokens"][0, tgt_ids])
        return future_cache[j]

    batches = [ids[s:s + batch_size] for s in range(0, len(ids), batch_size)]
    for bi, b in enumerate(batches):
        vids = np.stack([ds.get(int(i), frames) for i in b])
        base = enc.predict(vids, context)
        tgt_ids = base["target_ids"]
        own_future = norm(base["tokens"][:, tgt_ids])
        alt = [future_of(int(match[i]), tgt_ids) if match[i] >= 0 else None for i in b]
        conds = [("unperturbed", "unperturbed", 0.0, None)]
        for sc in scales:
            D = delta_all[b] * sc
            conds.append((f"steer@{sc:g}", "steer", sc, D))
            for kind in control_kinds:
                for k in range(n_controls):
                    if kind == "shuffled":
                        S = shuffled[k][b]
                        C = S * (np.linalg.norm(D, axis=1, keepdims=True)
                                 / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12))
                    else:
                        C = controls_for(D, kind, rng)
                    conds.append((f"{kind}_{k}@{sc:g}", kind, sc, C))
        for name, kind, sc, D in conds:
            if D is None:
                out = base
            else:
                d = torch.as_tensor(D, dtype=torch.float32, device=enc.device)
                out = enc.predict(vids, context, edits={layer: lambda h, d=d: h + d[:, None, :].to(h.dtype)})
            l2, l1 = _err(out["pred"], own_future)
            pm = norm(out["pred"]).mean(1).numpy()
            for r, i in enumerate(b):
                row = {"video": int(i), col: vals[i], "condition": name, "kind": kind, "scale": sc,
                       "l2": float(l2[r]), "l1": float(l1[r]),
                       "shift_rel": 0.0 if D is None else float(np.linalg.norm(D[r]) / norms[i])}
                if speed_col and speed_col in labels.columns:
                    row[speed_col] = labels[speed_col].iloc[i]
                if alt[r] is not None:
                    a2, a1 = _err(out["pred"][r:r + 1], alt[r][None])
                    row.update(l2_to_target=float(a2[0]), l1_to_target=float(a1[0]))
                rows.append(row)
                pooled.append(pm[r])
        if progress:
            progress(bi + 1, len(batches))
    df = pd.DataFrame(rows)
    df["target"] = near
    if sub_info:
        df.attrs["subspace"] = sub_info
    P = np.stack(pooled)
    df.attrs.update(layer=layer, target=near, context_tubelets=context, tubelets=Tt, method=method)
    _decode_predictions(df, P, col, near, speed_col)
    return df, P


def _decode_predictions(df, P, col, near, speed_col=None):
    """Read the direction (and speed) of the *predicted* future. Leave-one-video-out:
    every condition of video v is read by a ridge probe fitted on the unperturbed
    predictions of the other videos, so the reader never saw the video it scores.
    Adds reader_err_deg (unperturbed: how well the reader reads held-out predictions)."""
    base = (df.kind == "unperturbed").to_numpy()
    if not str(col).endswith("_deg") or base.sum() < 8 or df.loc[base, col].nunique() < 3:
        return
    th = np.deg2rad(df[col].to_numpy(dtype=float))
    Y = np.stack([np.sin(th), np.cos(th)], 1)
    spd = (df[speed_col].to_numpy(dtype=float)
           if speed_col and speed_col in df.columns and df[speed_col].nunique() > 1 else None)
    vid = df.video.to_numpy()
    ang = np.full(len(df), np.nan)
    sp = np.full(len(df), np.nan)
    for v in np.unique(vid):
        train = base & (vid != v)
        rows = vid == v
        Q = _fit_reader(P[train], Y[train])(P[rows])
        ang[rows] = np.rad2deg(np.arctan2(Q[:, 0], Q[:, 1])) % 360
        if spd is not None and np.ptp(spd[train]) > 0:
            sp[rows] = _fit_reader(P[train], spd[train, None])(P[rows])[:, 0]
    df["pred_dir_deg"] = ang
    df["pred_err_to_target_deg"] = np.abs((ang - float(near) + 180) % 360 - 180)
    df["pred_err_to_true_deg"] = np.abs((ang - df[col].to_numpy(dtype=float) + 180) % 360 - 180)
    if spd is not None:
        df["pred_speed"] = sp
        base_sp = pd.Series(sp[base], index=vid[base])
        df["pred_speed_change"] = sp - base_sp.reindex(vid).to_numpy()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per scale: mean error per condition kind, functional importance and pull against
    each control (controls averaged over their repeats per video first) and, when
    available, the decoded direction of the predicted future."""
    out = []
    base = df[df.kind == "unperturbed"].set_index("video")
    for sc in sorted(s for s in df.scale.unique() if s > 0):
        part = df[df.scale == sc]
        per = part.groupby(["video", "kind"]).mean(numeric_only=True).reset_index()
        w = per.pivot(index="video", columns="kind")
        kinds = [k for k in w["l2"].columns if k != "steer"]
        for metric in ("l2", "l1"):
            b = base[metric].reindex(w.index)
            r = {"scale": sc, "metric": metric, "shift_rel": w["shift_rel"]["steer"].mean(),
                 "videos": len(w), "unperturbed": b.mean(), "steer": w[metric]["steer"].mean()}
            tm = f"{metric}_to_target"
            if tm in w:
                # absolute approach: steer's error to the target video's future minus the
                # unperturbed one (negative = the steer brings the prediction closer)
                ap = w[tm]["steer"] - base[tm].reindex(w.index)
                r["approach"] = ap.mean()
                r["approach_se"] = ap.std(ddof=1) / np.sqrt(ap.notna().sum())
                ow = w[metric]["steer"] - b
                r["away_from_own"] = ow.mean()
                r["away_from_own_se"] = ow.std(ddof=1) / np.sqrt(ow.notna().sum())
            for kind in kinds:
                r[kind] = w[metric][kind].mean()
                fi = (w[metric]["steer"] - w[metric][kind]) / b
                r[f"importance_vs_{kind}"] = fi.mean()
                r[f"importance_vs_{kind}_se"] = fi.std(ddof=1) / np.sqrt(fi.notna().sum())
                r[f"steer_worse_than_{kind}_frac"] = float((w[metric]["steer"] > w[metric][kind]).mean())
                tm = f"{metric}_to_target"
                if tm in w:
                    bt = base[tm].reindex(w.index)
                    pull = (w[tm][kind] - w[tm]["steer"]) / bt
                    r[f"pull_vs_{kind}"] = pull.mean()
                    r[f"pull_vs_{kind}_se"] = pull.std(ddof=1) / np.sqrt(pull.notna().sum())
            if "pred_err_to_target_deg" in w:
                e = w["pred_err_to_target_deg"]
                r["reader_err_deg"] = base["pred_err_to_true_deg"].reindex(w.index).mean()
                r["pred_err_to_target_deg_unperturbed"] = base["pred_err_to_target_deg"].reindex(w.index).mean()
                r["pred_err_to_target_deg_steer"] = e["steer"].mean()
                for kind in kinds:
                    r[f"pred_err_to_target_deg_{kind}"] = e[kind].mean()
            if "pred_speed_change" in w:
                sb = base.reindex(w.index)
                r["speed_reader_err"] = (sb["pred_speed"] - sb[next(c for c in sb.columns if c == "speed")]).abs().mean() \
                    if "speed" in sb.columns else np.nan
                c = w["pred_speed_change"].abs()  # medians: a linear reader can extrapolate wildly
                r["speed_change_steer"] = c["steer"].median()
                for kind in kinds:
                    r[f"speed_change_{kind}"] = c[kind].median()
            out.append(r)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------- figure

def _ball_xy(video, t):
    """Pixel position of the moving ball in frame t: centroid of the pixels that differ
    most from the clip's per-pixel median (the static background)."""
    v = video.astype(np.float32).mean(-1)
    diff = np.abs(v[t] - np.median(v, 0))
    m = diff > max(diff.max() * 0.5, 1.0)
    if not m.any():
        return v.shape[2] / 2, v.shape[1] / 2
    ys, xs = np.nonzero(m)
    return xs.mean(), ys.mean()


def figure(df: pd.DataFrame, ds, path, scale: float, col: str = "direction_deg", examples: int = 6,
           frames: int = 16, context_frames: int | None = None, seed: int = 0, title: str = ""):
    """Two rows. Top: example videos at the last context frame, with arrows for the true
    motion, the direction decoded from the predicted future without intervention, after
    the steer, and after a same-size control. Bottom: for all videos, where the decoded
    predicted direction points relative to the target, per condition."""
    import matplotlib.pyplot as plt
    from .plots import CAT, INK, MUTED, SURFACE
    if "pred_dir_deg" not in df.columns:
        raise ValueError("no decoded directions in this run (needs an angle variable and >= 8 videos)")
    target = float(df.attrs.get("target", np.nan)) if df.attrs else np.nan
    if not np.isfinite(target):
        target = float(df.loc[df.kind == "steer", "pred_dir_deg"].median())
    part = df[(df.scale == scale) | (df.kind == "unperturbed")]
    if not (part.kind == "steer").any():
        raise ValueError(f"no rows at scale {scale}; scales in this run: {sorted(df.scale.unique())}")
    per = part.groupby(["video", "kind"])["pred_dir_deg"].agg(
        lambda a: np.rad2deg(np.arctan2(np.sin(np.deg2rad(a)).mean(), np.cos(np.deg2rad(a)).mean())) % 360)
    vids = per.index.get_level_values(0).unique().to_numpy()
    rng = np.random.default_rng(seed)
    # examples spread over true directions
    truth = df.drop_duplicates("video").set_index("video")[col]
    pick = []
    for d in rng.permutation(truth.loc[vids].unique()):
        cand = [v for v in vids if truth[v] == d and v not in pick]
        if cand:
            pick.append(int(rng.choice(cand)))
        if len(pick) == examples:
            break
    ctx = context_frames or frames // 2
    kinds = [("unperturbed", "#6f6e69", "model's prediction"), ("steer", CAT[7], f"after steer to {target:g}°")]
    ctrl = next((k for k in ("permuted", "random") if (per.index.get_level_values(1) == k).any()), None)
    if ctrl:
        kinds.append((ctrl, CAT[0], f"after same-size {ctrl} shift"))
    fig = plt.figure(figsize=(2.3 * max(len(pick), 4), 5.6))
    gs = fig.add_gridspec(2, max(len(pick), 4), height_ratios=[1.15, 1], wspace=0.45, hspace=0.45)

    def arrow(ax, x, y, deg, color, lw, L, ls="-"):
        dx, dy = np.cos(np.deg2rad(deg)) * L, -np.sin(np.deg2rad(deg)) * L  # image y points down
        ax.annotate("", (x + dx, y + dy), (x, y), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                    ls=ls, shrinkA=0, shrinkB=0, mutation_scale=12))

    for j, v in enumerate(pick):
        ax = fig.add_subplot(gs[0, j])
        clip = ds.get(int(v), frames)
        f = clip[ctx - 1]
        ax.imshow(f)
        x, y = _ball_xy(clip, ctx - 1)
        L = f.shape[0] * 0.28
        arrow(ax, x, y, truth[v], "#1baf7a", 3.2, L)
        for kind, color, _ in kinds:
            if (v, kind) in per.index:
                arrow(ax, x, y, per[(v, kind)], color, 2.0, L * (0.75 if kind == "unperturbed" else 0.9),
                      ls="--" if kind == ctrl else ":" if kind == "unperturbed" else "-")
        ax.set_title(f"moving {truth[v]:g}°", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    handles = [plt.Line2D([], [], color="#1baf7a", lw=3, label="true motion")] + [
        plt.Line2D([], [], color=c, lw=2, ls="--" if k == ctrl else ":" if k == "unperturbed" else "-", label=lab)
        for k, c, lab in kinds]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    for j, (kind, color, lab) in enumerate(kinds):
        ax = fig.add_subplot(gs[1, j], projection="polar")
        a = per.xs(kind, level=1).to_numpy()
        rel = np.deg2rad((a - target + 180) % 360 - 180)
        ax.hist(rel, bins=np.linspace(-np.pi, np.pi, 25), color=color, alpha=0.85, edgecolor=SURFACE)
        ax.set_theta_zero_location("N")
        ax.set_yticks([])
        ax.set_xticks(np.deg2rad([0, 180]))
        ax.set_xticklabels(["target", "opposite"], fontsize=7, color=MUTED)
        err = np.abs(np.rad2deg(rel)).mean()
        ax.set_title(f"{lab}\nmean error to target {err:.0f}°", fontsize=8.5, color=INK, pad=12)
    fig.suptitle(title, y=1.05, fontsize=11)
    fig.text(0.01, -0.02, "Arrows: direction decoded from the predicted second half of each clip (V-JEPA 2 "
             "predicts in latent space, so this is a probe reading, not a rendered prediction). Bottom: "
             "decoded predicted direction relative to the target, all videos.", fontsize=7.5, color=MUTED)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return pick


def verdict(summary: pd.DataFrame, n_se: float = 2.0, reader_max_deg: float = 45.0) -> pd.DataFrame:
    """Decision rule (L2), per layer and size. PASS requires all of:
      - the reader reads held-out predictions to within reader_max_deg;
      - approach: the steer brings the prediction closer to the real future of the
        matched target-direction video than no steering does (z <= -n_se);
      - pull: closer to that future than every control (z >= n_se).
    Pull alone is not enough: controls that damage the prediction more make a steer
    that damages it less look like a pull. (The first version of this rule used pull
    alone; summaries without the approach columns are judged by pull alone and say so.)"""
    out = []
    for _, r in summary[summary.metric == "l2"].iterrows():
        ctrls = [c[len("pull_vs_"):] for c in r.index if c.startswith("pull_vs_") and not c.endswith("_se")]
        margins = {c: r[f"pull_vs_{c}"] / r[f"pull_vs_{c}_se"] if r[f"pull_vs_{c}_se"] > 0 else np.nan
                   for c in ctrls}
        reader_ok = bool(r.get("reader_err_deg", np.nan) <= reader_max_deg)
        beats = bool(ctrls) and all(m >= n_se for m in margins.values())
        has_ap = "approach" in r.index and pd.notna(r.get("approach"))
        ap_z = r["approach"] / r["approach_se"] if has_ap and r["approach_se"] > 0 else np.nan
        approaches = bool(has_ap and ap_z <= -n_se)
        ok = reader_ok and beats and (approaches if has_ap else True)
        out.append({"layer": r.get("layer"), "size": r.scale, "reader_err_deg": r.get("reader_err_deg"),
                    "reader_ok": reader_ok, "approach_z": ap_z,
                    **{f"pull_z_vs_{c}": m for c, m in margins.items()},
                    "verdict": "NO VERDICT (reader too weak)" if not reader_ok else
                               ("PASS" if ok else "FAIL") + ("" if has_ap else " (pull only)")})
    return pd.DataFrame(out)


# ----------------------------------------------------------- reader cross-check

def recheck(df: pd.DataFrame, P: np.ndarray, col: str, target: float, k: int = 5) -> pd.DataFrame:
    """Re-read the predicted direction with two leave-one-video-out readers fitted on
    the unperturbed predictions of the other videos: the linear ridge reader used by
    `run`, and a non-linear one (k nearest neighbours by cosine similarity, circular
    mean of their directions). Returns, per scale and reader, the held-out reader error
    and the mean error to the target under each condition kind."""
    base = (df.kind == "unperturbed").to_numpy()
    vid = df.video.to_numpy()
    ang_true = df[col].to_numpy(dtype=float)
    th = np.deg2rad(ang_true)
    Y = np.stack([np.sin(th), np.cos(th)], 1)
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    lin = np.full(len(df), np.nan)
    knn = np.full(len(df), np.nan)
    for v in np.unique(vid):
        tr, rows = base & (vid != v), vid == v
        Q = _fit_reader(P[tr], Y[tr])(P[rows])
        lin[rows] = np.rad2deg(np.arctan2(Q[:, 0], Q[:, 1])) % 360
        sim = Pn[rows] @ Pn[tr].T
        nn = np.argsort(-sim, axis=1)[:, :k]
        t = th[tr][nn]
        knn[rows] = np.rad2deg(np.arctan2(np.sin(t).mean(1), np.cos(t).mean(1))) % 360
    out = []
    for name, a in (("linear", lin), (f"knn{k}", knn)):
        err_t = np.abs((a - target + 180) % 360 - 180)
        err_true = np.abs((a - ang_true + 180) % 360 - 180)
        d = df.assign(err_t=err_t)
        for sc in sorted(s for s in df.scale.unique() if s > 0):
            part = d[(d.scale == sc) | (d.kind == "unperturbed")]
            per = part.groupby(["video", "kind"]).err_t.mean().unstack()
            r = {"reader": name, "scale": sc, "reader_err_deg": err_true[base].mean()}
            for kind in per.columns:
                r[f"err_to_target_{kind}"] = per[kind].mean()
            out.append(r)
    return pd.DataFrame(out)


def flip(df: pd.DataFrame, metric: str = "l2") -> pd.DataFrame:
    """Reader-free check. For each video with a matched target-direction video, the gap
    is (error to the target video's real future) - (error to its own real future); it is
    positive when the prediction is closer to its own future. Per scale and condition
    kind: the fraction of predictions that flip (gap < 0, now closer to the target
    video's future), and the paired difference of the steer's flip rate from each
    control's over videos (controls averaged over repeats per video), with its z.
    Also the share of the mean unperturbed gap closed (ratio of means, robust to videos
    whose own gap is near zero)."""
    tm = f"{metric}_to_target"
    if tm not in df:
        return pd.DataFrame()
    d = df[df[tm].notna()].assign(gap=lambda x: x[tm] - x[metric])
    d = d.assign(flip=(d.gap < 0).astype(float))
    g0 = d[d.kind == "unperturbed"].set_index("video").gap
    out = []
    for sc in sorted(s for s in d.scale.unique() if s > 0):
        part = d[d.scale == sc]
        fv = part.groupby(["video", "kind"]).flip.mean().unstack()
        gv = part.groupby(["video", "kind"]).gap.mean().unstack()
        r = {"scale": sc, "metric": metric, "videos": int(len(fv)),
             "flip_unperturbed": float((g0 < 0).mean())}
        for kind in fv.columns:
            r[f"flip_{kind}"] = float(fv[kind].mean())
            r[f"gap_closed_{kind}"] = float(1 - gv[kind].mean() / g0.reindex(gv.index).mean())
        for kind in fv.columns:
            if kind == "steer":
                continue
            diff = fv["steer"] - fv[kind]
            se = diff.std(ddof=1) / np.sqrt(diff.notna().sum())
            r[f"flip_z_vs_{kind}"] = float(diff.mean() / se) if se > 0 else np.nan
        out.append(r)
    return pd.DataFrame(out)


def _ring_basis(angles_deg, harmonics):
    t = np.deg2rad(np.asarray(angles_deg, float))
    return np.stack([f(k * t) for k in range(1, harmonics + 1) for f in (np.cos, np.sin)], -1)


def ring_compare(XA, angA, XB, angB, harmonics: int = 2, folds: int = 5, seed: int = 0) -> dict:
    """How well a direction ring describes each of two activation sets, and whether the
    two rings are the same. Per set: held-out direction-probe error (deg), held-out
    share of variance explained by its own ring, share explained by the other set's
    ring (each set centred on its own mean, ring coefficients fixed, amplitude refitted
    by one scalar), and ring amplitude. Between rings: principal angles of their
    2*harmonics-dimensional spans, against random subspaces of that size."""
    from .subspace import principal_angles
    rng = np.random.default_rng(seed)

    def fit(X, a):
        F = _ring_basis(a, harmonics)
        return np.linalg.lstsq(F - F.mean(0), X - X.mean(0), rcond=None)[0]  # [2h, d]

    def stats(X, a):
        a = np.asarray(a, float)
        th = np.deg2rad(a)
        Y = np.stack([np.sin(th), np.cos(th)], 1)
        idx = rng.permutation(len(X)) % folds
        err, sse, sst = [], 0.0, 0.0
        for f in range(folds):
            tr, te = idx != f, idx == f
            Q = _fit_reader(X[tr], Y[tr])(X[te])
            pa = np.rad2deg(np.arctan2(Q[:, 0], Q[:, 1])) % 360
            err.append(np.abs((pa - a[te] + 180) % 360 - 180))
            C = fit(X[tr], a[tr])
            F = _ring_basis(a, harmonics)
            pred = (F[te] - F[tr].mean(0)) @ C + X[tr].mean(0)
            sse += ((X[te] - pred) ** 2).sum()
            sst += ((X[te] - X[tr].mean(0)) ** 2).sum()
        return float(np.concatenate(err).mean()), float(1 - sse / sst)

    def explained_by(X, a, C):
        F = _ring_basis(a, harmonics)
        G = (F - F.mean(0)) @ C
        Xc = X - X.mean(0)
        k = float((G * Xc).sum() / max((G * G).sum(), 1e-12))
        return float(1 - ((Xc - k * G) ** 2).sum() / (Xc ** 2).sum()), k

    CA, CB = fit(XA, angA), fit(XB, angB)
    QA, QB = np.linalg.qr(CA.T)[0], np.linalg.qr(CB.T)[0]
    ang = principal_angles(QA, QB)
    d, m = QA.shape
    rand = [principal_angles(np.linalg.qr(rng.standard_normal((d, m)))[0],
                             np.linalg.qr(rng.standard_normal((d, m)))[0]).mean() for _ in range(50)]
    out = {}
    for name, X, a, Cown, Coth in (("a", XA, angA, CA, CB), ("b", XB, angB, CB, CA)):
        e, r2 = stats(X, a)
        r2o, k = explained_by(X, a, Coth)
        r2s, _ = explained_by(X, a, Cown)
        out.update({f"{name}_probe_err_deg": e, f"{name}_ring_r2_heldout": r2,
                    f"{name}_ring_r2_own_fit": r2s, f"{name}_ring_r2_other_ring": r2o,
                    f"{name}_other_ring_gain": k,
                    f"{name}_ring_amplitude_rel": float(np.linalg.norm(Cown) / np.sqrt((X - X.mean(0)) .var(0).sum()))})
    out.update({"ring_dims": m, "principal_angles_deg": [round(float(x), 1) for x in ang],
                "mean_angle_deg": float(ang.mean()), "mean_angle_random_deg": float(np.mean(rand))})
    return out
