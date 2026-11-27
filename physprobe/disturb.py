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


def ring_shifts(X: np.ndarray, angles_deg: np.ndarray, target_deg: float, harmonics: int = 2):
    """Shift along a fitted ring instead of between condition means. A smooth model
    x ~ mu + sum_k A_k cos(k th) + B_k sin(k th) is fitted by least squares over all
    videos; each video is moved from its own angle to the target angle on that model,
    keeping everything the model does not explain (its residual) unchanged. The shift
    lies in the 2 * harmonics dimensions of the ring, so it carries no other factor."""
    th = np.deg2rad(angles_deg.astype(float))
    def basis(t):
        return np.stack([f(k * t) for k in range(1, harmonics + 1) for f in (np.cos, np.sin)], -1)
    F = basis(th)
    Fc = F - F.mean(0)
    coef, *_ = np.linalg.lstsq(Fc, X - X.mean(0), rcond=None)  # [2K, d]
    tgt = basis(np.full(len(th), np.deg2rad(float(target_deg))))
    return (tgt - F) @ coef


def _fit_reader(P, Y):
    from .subspace import IsoScaler, ridge_readout
    sc = IsoScaler(P)
    W, b0 = ridge_readout(sc(P), Y)
    return lambda Q: sc(Q) @ W + b0


def run(enc, ds, cache, layer: int, col: str, target, ids, context: int | None = None,
        n_controls: int = 3, control_kinds=("permuted", "random", "shuffled"),
        match_cols=("speed", "start_id"), batch_size: int = 4, seed: int = 0, frames: int | None = None,
        scales=(1.0,), target_norm: bool = True, method: str = "mean", sizes=None,
        speed_col: str | None = "speed", harmonics: int = 2, progress=None) -> pd.DataFrame:
    """One row per (video, condition). Conditions: unperturbed, and for every scale s
    steer@s and <kind>_<k>@s (controls with the norm of the scaled steer).

    method: "mean" (difference of condition means, all dimensions) or "ring" (move
            along a fitted ring of `harmonics` harmonics, see ring_shifts).
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
    if method == "ring":
        delta_all = ring_shifts(X, vals, float(near), harmonics)
    elif method != "mean":
        raise ValueError(f"method {method!r}")
    norms = np.linalg.norm(X, axis=1)

    def build(v):
        if method == "ring":
            return ring_shifts(X, v, float(near), harmonics)
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
    """The decision rule fixed before the runs (primary measure: pull, L2). Per layer
    and size: PASS if the reader reads held-out predictions to within reader_max_deg
    and the steer's pull exceeds every control's by at least n_se standard errors."""
    out = []
    for _, r in summary[summary.metric == "l2"].iterrows():
        ctrls = [c[len("pull_vs_"):] for c in r.index if c.startswith("pull_vs_") and not c.endswith("_se")]
        margins = {c: r[f"pull_vs_{c}"] / r[f"pull_vs_{c}_se"] if r[f"pull_vs_{c}_se"] > 0 else np.nan
                   for c in ctrls}
        reader_ok = bool(r.get("reader_err_deg", np.nan) <= reader_max_deg)
        beats = bool(ctrls) and all(m >= n_se for m in margins.values())
        out.append({"layer": r.get("layer"), "size": r.scale, "reader_err_deg": r.get("reader_err_deg"),
                    "reader_ok": reader_ok, **{f"pull_z_vs_{c}": m for c, m in margins.items()},
                    "verdict": "PASS" if (reader_ok and beats) else
                               "NO VERDICT (reader too weak)" if not reader_ok else "FAIL"})
    return pd.DataFrame(out)
