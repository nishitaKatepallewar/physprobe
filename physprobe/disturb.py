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


def run(enc, ds, cache, layer: int, col: str, target, ids, context: int | None = None,
        n_controls: int = 3, control_kinds=("permuted", "random"), match_cols=("speed", "start_id"),
        batch_size: int = 4, seed: int = 0, frames: int | None = None, scales=(1.0,),
        target_norm: bool = True, progress=None) -> pd.DataFrame:
    """One row per (video, condition). Conditions: unperturbed, and for every scale s
    steer@s and <kind>_<k>@s (controls with the norm of the scaled steer)."""
    import torch
    rng = np.random.default_rng(seed)
    frames = frames or cache.meta.get("frames", 16)
    Tt = frames // enc.tubelet
    context = context or Tt // 2
    labels = cache.labels
    X = cache.layer(layer).astype(np.float64)
    vals = labels[col].to_numpy()
    delta_all, near = condition_shifts(X, vals, target)
    match = matched_ids(labels, col, near, match_cols)
    ids = np.asarray(ids)
    norms = np.linalg.norm(X, axis=1)
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
                    conds.append((f"{kind}_{k}@{sc:g}", kind, sc, controls_for(D, kind, rng)))
        for name, kind, sc, D in conds:
            if D is None:
                out = base
            else:
                d = torch.as_tensor(D, dtype=torch.float32, device=enc.device)
                out = enc.predict(vids, context, edits={layer: lambda h, d=d: h + d[:, None, :].to(h.dtype)})
            l2, l1 = _err(out["pred"], own_future)
            pm = out["pred"].mean(1).numpy()
            for r, i in enumerate(b):
                row = {"video": int(i), col: vals[i], "condition": name, "kind": kind, "scale": sc,
                       "l2": float(l2[r]), "l1": float(l1[r]),
                       "shift_rel": 0.0 if D is None else float(np.linalg.norm(D[r]) / norms[i])}
                if alt[r] is not None:
                    a2, a1 = _err(out["pred"][r:r + 1], alt[r][None])
                    row.update(l2_to_target=float(a2[0]), l1_to_target=float(a1[0]))
                rows.append(row)
                pooled.append(pm[r])
        if progress:
            progress(bi + 1, len(batches))
    df = pd.DataFrame(rows)
    _decode_predicted_direction(df, np.stack(pooled), col, near)
    df.attrs.update(layer=layer, target=near, context_tubelets=context, tubelets=Tt)
    return df


def _decode_predicted_direction(df, P, col, near):
    """Read the direction of the *predicted* future: a ridge probe fitted on the
    unperturbed predictions (mean over predicted tokens) against the true direction,
    applied to every condition. Only for angle variables with enough videos."""
    from .subspace import IsoScaler, ridge_readout
    base = (df.kind == "unperturbed").to_numpy()
    if not str(col).endswith("_deg") or base.sum() < 8 or df.loc[base, col].nunique() < 3:
        return
    th = np.deg2rad(df[col].to_numpy(dtype=float))
    Y = np.stack([np.sin(th), np.cos(th)], 1)
    sc = IsoScaler(P[base])
    W, b0 = ridge_readout(sc(P[base]), Y[base])
    Q = sc(P) @ W + b0
    ang = np.rad2deg(np.arctan2(Q[:, 0], Q[:, 1])) % 360
    df["pred_dir_deg"] = ang
    df["pred_err_to_target_deg"] = np.abs((ang - float(near) + 180) % 360 - 180)
    df["pred_err_to_true_deg"] = np.abs((ang - df[col].to_numpy(dtype=float) + 180) % 360 - 180)


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
                r["pred_err_to_target_deg_unperturbed"] = base["pred_err_to_target_deg"].reindex(w.index).mean()
                r["pred_err_to_target_deg_steer"] = e["steer"].mean()
                for kind in kinds:
                    r[f"pred_err_to_target_deg_{kind}"] = e[kind].mean()
            out.append(r)
    return pd.DataFrame(out)
