"""Single-unit tuning (paper App. C.7 / C.8).

Direction: per-unit GLM  y = b0 + b_cos cos(theta) + b_sin sin(theta), ridge 1e-3,
5-fold cross-validated R^2 gain over an intercept-only model. Preferred direction
= atan2(b_sin, b_cos); tuning gain = sqrt(b_cos^2 + b_sin^2).

Speed: per-unit quadratic GLM  y = b0 + b_r r + b_r2 r^2; preferred speed
= -b_r / (2 b_r2) where the parabola opens downward, clipped to the observed range.

All units are fitted at once in closed form.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cv_delta_r2(D: np.ndarray, Y: np.ndarray, alpha=1e-3, n_splits=5, seed=0) -> np.ndarray:
    """Cross-validated R^2 of ridge(D -> Y) for every column of Y, measured against
    the training-fold mean (so an intercept-only model scores exactly 0)."""
    n = len(D)
    order = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(order, n_splits)
    ss_res = np.zeros(Y.shape[1])
    ss_tot = np.zeros(Y.shape[1])
    for te in folds:
        tr = np.setdiff1d(order, te)
        dmu, ymu = D[tr].mean(0), Y[tr].mean(0)
        Dc = D[tr] - dmu
        beta = np.linalg.solve(Dc.T @ Dc + alpha * np.eye(D.shape[1]), Dc.T @ (Y[tr] - ymu))
        pred = (D[te] - dmu) @ beta + ymu
        ss_res += ((Y[te] - pred) ** 2).sum(0)
        ss_tot += ((Y[te] - ymu) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-20)


def _fit(D, Y, alpha=1e-3):
    dmu, ymu = D.mean(0), Y.mean(0)
    Dc = D - dmu
    return np.linalg.solve(Dc.T @ Dc + alpha * np.eye(D.shape[1]), Dc.T @ (Y - ymu))


def _zscore(A):
    A = np.asarray(A, dtype=np.float64)
    return (A - A.mean(0)) / (A.std(0) + 1e-8)


def direction_tuning(acts: np.ndarray, angle_deg: np.ndarray, alpha=1e-3) -> pd.DataFrame:
    """acts [N, units] -> per-unit delta_r2, pref_deg, gain (in units of the unit's SD)."""
    Y = _zscore(acts)
    th = np.deg2rad(angle_deg)
    D = np.stack([np.cos(th), np.sin(th)], 1)
    beta = _fit(D, Y, alpha)
    return pd.DataFrame({"unit": np.arange(Y.shape[1]),
                         "delta_r2": _cv_delta_r2(D, Y, alpha),
                         "pref_deg": np.rad2deg(np.arctan2(beta[1], beta[0])) % 360,
                         "gain": np.hypot(beta[0], beta[1])})


def speed_tuning(acts: np.ndarray, speed: np.ndarray, alpha=1e-3) -> pd.DataFrame:
    Y = _zscore(acts)
    r = np.asarray(speed, dtype=np.float64)
    D = np.stack([r, r ** 2], 1)
    beta = _fit(D, Y, alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        pref = -beta[0] / (2 * beta[1])
    pref = np.where(beta[1] < 0, np.clip(pref, r.min(), r.max()), np.nan)
    return pd.DataFrame({"unit": np.arange(Y.shape[1]), "delta_r2": _cv_delta_r2(D, Y, alpha),
                         "pref_speed": pref, "b_r": beta[0], "b_r2": beta[1]})


def tiling_row(tun: pd.DataFrame, col="pref_deg", bins=24, lo=0.0, hi=360.0) -> np.ndarray:
    """Max delta_r2 among units whose preferred value falls in each bin (one heatmap row).
    Empty bins are NaN: a gap means no unit prefers that direction."""
    v = tun[col].to_numpy()
    ok = np.isfinite(v)
    idx = np.clip(((v[ok] - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
    row = np.full(bins, np.nan)
    d = tun["delta_r2"].to_numpy()[ok]
    for b in range(bins):
        if (idx == b).any():
            row[b] = d[idx == b].max()
    return row


def tuning_curves(acts: np.ndarray, angle_deg: np.ndarray, units: np.ndarray):
    """Mean z-scored response of selected units at each distinct angle -> (angles, [units, n_angles])."""
    Y = _zscore(acts[:, units])
    angs = np.unique(angle_deg)
    return angs, np.stack([Y[angle_deg == a].mean(0) for a in angs], 1)
