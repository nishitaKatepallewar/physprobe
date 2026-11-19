"""Subspace geometry: orthogonal probe sequences, principal angles, steering.

Orthogonal probe sequence (paper App. C.11): train a linear probe, project its
weight directions out of the activations, train again on what is left, and repeat
until the probe drops below a threshold. The number of removed dimensions is the
paper's estimate of how many dimensions carry the variable; the removed directions
together span the variable's subspace.

Everything here works in a common, geometry-preserving space: activations are
centred and divided by ONE scalar (not z-scored per dimension), so subspaces from
different tasks and datasets can be compared by principal angles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .probes import Target, circ_mae, r2_score

# Stop thresholds from the paper (App. C.11).
THRESHOLDS = {"circular": 0.1, "scalar": 0.05, "vector": 0.05, "binary": 0.55}
EPOCHS = {"circular": 100, "scalar": 50, "vector": 50, "binary": 50}


class IsoScaler:
    """Centre and scale by a single scalar: an isotropic map, so angles are preserved."""

    def __init__(self, X):
        self.mu = X.mean(0, keepdims=True)
        self.s = float(X.std(0).mean()) + 1e-8

    def __call__(self, X):
        return (X - self.mu) / self.s

    def inverse(self, Z):
        return Z * self.s + self.mu


def train_probe(X, Y, binary, epochs, lr=1e-3, wd=1e-4, batch_size=32, seed=0, device="cpu"):
    """Adam + L2 (the paper's 'Adam, lr 1e-3, weight decay 1e-4'). Weights start at
    zero, so they stay inside the span of the (already deflated) training data."""
    g = torch.Generator().manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    Y = torch.as_tensor(Y, dtype=torch.float32, device=device)
    W = torch.zeros(X.shape[1], Y.shape[1], device=device, requires_grad=True)
    b = torch.zeros(Y.shape[1], device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        for s in range(0, n, batch_size):
            i = perm[s:s + batch_size]
            p = X[i] @ W + b
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(p, Y[i]) if binary
                    else torch.nn.functional.mse_loss(p, Y[i]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return W.detach().cpu().numpy().astype(np.float64), b.detach().cpu().numpy().astype(np.float64)


def _metric(kind, Y, pred):
    if kind == "binary":
        return float(((pred[:, 0] > 0) == (Y[:, 0] > 0.5)).mean())
    return r2_score(Y, pred)


@dataclass
class ProbeSequence:
    kind: str
    scaler: IsoScaler
    W: list = field(default_factory=list)  # each [d, k], in the scaled space
    b: list = field(default_factory=list)  # each [k]
    metrics: list = field(default_factory=list)  # validation metric of every probe tried
    Q: np.ndarray | None = None  # [d, dim] orthonormal basis of the removed subspace

    @property
    def n_probes(self) -> int:
        return len(self.W)

    @property
    def dim(self) -> int:
        return 0 if self.Q is None else self.Q.shape[1]

    def basis(self, m: int | None = None) -> np.ndarray:
        """Orthonormal basis spanned by the first m probes."""
        Ws = np.concatenate(self.W[:m], axis=1)
        return np.linalg.qr(Ws)[0]


def orthogonal_probe_sequence(X_tr, t_tr: Target, X_va, t_va: Target, max_probes=200,
                              threshold=None, epochs=None, lr=1e-3, wd=1e-4, batch_size=32,
                              seed=0, device="cpu", min_probes=0, verbose=False) -> ProbeSequence:
    kind = t_tr.kind
    binary = kind == "binary"
    thr = THRESHOLDS[kind] if threshold is None else threshold
    ep = EPOCHS[kind] if epochs is None else epochs
    sc = IsoScaler(X_tr)
    A, B = sc(X_tr).astype(np.float64), sc(X_va).astype(np.float64)
    seq = ProbeSequence(kind, sc)
    Qs = []
    # Regression targets other than (sin, cos) are standardised for training, so one
    # learning rate suits every variable; weights are mapped back to target units.
    if kind in ("scalar", "vector"):
        ymu, ysd = t_tr.Y.mean(0, keepdims=True), t_tr.Y.std(0, keepdims=True) + 1e-8
    else:
        ymu, ysd = np.zeros((1, t_tr.Y.shape[1])), np.ones((1, t_tr.Y.shape[1]))
    Ytr = (t_tr.Y - ymu) / ysd
    for k in range(max_probes):
        W, b = train_probe(A, Ytr, binary, ep, lr, wd, batch_size, seed + k, device)
        W, b = W * ysd, b * ysd[0] + ymu[0]
        if Qs:  # numerical hygiene: keep the new probe exactly orthogonal to earlier ones
            Qprev = np.concatenate(Qs, 1)
            W = W - Qprev @ (Qprev.T @ W)
        m = _metric(kind, t_va.Y, B @ W + b)
        seq.metrics.append(m)
        if verbose:
            print(f"  probe {k:3d}: val {'acc' if binary else 'R2'} = {m:.3f}")
        if m < thr and k >= min_probes:
            break
        Qk = np.linalg.qr(W)[0]
        A = A - (A @ Qk) @ Qk.T
        B = B - (B @ Qk) @ Qk.T
        seq.W.append(W)
        seq.b.append(b)
        Qs.append(Qk)
    if Qs:
        seq.Q = np.linalg.qr(np.concatenate(Qs, 1))[0]
    return seq


# --------------------------------------------------------------- subspace overlap


def principal_angles(QA: np.ndarray, QB: np.ndarray) -> np.ndarray:
    """Principal angles (degrees, ascending) between the column spaces of QA and QB."""
    s = np.linalg.svd(QA.T @ QB, compute_uv=False)
    return np.rad2deg(np.arccos(np.clip(s, -1.0, 1.0)))


def overlap(QA: np.ndarray, QB: np.ndarray) -> dict:
    """Paper App. C.4: fraction of B's subspace captured by A, vs. a random-subspace
    baseline dim(A)/d. Also the chance mean principal angle, estimated by sampling."""
    d = QA.shape[0]
    ang = principal_angles(QA, QB)
    ov = float(np.linalg.norm(QA.T @ QB) ** 2 / QB.shape[1])
    rng = np.random.default_rng(0)
    rand = [principal_angles(np.linalg.qr(rng.standard_normal((d, QA.shape[1])))[0],
                             np.linalg.qr(rng.standard_normal((d, QB.shape[1])))[0]).mean()
            for _ in range(20)]
    return {"dim_a": QA.shape[1], "dim_b": QB.shape[1], "mean_angle_deg": float(ang.mean()),
            "min_angle_deg": float(ang.min()), "overlap": ov,
            "overlap_random": QA.shape[1] / d, "mean_angle_random_deg": float(np.mean(rand))}


# ---------------------------------------------------------------------- steering


def steer(Z: np.ndarray, seq: ProbeSequence, target_deg: float, m: int | None = None):
    """Subspace steering (paper App. C.12) for a circular variable.

    Z: activations already in seq.scaler's space, [n, d].
    Using the first m probes: V = orthonormal basis of their weights; split each
    activation into V c + x_perp; replace c by the least-squares c* for which every
    one of the m probes reads out target_deg; return V c* + x_perp.
    """
    m = seq.n_probes if m is None else m
    Ws = np.concatenate(seq.W[:m], axis=1)  # [d, 2m], columns ordered (sin, cos) per probe
    bs = np.concatenate(seq.b[:m])
    V = np.linalg.qr(Ws)[0]
    th = np.deg2rad(target_deg)
    y = np.tile([np.sin(th), np.cos(th)], m) - bs
    c_star = np.linalg.lstsq(Ws.T @ V, y, rcond=None)[0]
    Z_perp = Z - (Z @ V) @ V.T
    return Z_perp + c_star[None] @ V.T


def ridge_readout(Z, Y, alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """Ridge probe in dual form (n << d), penalty chosen by leave-one-out error.
    Used as the independent 'evaluation probe'. Returns W [d, k], b [k]."""
    mu, ymu = Z.mean(0, keepdims=True), Y.mean(0, keepdims=True)
    Zc, Yc = Z - mu, Y - ymu
    lam, V = np.linalg.eigh(Zc @ Zc.T)
    lam = np.clip(lam, 0, None)
    scale = lam.mean() + 1e-12
    VtY = V.T @ Yc
    best = None
    for a in alphas:
        shrink = lam / (lam + a * scale)
        resid = Yc - V @ (shrink[:, None] * VtY)
        hii = np.clip((V ** 2) @ shrink, 0, 1 - 1e-6)
        loo = float(((resid / (1 - hii)[:, None]) ** 2).mean())
        if best is None or loo < best[0]:
            best = (loo, a)
    coef = V @ (VtY / (lam + best[1] * scale)[:, None])
    W = Zc.T @ coef
    return W, (ymu - mu @ W)[0]


# ------------------------------------------------------- low-dimensional controls
#
# A linear probe can only use the cross-covariance between activations and the
# target. For a k-column target that covariance has rank <= k, so projecting out
# those k directions already makes the target linearly undecodable (the idea behind
# LEACE, Belrose et al. 2023). An orthogonal probe sequence removes probe *weight*
# directions instead, which generally differ from the covariance directions when
# the noise is anisotropic, so it can keep finding signal for many rounds even when
# the signal itself lives in k dimensions (see `selftest`). These controls give the
# low-dimensional side of the story to set against the probe-sequence count.


def cov_subspace(Z, Y) -> np.ndarray:
    """Orthonormal basis [d, k] of the activation/target cross-covariance."""
    C = (Z - Z.mean(0, keepdims=True)).T @ (Y - Y.mean(0, keepdims=True))
    return np.linalg.qr(C)[0]


def erasure_check(Z_tr, Y_tr, Z_va, Y_va, kind: str) -> dict:
    """Probe score on validation rows before and after removing the rank-k
    covariance subspace estimated on training rows."""
    Q = cov_subspace(Z_tr, Y_tr)
    out = {}
    for name, (A, B) in {"before": (Z_tr, Z_va),
                         "after": (Z_tr - (Z_tr @ Q) @ Q.T, Z_va - (Z_va @ Q) @ Q.T)}.items():
        W, b = ridge_readout(A, Y_tr)
        pred = B @ W + b
        if kind == "binary":
            out[name] = float(((pred[:, 0] > 0.5) == (Y_va[:, 0] > 0.5)).mean())
        else:
            out[name] = r2_score(Y_va, pred)
    out["dims_removed"] = Q.shape[1]
    return out


def class_means(Z, angle_deg):
    levels = np.unique(angle_deg)
    return levels, np.stack([Z[angle_deg == a].mean(0) for a in levels])


def mean_shift_basis(mus: np.ndarray, k: int | None = None) -> np.ndarray:
    """Orthonormal basis [d, k] of the top-k principal axes of the condition means
    (k = None: all of them, at most n_levels - 1)."""
    U = np.linalg.svd((mus - mus.mean(0, keepdims=True)).T, full_matrices=False)[0]
    return U[:, :(len(mus) - 1 if k is None else k)]


def steer_mean_shift(Z, src_level_idx, mus, target_idx, Q=None):
    """Difference-of-means steering: add (mean of target condition - mean of the
    video's own condition), optionally restricted to span(Q). Every accurate linear
    read-out must respond to this shift, because it is what separates the conditions.
    The intervention has rank <= n_levels - 1 (2 for a planar ring)."""
    delta = mus[target_idx][None] - mus[src_level_idx]
    if Q is not None:
        delta = (delta @ Q) @ Q.T
    return Z + delta


def steering_experiment(X, target: Target, target_deg=90.0, train_frac=0.7, ms=None,
                        strict=False, seed=0, device="cpu", max_probes=60, min_probes=0,
                        verbose=False) -> dict:
    """Paper protocol: build the probe sequence on a 70% split, steer the 30% split
    toward target_deg, and read the result with a probe trained only on that 30%.

    strict=True instead trains the evaluation probe on half of the held-out split
    and steers the other half, so the evaluation probe has never seen the
    activations it is scoring.

    Also reported: a difference-of-means control (see `steer_mean_shift`), read by
    the same evaluation probe, for comparison with the many-probe subspace steer.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)
    ntr = int(round(train_frac * n))
    tr, te = perm[:ntr], perm[ntr:]
    sub = lambda idx: Target(target.name, target.kind, target.Y[idx], target.angle[idx])
    # the sequence needs its own validation rows for the stop rule: 80/20 inside train
    nva = int(round(0.2 * len(tr)))
    seq = orthogonal_probe_sequence(X[tr[nva:]], sub(tr[nva:]), X[tr[:nva]], sub(tr[:nva]),
                                    max_probes=max_probes, min_probes=min_probes, seed=seed,
                                    device=device, verbose=verbose)
    Zte = seq.scaler(X[te]).astype(np.float64)
    if strict:
        h = len(te) // 2
        fit_idx, ev_idx = np.arange(h), np.arange(h, len(te))
    else:
        fit_idx = ev_idx = np.arange(len(te))
    We, be = ridge_readout(Zte[fit_idx], target.Y[te][fit_idx])
    read = lambda Z: Z @ We + be
    ang = target.angle[te][ev_idx]
    base_pred = read(Zte[ev_idx])
    tgt = np.full(len(ev_idx), target_deg)
    res = {"n_probes": seq.n_probes, "dim": seq.dim, "seq_metrics": seq.metrics,
           "eval_probe_r2": r2_score(target.Y[te][ev_idx], base_pred),
           "eval_probe_mae_deg": circ_mae(ang, base_pred),
           "baseline_mae_to_target_deg": circ_mae(tgt, base_pred), "curve": [], "seq": seq}
    # Low-dimensional control, fitted on the same training rows as the probe sequence:
    # shift each held-out activation by the difference of condition means.
    Ztr = seq.scaler(X[tr]).astype(np.float64)
    levels, mus = class_means(Ztr, target.angle[tr])
    near = int(np.argmin(np.abs((levels - target_deg + 180) % 360 - 180)))
    src = np.searchsorted(levels, target.angle[te][ev_idx])
    controls = {}
    for name, k in (("mean shift, top-2 axes", 2), ("mean shift, all axes", None)):
        Q = mean_shift_basis(mus, k)
        Zs = steer_mean_shift(Zte[ev_idx], src, mus, near, Q)
        controls[name] = {"dims": int(Q.shape[1]), "mae_to_target_deg": circ_mae(tgt, read(Zs)),
                          "relative_shift": float(np.linalg.norm(Zs - Zte[ev_idx], axis=1).mean()
                                                  / np.linalg.norm(Zte[ev_idx], axis=1).mean())}
    res["controls"] = controls
    res["control_target_deg"] = float(levels[near])
    res["levels"], res["class_means"] = levels, mus
    if seq.n_probes == 0:
        return res
    ms = ms or sorted({1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, seq.n_probes})
    for m in [m for m in ms if m <= seq.n_probes]:
        Zs = steer(Zte[ev_idx], seq, target_deg, m)
        pred = read(Zs)
        res["curve"].append({"n_probes": m, "dims": 2 * m,
                             "mae_to_target_deg": circ_mae(tgt, pred),
                             "relative_shift": float(np.linalg.norm(Zs - Zte[ev_idx], axis=1).mean()
                                                     / np.linalg.norm(Zte[ev_idx], axis=1).mean())})
    return res
