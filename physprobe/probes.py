"""Linear probes with grouped cross-validation (paper Sec. 3.2, App. B).

Targets come in four kinds:
    circular  angle in degrees  -> regress (sin, cos), report R^2 and circular MAE
    scalar    one number        -> regress, report R^2
    vector    several columns   -> regress jointly, report R^2
    binary    0/1               -> logistic regression, report accuracy

Two solvers:
    adamw  the paper's recipe: AdamW, 5 learning rates x 4 weight decays = 20
           configurations, the best chosen on a validation split carved out of each
           training fold. All 20 are trained at once as one batched model.
    ridge  closed-form ridge regression (on +/-1 labels for binary targets) with the
           penalty chosen on the same kind of validation split. Much faster.
           Much faster; use it to check that a result is not an optimiser artefact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

PAPER_SWEEP = [(lr, wd) for lr in (1e-4, 3e-4, 1e-3, 3e-3, 5e-3) for wd in (0.01, 0.1, 0.4, 0.8)]


# ---------------------------------------------------------------------- targets


@dataclass
class Target:
    name: str
    kind: str  # circular | scalar | vector | binary
    Y: np.ndarray  # [N, k] regression targets or [N, 1] 0/1
    angle: np.ndarray | None = None  # degrees, for circular


def make_target(labels: pd.DataFrame, cols: str, kind: str) -> Target:
    if kind == "circular":
        th = np.deg2rad(labels[cols].to_numpy(dtype=float))
        return Target(cols, kind, np.stack([np.sin(th), np.cos(th)], 1), np.rad2deg(th) % 360)
    if kind == "binary":
        y = labels[cols].to_numpy()
        u = np.unique(y)
        if len(u) != 2:
            raise ValueError(f"{cols!r} is not binary: {u}")
        return Target(cols, kind, (y == u[1]).astype(np.float64)[:, None])
    names = cols.split(",")
    return Target(cols, "vector" if len(names) > 1 else "scalar",
                  labels[names].to_numpy(dtype=float))


def parse_targets(specs, labels: pd.DataFrame, defaults: dict) -> list[Target]:
    """specs like ['direction_deg:circular', 'speed:scalar']; empty -> dataset defaults."""
    pairs = [s.rsplit(":", 1) for s in specs] if specs else list(defaults.items())
    return [make_target(labels, c, k) for c, k in pairs]


# ---------------------------------------------------------------------- metrics


def r2_score(y, p) -> float:
    ss_res = ((y - p) ** 2).sum()
    ss_tot = ((y - y.mean(0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def circ_mae(angle_deg, pred_sincos) -> float:
    pa = np.rad2deg(np.arctan2(pred_sincos[:, 0], pred_sincos[:, 1]))
    d = np.abs((pa - angle_deg + 180.0) % 360.0 - 180.0)
    return float(d.mean())


def score(t: Target, idx, pred) -> dict:
    """pred: regression outputs, or logits for binary."""
    y = t.Y[idx]
    if t.kind == "binary":
        return {"accuracy": float(((pred[:, 0] > 0) == (y[:, 0] > 0.5)).mean())}
    out = {"r2": r2_score(y, pred)}
    if t.kind == "circular":
        out["circ_mae_deg"] = circ_mae(t.angle[idx], pred)
    return out


def main_metric(kind: str) -> str:
    return "accuracy" if kind == "binary" else "r2"


# ---------------------------------------------------------------------- fitting


class Scaler:
    """z-score features with training-fold statistics."""

    def __init__(self, X):
        self.mu = X.mean(0, keepdims=True)
        self.sd = X.std(0, keepdims=True) + 1e-6

    def __call__(self, X):
        return (X - self.mu) / self.sd


def _fit_adamw_sweep(Xtr, Ytr, binary, sweep, epochs, batch_size, device, seed):
    """Train one linear probe per (lr, wd) config, all in parallel.

    Returns W [C, d, k], b [C, k]. The configs share nothing: the summed loss has
    no cross terms, and AdamW (decoupled weight decay, as in torch.optim.AdamW) is
    applied with a per-config learning rate and decay. Biases are not decayed.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    Y = torch.as_tensor(Ytr, dtype=torch.float32, device=device)
    n, d = X.shape
    k, C = Y.shape[1], len(sweep)
    lr = torch.tensor([s[0] for s in sweep], device=device).view(C, 1, 1)
    wd = torch.tensor([s[1] for s in sweep], device=device).view(C, 1, 1)
    W = torch.zeros(C, d, k, device=device)
    b = torch.zeros(C, 1, k, device=device)
    mW, vW, mb, vb = (torch.zeros_like(t) for t in (W, W, b, b))
    b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
    bsz = n if batch_size is None else min(batch_size, n)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        for s in range(0, n, bsz):
            xb, yb = X[perm[s:s + bsz]], Y[perm[s:s + bsz]]
            pred = torch.einsum("nd,cdk->cnk", xb, W) + b
            # d(loss)/d(pred) for a mean over (n, k): MSE -> 2 (pred - y); BCE -> sigmoid(pred) - y
            r = (torch.sigmoid(pred) - yb if binary else 2.0 * (pred - yb)) / (len(xb) * k)
            gW, gb = torch.einsum("nd,cnk->cdk", xb, r), r.sum(1, keepdim=True)
            step += 1
            for p, gr, m, v, decay in ((W, gW, mW, vW, wd), (b, gb, mb, vb, None)):
                m.mul_(b1).add_(gr, alpha=1 - b1)
                v.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                if decay is not None:
                    p.mul_(1 - lr * decay)
                p.sub_(lr * (m / (1 - b1 ** step)) / ((v / (1 - b2 ** step)).sqrt() + eps))
    return W.cpu().numpy(), b[:, 0].cpu().numpy()


def _inner_split(groups, seed, frac=0.2):
    """Grouped train/val split of a training fold (falls back to rows if one group)."""
    rng = np.random.default_rng(seed)
    u = np.unique(groups)
    if len(u) < 2:
        m = rng.random(len(groups)) < frac
        return ~m, m
    val_g = rng.choice(u, size=max(1, int(round(frac * len(u)))), replace=False)
    m = np.isin(groups, val_g)
    return ~m, m


def fit_predict(Xtr, ttr: Target, tr_idx, Xte, groups_tr, solver="adamw", sweep=PAPER_SWEEP,
                epochs=100, batch_size=64, device="cpu", seed=0):
    """Fit on a training fold (with internal model selection); predict the test fold.

    Returns (predictions [n_test, k], info dict, (W [d, k], b [k]) in z-scored space).
    """
    binary = ttr.kind == "binary"
    Y = ttr.Y[tr_idx]
    sc = Scaler(Xtr)
    Xs, Xt = sc(Xtr), sc(Xte)
    ymu = np.zeros((1, Y.shape[1])) if binary else Y.mean(0, keepdims=True)
    ysd = np.ones((1, Y.shape[1])) if binary else Y.std(0, keepdims=True) + 1e-8
    Yn = (Y - ymu) / ysd

    if solver == "ridge":
        import warnings

        from scipy.linalg import LinAlgWarning
        # the smallest penalties are nearly singular when features outnumber videos;
        # validation then rejects them, so the warning is noise
        warnings.filterwarnings("ignore", category=LinAlgWarning)
        # penalty chosen on a grouped validation split (same rule as the AdamW sweep),
        # then refitted on the whole training fold
        itr, iva = _inner_split(groups_tr, seed)
        if binary:
            # ridge regression on +/-1 labels (a least-squares linear classifier): closed
            # form, so it is as fast as the regression case; the sign is the prediction
            from sklearn.linear_model import Ridge
            pm = 2.0 * Y - 1.0
            grid = np.logspace(-2, 5, 15)
            val = [((Ridge(alpha=al).fit(Xs[itr], pm[itr]).predict(Xs[iva]).reshape(-1) > 0)
                    == (Y[iva, 0] > 0.5)).mean() for al in grid]
            best = float(grid[int(np.argmax(val))])
            m = Ridge(alpha=best).fit(Xs, pm)
            W, b, info = m.coef_.reshape(1, -1).T, np.atleast_1d(m.intercept_), {"alpha": best}
        else:
            from sklearn.linear_model import Ridge
            grid = np.logspace(-2, 5, 15)
            val = [-((Ridge(alpha=al).fit(Xs[itr], Yn[itr]).predict(Xs[iva]).reshape(-1, Yn.shape[1])
                      - Yn[iva]) ** 2).mean() for al in grid]
            best = float(grid[int(np.argmax(val))])
            m = Ridge(alpha=best).fit(Xs, Yn)
            W, b, info = m.coef_.reshape(Yn.shape[1], -1).T, np.atleast_1d(m.intercept_), {"alpha": best}
    elif solver == "adamw":
        itr, iva = _inner_split(groups_tr, seed)
        W_all, b_all = _fit_adamw_sweep(Xs[itr], Yn[itr], binary, sweep, epochs, batch_size, device, seed)
        pv = np.einsum("nd,cdk->cnk", Xs[iva], W_all) + b_all[:, None]
        if binary:
            val = ((pv[..., 0] > 0) == (Y[iva][None, :, 0] > 0.5)).mean(1)
            # break accuracy ties with the log-loss
            ll = np.logaddexp(0, -np.where(Y[iva][None] > 0.5, pv, -pv)).mean((1, 2))
            best = int(np.lexsort((ll, -val))[0])
        else:
            val = -((pv - Yn[iva][None]) ** 2).mean((1, 2))
            best = int(np.argmax(val))
        W, b = W_all[best], b_all[best]
        info = {"lr": sweep[best][0], "wd": sweep[best][1]}
    else:
        raise ValueError(solver)
    pred = (Xt @ W + b) * ysd + ymu
    return pred, info, (W, b)


def make_folds(groups, y_strat=None, n_splits=5):
    """Grouped K-fold (stratified for classification). Fewer folds if few groups."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    n = min(n_splits, len(np.unique(groups)))
    if n < 2:
        raise ValueError("need at least 2 groups for cross-validation")
    if y_strat is not None:
        return list(StratifiedGroupKFold(n, shuffle=True, random_state=0).split(groups, y_strat, groups))
    return list(GroupKFold(n).split(groups, groups=groups))


def resolve_groups(labels: pd.DataFrame, group_col: str | None) -> np.ndarray:
    if group_col and group_col in labels.columns:
        return pd.factorize(labels[group_col])[0]
    # no grouping column: each video is its own group -> plain K-fold
    return np.arange(len(labels))


def cv_probe(X, target: Target, groups, n_splits=5, **fit_kw) -> pd.DataFrame:
    """Grouped K-fold probe of one feature matrix. One row per fold."""
    strat = target.Y[:, 0] if target.kind == "binary" else None
    rows = []
    for f, (tr, te) in enumerate(make_folds(groups, strat, n_splits)):
        pred, info, _ = fit_predict(X[tr], target, tr, X[te], groups[tr], **fit_kw)
        rows.append({"fold": f, **score(target, te, pred), **info})
    return pd.DataFrame(rows)


def holdout_probe(X, target: Target, groups, test_mask, **fit_kw) -> pd.DataFrame:
    """Train on ~test_mask, test on test_mask (e.g. generalise to unseen speeds)."""
    tr, te = np.where(~test_mask)[0], np.where(test_mask)[0]
    pred, info, _ = fit_predict(X[tr], target, tr, X[te], groups[tr], **fit_kw)
    return pd.DataFrame([{"fold": 0, **score(target, te, pred), **info}])


def layerwise(cache, targets: list[Target], group_col=None, layers=None, test_mask=None,
              row_mask=None, progress=True, **fit_kw) -> pd.DataFrame:
    """Probe every layer of an activation cache. Long-format result:
    columns layer, target, kind, fold, <metrics>."""
    labels = cache.labels
    groups = resolve_groups(labels, group_col or cache.group)
    keep = np.ones(len(labels), bool) if row_mask is None else np.asarray(row_mask)
    layers = list(range(cache.n_layers)) if layers is None else list(layers)
    it = layers
    if progress:
        from tqdm import tqdm
        it = tqdm(layers, desc="probe layers", unit="layer")
    out = []
    for l in it:
        X = cache.layer(l)[keep]
        for t in targets:
            tk = Target(t.name, t.kind, t.Y[keep], None if t.angle is None else t.angle[keep])
            if test_mask is not None:
                df = holdout_probe(X, tk, groups[keep], np.asarray(test_mask)[keep], **fit_kw)
            else:
                df = cv_probe(X, tk, groups[keep], **fit_kw)
            df.insert(0, "kind", t.kind)
            df.insert(0, "target", t.name)
            df.insert(0, "layer", l)
            out.append(df)
    return pd.concat(out, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD across folds for every (layer, target)."""
    metrics = [c for c in ("r2", "circ_mae_deg", "accuracy") if c in df.columns]
    g = df.groupby(["target", "kind", "layer"])[metrics].agg(["mean", "std"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    return g.reset_index()


def emergence_layer(summary: pd.DataFrame, target: str, chance: float | None = None,
                    frac: float = 0.5) -> int | None:
    """First layer where the probe has covered `frac` of the way from its floor to its
    peak. A crude but explicit definition of 'where the variable becomes decodable'."""
    s = summary[summary.target == target].sort_values("layer")
    col = "accuracy_mean" if "accuracy_mean" in s and s["accuracy_mean"].notna().any() else "r2_mean"
    v = s[col].to_numpy()
    lo = chance if chance is not None else max(v.min(), 0.0 if col == "r2_mean" else 0.5)
    if v.max() < (0.2 if col == "r2_mean" else 0.6):  # never decodable: no emergence layer
        return None
    thr = lo + frac * (v.max() - lo)
    hit = np.where(v >= thr)[0]
    return int(s["layer"].to_numpy()[hit[0]]) if len(hit) else None


# -------------------------------------------------------------- attentive probe


class AttentiveProbe(torch.nn.Module):
    """Patch-preserving probe: one learned query cross-attends over patch tokens,
    followed by an MLP head. The paper names 'attentive-MLP' probes without giving
    the architecture; this is the V-JEPA-style attentive pooler in miniature."""

    def __init__(self, d, out, heads=8, hidden=512):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d)
        self.query = torch.nn.Parameter(torch.zeros(1, 1, d))
        torch.nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = torch.nn.MultiheadAttention(d, heads, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, hidden),
                                        torch.nn.GELU(), torch.nn.Linear(hidden, out))

    def forward(self, x):
        x = self.norm(x)
        q = self.query.expand(len(x), -1, -1)
        return self.head(self.attn(q, x, x, need_weights=False)[0][:, 0])


def cv_attentive(tokens, target: Target, groups, n_splits=5, epochs=60, lr=1e-3, wd=0.05,
                 batch_size=32, device="cpu", seed=0) -> pd.DataFrame:
    """Grouped K-fold with the attentive probe on patch tokens [N, P, d]."""
    binary = target.kind == "binary"
    strat = target.Y[:, 0] if binary else None
    rows = []
    for f, (tr, te) in enumerate(make_folds(groups, strat, n_splits)):
        torch.manual_seed(seed + f)
        Xtr = torch.as_tensor(np.asarray(tokens[np.sort(tr)], dtype=np.float32), device=device)
        tr = np.sort(tr)
        mu, sd = Xtr.mean((0, 1), keepdim=True), Xtr.std((0, 1), keepdim=True) + 1e-6
        Xtr = (Xtr - mu) / sd
        ymu = 0.0 if binary else target.Y[tr].mean(0, keepdims=True)
        ysd = 1.0 if binary else target.Y[tr].std(0, keepdims=True) + 1e-8
        Y = torch.as_tensor((target.Y[tr] - ymu) / ysd, dtype=torch.float32, device=device)
        model = AttentiveProbe(Xtr.shape[-1], Y.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for _ in range(epochs):
            perm = torch.randperm(len(Xtr), device=device)
            for s in range(0, len(Xtr), batch_size):
                i = perm[s:s + batch_size]
                p = model(Xtr[i])
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(p, Y[i]) if binary
                        else torch.nn.functional.mse_loss(p, Y[i]))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
        model.eval()
        te = np.sort(te)
        with torch.no_grad():
            Xte = (torch.as_tensor(np.asarray(tokens[te], dtype=np.float32), device=device) - mu) / sd
            pred = torch.cat([model(Xte[s:s + 64]) for s in range(0, len(Xte), 64)]).cpu().numpy()
        pred = pred * ysd + ymu
        rows.append({"fold": f, **score(target, te, pred)})
    return pd.DataFrame(rows)
