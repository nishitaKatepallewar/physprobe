"""Manifold tools: what shape does a variable trace out in activation space?

The unit of analysis is the *condition mean*: average the activations of all
videos that share a level of the variable(s) of interest (averaging out every
nuisance factor), then study the geometry of those means.

    condition_means      average by one or more label columns
    pca                  components, variance explained, participation ratio
    split_half_means     the same on two independent halves (for noise-corrected estimates)
    variance_partition   how much variance each factor (and their interaction) explains
    cv_spectrum          noise-corrected dimensionality of the condition-mean manifold
    harmonics_1d / _2d   Fourier content of a circular variable (ring / torus tests)
    ring_fit             is the 2-D projection a circle, in the right order?
    rdm                  pairwise distances between condition means
    twonn_dimension      intrinsic dimension of a point cloud
    embed                Isomap / MDS / (optional) UMAP for nonlinear views
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def condition_means(X: np.ndarray, labels: pd.DataFrame, by: list[str]):
    """-> (keys DataFrame [C, len(by)], means [C, d], counts [C]) sorted by `by`."""
    labels = labels.reset_index(drop=True)
    if len(by) == 1:
        g = labels.groupby(by[0], sort=True).indices
        keys = pd.DataFrame({by[0]: list(g.keys())})
    else:
        g = labels.groupby(by, sort=True).indices
        keys = pd.DataFrame(list(g.keys()), columns=by)
    means = np.stack([X[idx].mean(0) for idx in g.values()])
    return keys, means, np.array([len(i) for i in g.values()])


def pca(X: np.ndarray, k: int = 10):
    """-> dict(scores [n, k], components [k, d], var_ratio [k], mean, pr)
    pr = participation ratio (sum l)^2 / sum l^2, an effective dimensionality."""
    mu = X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    lam = S ** 2
    k = min(k, len(S))
    return {"scores": U[:, :k] * S[:k], "components": Vt[:k], "var_ratio": lam[:k] / lam.sum(),
            "mean": mu, "pr": float(lam.sum() ** 2 / (lam ** 2).sum()), "eigvals": lam}


def split_half_means(X: np.ndarray, labels: pd.DataFrame, by: list[str], seed: int = 0,
                     group: str | None = None):
    """Condition means on two independent halves of the data, for noise-corrected
    estimates: noise is independent between halves, so products of the two halves
    estimate *signal* power, whereas plain squared means also count noise.

    With `group` (a replicate factor crossed with the conditions, e.g. start_id) the
    halves are two random sets of group levels, the same for every condition. Then
    both halves stay balanced over the other design factors, and the group's own
    main effect is common to all conditions (it drops out on centring). Without a
    usable group, each condition's videos are split at random.
    -> (keys, means_a [C, d], means_b [C, d])"""
    labels = labels.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    g = labels.groupby(by[0] if len(by) == 1 else by, sort=True).indices
    keys = pd.DataFrame({by[0]: list(g.keys())}) if len(by) == 1 else pd.DataFrame(list(g.keys()), columns=by)
    in_a = None
    if group is not None and group in labels.columns and group not in by:
        lv = rng.permutation(labels[group].unique())
        if len(lv) >= 2:
            in_a = labels[group].isin(lv[:len(lv) // 2]).to_numpy()
            if any(in_a[idx].all() or not in_a[idx].any() for idx in g.values()):
                in_a = None  # some condition lacks one half: groups are not crossed with it
    A, B = [], []
    for idx in g.values():
        if len(idx) < 2:
            raise ValueError(f"need >= 2 videos per level of {by} for a split-half estimate")
        if in_a is None:
            idx = rng.permutation(idx)
            ia, ib = idx[:len(idx) // 2], idx[len(idx) // 2:]
        else:
            ia, ib = idx[in_a[idx]], idx[~in_a[idx]]
        A.append(X[ia].mean(0))
        B.append(X[ib].mean(0))
    return keys, np.stack(A), np.stack(B)


def variance_partition(X: np.ndarray, labels: pd.DataFrame, factors: list[str],
                       random: str | None = None) -> dict:
    """Split total activation variance (summed over dimensions) into each factor's
    main effect, the pairwise interactions, and a residual (higher-order
    interactions + noise): a multivariate ANOVA for balanced, crossed designs,
    which is what the generators produce. Pass ALL design factors.

    Every effect is corrected by subtracting (its degrees of freedom) x (an error
    mean square), so an effect that is not really there scores ~0 instead of its
    chance share. `random` names the replicate factor (e.g. start_id). The videos
    are deterministic, so the only thing a main effect can fail to do is generalise
    across replicates; each fixed factor's main effect is therefore tested against
    its interaction with `random` (a mixed-model ANOVA). Everything else is tested
    against the residual.

    Additive (factorised) codes put variance in main effects; conjunctive codes put
    it in interactions.
    """
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(0, keepdims=True)
    total = float((Xc ** 2).sum())
    cells = labels.groupby(factors).size()
    if cells.nunique() != 1 or len(cells) != int(np.prod([labels[f].nunique() for f in factors])):
        print(f"  note: design is not balanced/crossed over {factors}; variance shares are approximate")
    ss, df = {}, {}
    for f in factors:
        _, m, n = condition_means(Xc, labels, [f])
        ss[f], df[f] = float((n[:, None] * m ** 2).sum()), len(m) - 1
    for i, a in enumerate(factors):
        for b in factors[i + 1:]:
            _, m, n = condition_means(Xc, labels, [a, b])
            ss[f"{a} x {b}"] = float((n[:, None] * m ** 2).sum()) - ss[a] - ss[b]
            df[f"{a} x {b}"] = df[a] * df[b]
    resid_df = len(Xc) - 1 - sum(df.values())
    mse = max(total - sum(ss.values()), 0.0) / resid_df if resid_df > 0 else 0.0

    def error_ms(k):
        if random in factors and " x " not in k and k != random:
            inter = f"{k} x {random}" if f"{k} x {random}" in ss else f"{random} x {k}"
            return max(ss[inter], 0.0) / df[inter]
        return mse

    out = {k: max(ss[k] - df[k] * error_ms(k), 0.0) / total for k in ss}
    out["residual"] = max(1.0 - sum(out.values()), 0.0)
    return out


def cv_spectrum(X: np.ndarray, labels: pd.DataFrame, by: list[str], group: str | None = None,
                n_splits: int = 16) -> dict:
    """Noise-corrected dimensionality of a condition-mean manifold. For each
    split-half, principal axes come from half A and each axis is credited with the
    product of the two halves' projections, which averages to zero for axes that
    carry only noise. -> dims90 (median number of axes holding 90% of the signal
    variance), pr (participation ratio of the split-averaged spectrum), eig."""
    eigs, d90 = [], []
    for s in range(n_splits):
        _, MA, MB = split_half_means(X, labels, by, seed=s, group=group)
        A = MA - MA.mean(0, keepdims=True)
        B = MB - MB.mean(0, keepdims=True)
        Vt = np.linalg.svd(A, full_matrices=False)[2]
        e = ((A @ Vt.T) * (B @ Vt.T)).sum(0)
        eigs.append(e)
        ec = np.sort(np.clip(e, 0, None))[::-1]
        d90.append(int(np.searchsorted(np.cumsum(ec) / max(ec.sum(), 1e-20), 0.9) + 1) if ec.sum() > 0 else 0)
    eig = np.clip(np.mean(eigs, 0), 0, None)
    tot = eig.sum()
    return {"eig": eig, "pr": float(tot ** 2 / (eig ** 2).sum()) if tot > 0 else 0.0,
            "dims90": int(np.median(d90))}


def harmonics_1d(means: np.ndarray) -> np.ndarray:
    """means [M, d] at M equally spaced angles -> fraction of across-angle variance at
    each harmonic 1..M//2. A pure ring code (a cos + b sin per unit) is all harmonic 1.
    Biased toward a flat spectrum when the means are noisy; see harmonics_1d_cv."""
    return _norm_harm(_harm_power(means, means), len(means))


def _harm_power(MA, MB):
    FA = np.fft.rfft(MA - MA.mean(0, keepdims=True), axis=0)
    FB = np.fft.rfft(MB - MB.mean(0, keepdims=True), axis=0)
    return (FA * FB.conj()).real.sum(1)[1:]


def _norm_harm(p, M):
    p = np.clip(np.array(p, dtype=float), 0, None)
    if M % 2 == 0:
        p[:-1] *= 2  # every harmonic but Nyquist has a conjugate twin
    else:
        p *= 2
    return p / max(p.sum(), 1e-20)


def harmonics_1d_cv(X: np.ndarray, labels: pd.DataFrame, by: str, group: str | None = None,
                    n_splits: int = 16) -> np.ndarray:
    """Noise-corrected harmonic spectrum: cross-power between split-half condition
    means, averaged over splits."""
    ps = []
    for s in range(n_splits):
        _, MA, MB = split_half_means(X, labels, [by], seed=s, group=group)
        ps.append(_harm_power(MA, MB))
    return _norm_harm(np.mean(ps, 0), len(MA))


def reliability(X: np.ndarray, labels: pd.DataFrame, by: list[str], group: str | None = None,
                n_splits: int = 16) -> float:
    """Split-half reliability of the condition means: correlation between the two
    halves' centred mean patterns. Near 0 = the variable leaves no reproducible
    trace (geometry statistics are then meaningless); near 1 = clean manifold."""
    rs = []
    for s in range(n_splits):
        _, MA, MB = split_half_means(X, labels, by, seed=s, group=group)
        A, B = MA - MA.mean(0, keepdims=True), MB - MB.mean(0, keepdims=True)
        rs.append(float((A * B).sum() / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-20)))
    return float(np.mean(rs))


def harmonic_plane_dim(means: np.ndarray, k: int = 1) -> dict:
    """Geometry of harmonic k of a circular variable. Each unit contributes
    a_n cos(k theta) + b_n sin(k theta); the vectors a, b span a plane. Returns the
    balance between them (1 = perfect circle, 0 = collapsed to a line) and their angle."""
    M = len(means)
    F = np.fft.rfft(means - means.mean(0, keepdims=True), axis=0)[k]
    a, b = F.real, F.imag
    s = np.linalg.svd(np.stack([a, b]), compute_uv=False)
    cosang = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20)
    return {"circularity": float(s[1] / max(s[0], 1e-20)),
            "ab_angle_deg": float(np.rad2deg(np.arccos(np.clip(cosang, 0, 1))))}


def harmonics_2d(means_grid: np.ndarray, kmax: int = 2) -> pd.DataFrame:
    """means_grid [Ma, Mb, d] over two circular variables -> variance fraction at
    every (ka, kb) harmonic pair. (1,0) and (0,1) are the two independent rings of a
    factorised torus; (1,1) / (1,-1) are conjunctive terms."""
    G = means_grid - means_grid.mean((0, 1), keepdims=True)
    P = (np.abs(np.fft.fft2(G, axes=(0, 1))) ** 2).sum(-1)
    P = P / P.sum()
    Ma, Mb = P.shape
    rows = []
    for ka in range(0, kmax + 1):
        for kb in range(-kmax, kmax + 1):
            if (ka, kb) == (0, 0) or (ka == 0 and kb < 0):
                continue
            val = P[ka % Ma, kb % Mb] + P[(-ka) % Ma, (-kb) % Mb]  # fold conjugate pair
            if (ka % Ma, kb % Mb) == ((-ka) % Ma, (-kb) % Mb):
                val = P[ka % Ma, kb % Mb]
            rows.append({"ka": ka, "kb": kb, "var_frac": float(val)})
    return pd.DataFrame(rows)


def ring_fit(xy: np.ndarray, angle_deg: np.ndarray) -> dict:
    """Is a 2-D point set (one point per angle) a ring in the right order?
    radius_cv: spread of distances from the centroid (0 = perfect circle);
    circ_corr: |mean resultant| of (embedded angle -/+ true angle), 1 = angles are
    preserved up to rotation and reflection."""
    c = xy - xy.mean(0, keepdims=True)
    r = np.linalg.norm(c, axis=1)
    phi = np.arctan2(c[:, 1], c[:, 0])
    th = np.deg2rad(angle_deg)
    cc = max(abs(np.exp(1j * (phi - th)).mean()), abs(np.exp(1j * (phi + th)).mean()))
    return {"radius_cv": float(r.std() / max(r.mean(), 1e-20)), "circ_corr": float(cc)}


def rdm(means: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(means, metric))


def twonn_dimension(X: np.ndarray) -> float:
    """TwoNN intrinsic-dimension estimate (Facco et al. 2017) from the ratio of each
    point's 2nd to 1st nearest-neighbour distance. Needs a few hundred points."""
    from sklearn.neighbors import NearestNeighbors
    d, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    ok = d[:, 1] > 0
    mu = d[ok, 2] / d[ok, 1]
    return float(len(mu) / np.log(mu).sum())


def embed(X: np.ndarray, method: str = "isomap", k: int = 3, n_neighbors: int = 8) -> np.ndarray:
    if method == "pca":
        return pca(X, k)["scores"]
    if method == "isomap":
        from sklearn.manifold import Isomap
        return Isomap(n_neighbors=min(n_neighbors, len(X) - 1), n_components=k).fit_transform(X)
    if method == "mds":
        from sklearn.manifold import MDS
        return MDS(n_components=k, random_state=0).fit_transform(X)
    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError("pip install umap-learn to use --embed umap") from e
        return umap.UMAP(n_components=k, n_neighbors=min(n_neighbors, len(X) - 1),
                         random_state=0).fit_transform(X)
    raise ValueError(method)


def persistent_h1(X: np.ndarray) -> dict | None:
    """Longest-lived 1-cycle (loop) of a point cloud, if `ripser` is installed.
    A ring gives one loop far more persistent than the rest."""
    try:
        from ripser import ripser
    except ImportError:
        return None
    dg = ripser(X, maxdim=1)["dgms"][1]
    if len(dg) == 0:
        return {"top": 0.0, "second": 0.0}
    life = np.sort(dg[:, 1] - dg[:, 0])[::-1]
    return {"top": float(life[0]), "second": float(life[1]) if len(life) > 1 else 0.0}


def is_circular_grid(values: np.ndarray) -> bool:
    """True if the sorted unique values are equally spaced and wrap around 360."""
    v = np.sort(np.unique(values))
    if len(v) < 4:
        return False
    step = np.diff(v)
    return bool(np.allclose(step, step[0]) and np.isclose(v[0] + 360 - v[-1], step[0]))
