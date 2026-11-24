"""Find and load everything the GUI shows. No Streamlit in here, so it can be tested
on its own and reused from notebooks.

A "runs" folder is searched recursively for four kinds of result:
    activation caches   folders with meta.json + pooled.npy   (from `extract`)
    probe runs          folders with probe_summary.csv        (from `probe`, `ablate`)
    ablations           folders with ablation_table.csv       (from `ablate`)
    attention stats     attention_distance.csv / attn_distance.npz
A "data" folder is searched for datasets (folders with dataset.json).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import manifold as mf

MIN_RELIABILITY = 0.3  # same default as `physprobe manifold --min-reliability`
SKIP_LABELS = {"video_id", "seed", "scene_seed", "start_x", "start_y", "path", "source_id"}


@dataclass
class Found:
    caches: list
    probes: list
    ablations: list
    attention: list
    datasets: list


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root)) or "."
    except ValueError:
        return str(p)


def discover(runs: str | Path, data: str | Path | None = None) -> Found:
    """Scan one results folder (live results or an exported bundle).

    Each activation set gets a kind: "live" (activation arrays on disk), "bundle"
    (precomputed shapes from `physprobe export`) or "missing" (neither: results only).
    """
    runs = Path(runs)
    caches, probes, ablations, attention, datasets = [], [], [], [], []
    if data is None and (runs / "bundle.json").exists():
        data = runs / "datasets"  # bundles carry their dataset previews
    if runs.exists():
        for meta in sorted(runs.rglob("meta.json")):
            d = meta.parent
            try:
                info = _cache_info(d)
            except (ValueError, KeyError):
                continue
            if info["layers"] is None:
                continue  # some other kind of meta.json
            live = (d / "pooled.npy").exists() and (d / "labels.csv").exists()
            bundled = (d / "shapes").is_dir() and any((d / "shapes").glob("*.npz"))
            kind = "live" if live else "bundle" if bundled else "missing"
            caches.append({"path": str(d), "name": _rel(d, runs), **info, "kind": kind,
                           "missing": kind == "missing"})
        for f in sorted(runs.rglob("probe_summary.csv")):
            probes.append({"path": str(f.parent), "name": _rel(f.parent, runs)})
        for f in sorted(runs.rglob("ablation_table.csv")):
            ablations.append({"path": str(f.parent), "name": _rel(f.parent, runs)})
        seen = set()
        # per-head statistics (.npz next to the activations) first, summary tables second
        for f in sorted(runs.rglob("attn_distance.npz")) + sorted(runs.rglob("attention_distance.csv")):
            if f.parent not in seen:
                seen.add(f.parent)
                attention.append({"path": str(f.parent), "name": _rel(f.parent, runs)})
    if data is not None and Path(data).exists():
        for f in sorted(Path(data).rglob("dataset.json")):
            if (f.parent / "manifest.csv").exists() is False:
                continue
            m = json.loads(f.read_text())
            n = sum(1 for _ in open(f.parent / "manifest.csv")) - 1 if (f.parent / "manifest.csv").exists() else 0
            datasets.append({"path": str(f.parent), "name": _rel(f.parent, Path(data)), "type": m.get("type"),
                             "family": m.get("family") or Path(str(m.get("source", ""))).name, "videos": n,
                             "targets": m.get("targets", {}),
                             "preview": str(f.parent / "preview.png") if (f.parent / "preview.png").exists() else None})
    return Found(caches, probes, ablations, attention, datasets)


def discover_many(roots: list) -> Found:
    """roots: [(label, runs, data)]. Names get a 'label: ' prefix when label is set."""
    merged = Found([], [], [], [], [])
    for label, runs, data in roots:
        f = discover(runs, data)
        for field in ("caches", "probes", "ablations", "attention", "datasets"):
            for e in getattr(f, field):
                if label:
                    e = {**e, "name": f"{label}: {e['name']}"}
                getattr(merged, field).append(e)
    return merged


# ------------------------------------------------------------------ bundles


class BundleSet:
    """An activation set from a results bundle: labels, metadata and precomputed shapes."""

    def __init__(self, path: str):
        self.root = Path(path)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.labels = pd.read_csv(self.root / "labels.csv")
        self.n_layers = self.meta["n_layers"]

    @property
    def targets(self) -> dict:
        return self.meta.get("dataset_meta", {}).get("targets", {})

    @property
    def group(self):
        return self.meta.get("dataset_meta", {}).get("group")

    @property
    def pez(self) -> int:
        return self.n_layers // 3

    def shape_labels(self) -> list:
        return sorted(p.stem for p in (self.root / "shapes").glob("*.npz"))

    def trajectory(self, by: str) -> dict:
        z = np.load(self.root / "shapes" / f"{by}.npz", allow_pickle=False)
        return {"levels": z["levels"], "scores": z["scores"].astype(np.float64), "var": z["var"],
                "videos": z["videos"].astype(np.float64), "video_levels": z["video_levels"],
                "circular": bool(z["circular"])}

    def metrics(self, by: str) -> pd.DataFrame:
        return pd.read_csv(self.root / "shapes" / f"{by}.csv")


def _cache_info(d: Path) -> dict:
    m = json.loads((d / "meta.json").read_text())
    dm = m.get("dataset_meta", {})
    return {"model": m.get("model", "?"), "videos": m.get("n"), "layers": m.get("n_layers"),
            "dim": m.get("d_model"), "dataset": dm.get("family") or dm.get("type", "?"),
            "fc1": bool(m.get("fc1_layers")), "time": bool(m.get("time_layers")),
            "ablated": m.get("attn_edit") is not None}


# ------------------------------------------------------------------- activations


def load_cache(path: str):
    from ..extract import ActivationCache
    return ActivationCache(path)


def label_columns(labels: pd.DataFrame) -> list:
    """Columns worth tracing: a handful to a few dozen distinct values."""
    out = []
    for c in labels.columns:
        if c in SKIP_LABELS:
            continue
        n = labels[c].nunique()
        if 2 <= n <= 60 and n < len(labels):
            out.append(c)
    return out


def is_circular(labels: pd.DataFrame, col: str, targets: dict) -> bool:
    vals = labels[col].to_numpy()
    if not np.issubdtype(vals.dtype, np.number):
        return False
    return targets.get(col) == "circular" or (col.endswith("_deg") and mf.is_circular_grid(vals))


def manifold_trajectory(cache, by: str, where: str | None = None, k: int = 3,
                        with_videos: bool = False, align: bool = True) -> dict:
    """Condition means of `by` at every layer, projected onto each layer's own top-k
    principal axes, aligned layer to layer for smooth animation.

    -> dict(levels [C], scores [L, C, k], var [L, k], videos [L, N, k] (optional),
            video_levels [N])
    """
    labels = cache.labels
    keep = labels.eval(where).to_numpy(bool) if where else np.ones(len(labels), bool)
    lab = labels[keep].reset_index(drop=True)
    idx = lab.groupby(by, sort=True).indices
    levels = np.array(list(idx.keys()))
    L = cache.n_layers
    k = min(k, len(levels) - 1)
    scores = np.zeros((L, len(levels), k))
    var = np.zeros((L, k))
    vids = np.zeros((L, int(keep.sum()), k)) if with_videos else None
    prev = None
    for l in range(L):
        X = cache.layer(l)[keep].astype(np.float64)
        M = np.stack([X[i].mean(0) for i in idx.values()])
        P = mf.pca(M, k)
        S = P["scores"]
        if align and prev is not None:
            # orthogonal Procrustes: rotate/reflect this layer's axes to best match the
            # previous layer, so the shape doesn't flip or spin between frames (distances
            # and the shape itself are unchanged). Videos get the same map.
            U, _, Vt = np.linalg.svd(S.T @ prev)
            R = U @ Vt
        else:
            R = np.eye(S.shape[1])
        scores[l] = S @ R
        var[l] = P["var_ratio"][:k]
        if with_videos:
            vids[l] = ((X - P["mean"]) @ P["components"].T) @ R
        prev = scores[l]
    return {"levels": levels, "scores": scores, "var": var, "videos": vids,
            "video_levels": lab[by].to_numpy()}


def manifold_metrics(cache, by: str, where: str | None = None, circular: bool = False) -> pd.DataFrame:
    """Per-layer statistics of the `by` manifold (the same ones `physprobe manifold`
    reports): split-half reliability, noise-corrected dimensionality, and for circular
    variables the Fourier content and ring fit."""
    labels = cache.labels
    keep = labels.eval(where).to_numpy(bool) if where else np.ones(len(labels), bool)
    lab = labels[keep].reset_index(drop=True)
    group = cache.group if cache.group in lab.columns and cache.group != by else None
    rows = []
    for l in range(cache.n_layers):
        X = cache.layer(l)[keep]
        row = {"layer": l, "reliability": mf.reliability(X, lab, [by], group, n_splits=8)}
        cv = mf.cv_spectrum(X, lab, [by], group, n_splits=8)
        row["dimensions"] = cv["dims90"]
        if circular:
            h = mf.harmonics_1d_cv(X, lab, by, group, n_splits=8)
            for j in range(min(4, len(h))):
                row[f"harmonic {j + 1}"] = float(h[j])
            keys, M, _ = mf.condition_means(X, lab, [by])
            P = mf.pca(M, 2)
            row["ring order"] = mf.ring_fit(P["scores"][:, :2], keys[by].to_numpy(float))["circ_corr"]
        rows.append(row)
    df = pd.DataFrame(rows)
    # where there is no reproducible shape, its shape statistics are noise: blank them
    noisy = df["reliability"] < MIN_RELIABILITY
    df.loc[noisy, [c for c in df.columns if c not in ("layer", "reliability")]] = np.nan
    return df


def quick_probe(cache, targets: list, group: str | None = None) -> pd.DataFrame:
    """Ridge probes at every layer (fast); summary table like `physprobe probe` writes."""
    from .. import probes
    tg = probes.parse_targets(targets, cache.labels, cache.targets)
    df = probes.layerwise(cache, tg, group, progress=False, solver="ridge")
    return probes.summarize(df)


# ---------------------------------------------------------------------- results


def read_probe_summary(path: str) -> pd.DataFrame:
    return pd.read_csv(Path(path) / "probe_summary.csv")


def read_ablation(path: str) -> dict:
    """-> {"table": DataFrame, "curves": {condition: summary}, "layers": [...]}"""
    d = Path(path)
    tab = pd.read_csv(d / "ablation_table.csv", index_col=0)
    curves, layers = {}, None
    for sub in sorted(p for p in d.iterdir() if p.is_dir()):
        f = sub / "probe_summary.csv"
        if f.exists():
            curves[sub.name] = pd.read_csv(f)
        m = sub / "meta.json"
        if m.exists() and layers is None:
            e = json.loads(m.read_text()).get("attn_edit")
            if e:
                layers = e.get("layers")
    order = sorted(curves, key=lambda n: (n != "base", n))
    return {"table": tab, "curves": {n: curves[n] for n in order}, "layers": layers}


def read_attention(path: str) -> dict:
    d = Path(path)
    out = {}
    if (d / "attn_distance.npz").exists():
        z = np.load(d / "attn_distance.npz")
        out["spatial"], out["temporal"] = z["spatial"], z["temporal"]
    elif (d / "attention_distance.csv").exists():
        df = pd.read_csv(d / "attention_distance.csv")
        out["table"] = df
    return out
