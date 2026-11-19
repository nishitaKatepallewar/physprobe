"""Run a dataset through an encoder and cache activations on disk.

Cache layout (a folder):
    meta.json            model, resolution, frames, layer lists, attention edit
    labels.csv           copy of the dataset manifest (row i <-> activation row i)
    pooled.npy           [N, L, d]  float32  mean-pooled residual stream, every block
    patch_L{l}.npy       [N, G*G, d] float16 time-averaged patch tokens
    time_L{l}.npy        [N, Tt, d]  float16 space-averaged temporal tokens
    fc1_L{l}.npy         [N, d_mlp]  float16 MLP hidden units (post-activation)
    fc2_L{l}.npy         [N, d]      float16 MLP outputs
    attn_distance.npz    spatial / temporal mean attention distance, [L, heads]
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # torch is only needed to extract, not to read a cache
    from .datasets import VideoDataset
    from .models import AttnEdit, Encoder


def parse_layers(spec: str | None, n_layers: int) -> tuple:
    """'all' | 'none' | 'thirds' | '4,8,12' | '0:24:4'."""
    if spec is None or spec in ("none", ""):
        return ()
    if spec == "all":
        return tuple(range(n_layers))
    if spec == "thirds":  # a handful of layers around one-third depth, plus anchors
        pez = n_layers // 3
        return tuple(sorted({0, max(pez - 2, 0), pez, pez + 2, n_layers // 2, n_layers - 1}))
    if ":" in spec:
        a, b, s = (int(v) for v in spec.split(":"))
        return tuple(range(a, min(b, n_layers), s))
    return tuple(int(v) % n_layers for v in spec.split(","))


def extract(ds: VideoDataset, enc: Encoder, out: str | Path, batch_size: int = 8,
            frames: int | None = None, patch_layers=(), time_layers=(), fc1_layers=(),
            fc2_layers=(), attn_edit: AttnEdit | None = None, attn_stats: bool = False,
            limit: int | None = None, progress: bool = True) -> "ActivationCache":
    from .models import Capture
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    N = len(ds) if limit is None else min(limit, len(ds))
    T = frames or ds.meta.get("frames", 16)
    cap = Capture(True, tuple(patch_layers), tuple(time_layers), tuple(fc1_layers), tuple(fc2_layers))
    Tt, G, _ = enc.grid(T)
    L, d = enc.n_layers, enc.d_model
    mm = {"pooled": open_memmap(out / "pooled.npy", "w+", np.float32, (N, L, d))}
    for l in cap.patch_layers:
        mm[f"patch_L{l}"] = open_memmap(out / f"patch_L{l}.npy", "w+", np.float16, (N, G * G, d))
    for l in cap.time_layers:
        mm[f"time_L{l}"] = open_memmap(out / f"time_L{l}.npy", "w+", np.float16, (N, Tt, d))
    for l in cap.fc1_layers:
        mm[f"fc1_L{l}"] = open_memmap(out / f"fc1_L{l}.npy", "w+", np.float16, (N, enc.d_mlp))
    for l in cap.fc2_layers:
        mm[f"fc2_L{l}"] = open_memmap(out / f"fc2_L{l}.npy", "w+", np.float16, (N, d))
    a_sp = np.zeros((L, enc.n_heads))
    a_tp = np.zeros((L, enc.n_heads))

    it = range(0, N, batch_size)
    if progress:
        from tqdm import tqdm
        it = tqdm(it, desc=f"extract {enc.name}", unit="batch")
    for s in it:
        idx = range(s, min(s + batch_size, N))
        vids = np.stack([ds.get(i, T) for i in idx])
        r = enc.run(vids, cap, attn_edit=attn_edit, attn_stats=attn_stats)
        for k, arr in mm.items():
            arr[s:s + len(idx)] = r[k]
        if attn_stats:
            a_sp += r["attn_spatial"]
            a_tp += r["attn_temporal"]
    for arr in mm.values():
        arr.flush()
    if attn_stats:
        np.savez(out / "attn_distance.npz", spatial=a_sp / N, temporal=a_tp / N)
    ds.labels.iloc[:N].to_csv(out / "labels.csv", index=False)
    meta = {"model": enc.name, "kind": enc.kind, "res": enc.res, "frames": T, "n": N,
            "n_layers": L, "d_model": d, "d_mlp": enc.d_mlp, "n_heads": enc.n_heads,
            "grid": [Tt, G, G], "patch_layers": list(cap.patch_layers),
            "time_layers": list(cap.time_layers), "fc1_layers": list(cap.fc1_layers),
            "fc2_layers": list(cap.fc2_layers),
            "attn_edit": asdict(attn_edit) if attn_edit else None,
            "dataset": str(ds.root), "dataset_meta": ds.meta}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return ActivationCache(out)


class ActivationCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.labels = pd.read_csv(self.root / "labels.csv")
        self.pooled = np.load(self.root / "pooled.npy", mmap_mode="r")
        self.n_layers = self.meta["n_layers"]
        self.d_model = self.meta["d_model"]

    def __len__(self):
        return len(self.labels)

    @property
    def targets(self) -> dict:
        return self.meta["dataset_meta"].get("targets", {})

    @property
    def group(self):
        return self.meta["dataset_meta"].get("group")

    @property
    def pez(self) -> int:
        """Nominal 'physics emergence' layer: one-third depth (layer 8 of 24 for ViT-L)."""
        return self.n_layers // 3

    def layer(self, l: int) -> np.ndarray:
        return np.asarray(self.pooled[:, l], dtype=np.float32)

    def get(self, kind: str, l: int) -> np.ndarray:
        f = self.root / f"{kind}_L{l}.npy"
        if not f.exists():
            have = self.meta.get(f"{kind}_layers", [])
            raise FileNotFoundError(f"{kind} activations for layer {l} were not extracted "
                                    f"(available layers: {have}). Re-run `extract` with "
                                    f"--{kind}-layers.")
        return np.load(f, mmap_mode="r")
