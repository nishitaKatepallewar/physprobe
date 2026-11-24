"""Results bundles: everything the explorer shows, without the activation arrays.

A bundle is a folder (or a .zip of one) with the same layout as the results folder
it came from, plus a bundle.json marker. For each activation set it keeps the
metadata and labels and replaces the arrays with precomputed shapes:

    <set>/meta.json, labels.csv           copied
    <set>/shapes/<label>.npz              condition-mean trajectory over layers (3 PCs),
                                          videos projected onto the same axes (float16)
    <set>/shapes/<label>.csv              per-layer shape statistics
    <set>/attn_distance.npz               copied if present

Probe runs, ablation runs and attention tables are copied as they are. Datasets get a
preview image and their manifest. Typical size: a few MB for a full replication run,
against hundreds of MB for the activations.
"""
from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

FORMAT = "physprobe-bundle"
VERSION = 1


def _labels_to_export(cache) -> list:
    from .gui.data import label_columns
    from .stimuli import FAMILIES
    cols = label_columns(cache.labels)
    fam = FAMILIES.get(cache.meta.get("dataset_meta", {}).get("family"))
    wanted = set(cache.targets) | set(fam.grid if fam else ()) | {"possible", "shuffled", "violation"}
    keep = [c for c in cols if c in wanted]
    return keep or cols[:4]


def export_bundle(runs: str | Path, out: str | Path, data: str | Path | None = None,
                  labels: list | None = None, zip_it: bool = False, progress=print) -> Path:
    from .gui import data as D
    runs, out = Path(runs).resolve(), Path(out).resolve()
    if out == runs or runs in out.parents:
        raise ValueError("the bundle must be written outside the results folder")
    out.mkdir(parents=True, exist_ok=True)
    found = D.discover(runs, data)
    n_shapes = 0

    for c in found.caches:
        if c["missing"] or c.get("kind") == "bundle":
            continue
        src = Path(c["path"])
        dst = out / src.relative_to(runs)
        (dst / "shapes").mkdir(parents=True, exist_ok=True)
        for f in ("meta.json", "labels.csv", "attn_distance.npz"):
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        cache = D.load_cache(str(src))
        cols = labels or _labels_to_export(cache)
        for by in cols:
            if by not in cache.labels.columns:
                continue
            t0 = time.time()
            circ = D.is_circular(cache.labels, by, cache.targets)
            tr = D.manifold_trajectory(cache, by, None, k=3, with_videos=True)
            np.savez_compressed(dst / "shapes" / f"{by}.npz", levels=tr["levels"],
                                scores=tr["scores"].astype(np.float32), var=tr["var"].astype(np.float32),
                                videos=tr["videos"].astype(np.float16), video_levels=tr["video_levels"],
                                circular=np.array(circ))
            D.manifold_metrics(cache, by, None, circ).to_csv(dst / "shapes" / f"{by}.csv", index=False)
            n_shapes += 1
            progress(f"  {c['name']}: {by} ({time.time() - t0:.1f}s)")

    def copy_tree_files(entries, names):
        for e in entries:
            src = Path(e["path"])
            for f in src.rglob("*"):
                if f.is_file() and (f.name in names or f.suffix == ".png"):
                    rel = f.relative_to(runs)
                    (out / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, out / rel)

    copy_tree_files(found.probes, {"probe_summary.csv"})
    for a in found.ablations:  # the table, each condition's summary and its meta (for the ablated layers)
        src = Path(a["path"])
        for f in [src / "ablation_table.csv", *src.glob("*/probe_summary.csv"), *src.glob("*/meta.json")]:
            if f.exists():
                rel = f.relative_to(runs)
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out / rel)
    copy_tree_files(found.attention, {"attention_distance.csv", "attn_distance.npz"})

    for d in found.datasets:
        src = Path(d["path"])
        dst = out / "datasets" / d["name"]
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("dataset.json", "manifest.csv"):
            shutil.copy2(src / f, dst / f)
        try:
            from .datasets import VideoDataset, write_contact_sheet
            write_contact_sheet(VideoDataset(src), dst / "preview.png", n=6)
        except Exception as e:  # noqa: BLE001 - a preview is optional
            progress(f"  (no preview for {d['name']}: {e})")

    (out / "bundle.json").write_text(json.dumps({
        "format": FORMAT, "version": VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": runs.name, "activation_sets": sum(not c["missing"] for c in found.caches),
        "shapes": n_shapes, "probe_runs": len(found.probes), "ablation_runs": len(found.ablations)}, indent=2))
    if zip_it:
        z = out.with_suffix(".zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(out))
        return z
    return out


def is_bundle(path: str | Path) -> bool:
    f = Path(path) / "bundle.json"
    return f.exists() and json.loads(f.read_text()).get("format") == FORMAT


def safe_extract(zip_path, dest: str | Path, max_bytes: int = 500 * 2 ** 20) -> Path:
    """Extract an uploaded bundle zip, refusing absolute paths, '..' and oversized archives.
    Returns the bundle root (the folder holding bundle.json)."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        total = sum(i.file_size for i in zf.infolist())
        if total > max_bytes:
            raise ValueError(f"archive expands to {total / 2 ** 20:.0f} MB; the limit is {max_bytes / 2 ** 20:.0f} MB")
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if dest not in target.parents and target != dest:
                raise ValueError(f"unsafe path in archive: {info.filename}")
            if info.filename.endswith((".npy",)):
                continue  # activation arrays are never needed in a bundle
            zf.extract(info, dest)
    roots = [p.parent for p in dest.rglob("bundle.json")]
    if not roots:
        raise ValueError("no bundle.json in the archive: bundles are made with `python -m physprobe export`")
    root = min(roots, key=lambda p: len(p.parts))
    if not is_bundle(root):
        raise ValueError("bundle.json is not a physprobe bundle")
    return root
