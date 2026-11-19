"""Datasets: a manifest (one row per video, with labels) plus a way to get frames.

On disk a dataset is a folder with
    dataset.json   {"type": "synthetic" | "folder", ...render or sampling settings}
    manifest.csv   one row per video; label columns are free-form

* synthetic: frames are re-rendered on demand from the manifest row (deterministic),
  so nothing heavy is stored.
* folder: the manifest has a `path` column pointing at a video file, a directory of
  frames, or a .npy array [T, H, W, 3]; frames are sampled to the requested length.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import stimuli

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class VideoDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "dataset.json").read_text())
        self.labels = pd.read_csv(self.root / "manifest.csv")
        self.kind = self.meta["type"]
        if self.kind == "synthetic":
            for plug in self.meta.get("plugins", []):
                if not Path(plug).exists() and self.meta.get("family") in stimuli.FAMILIES:
                    continue  # the family became built-in (e.g. velocity_centered, velocity_stepped)
                load_plugin(plug)
        self.source = VideoDataset(self.meta["source"]) if self.kind == "shuffled" else None

    def __len__(self):
        return len(self.labels)

    @property
    def targets(self) -> dict:
        return self.meta.get("targets", {})

    @property
    def group(self) -> str | None:
        return self.meta.get("group")

    def get(self, i: int, T: int | None = None) -> np.ndarray:
        """uint8 frames [T, H, W, 3] for video i."""
        row = self.labels.iloc[i].to_dict()
        if self.kind == "shuffled":
            v = self.source.get(int(row["source_id"]), T)
            if row["shuffled"]:
                # shuffle whole blocks of `block` consecutive frames; block=2 matches the
                # tubelet size, so motion inside every token stays natural and only the
                # order across tokens is broken
                k = int(self.meta.get("block", 1))
                nb = len(v) // k
                perm = np.random.default_rng(int(row["source_id"]) + 12345).permutation(nb)
                if (perm == np.arange(nb)).all():
                    perm = perm[::-1]
                idx = (perm[:, None] * k + np.arange(k)[None]).ravel()
                v = np.concatenate([v[idx], v[nb * k:]])
            return v
        if self.kind == "synthetic":
            m = self.meta
            return stimuli.render_row(m["family"], row, T or m["frames"], m["render_res"],
                                      m.get("floor", "noise"), m.get("view_m", stimuli.VIEW_M))
        p = Path(row["path"])
        if not p.is_absolute():
            p = self.root / p
        return load_frames(p, T or self.meta.get("frames", 16),
                           self.meta.get("sampling", "uniform"), self.meta.get("stride", 1))


def load_plugin(path: str):
    """Import a python file that registers extra stimulus families."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"physprobe_plugin_{Path(path).stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sample_indices(n: int, T: int, sampling: str = "uniform", stride: int = 1) -> np.ndarray:
    if sampling == "uniform":  # T frames spread over the whole clip
        return np.round(np.linspace(0, n - 1, T)).astype(int)
    if sampling == "center":  # T consecutive (strided) frames around the middle
        span = (T - 1) * stride
        start = max(0, (n - 1 - span) // 2)
    elif sampling == "first":
        start = 0
    else:
        raise ValueError(f"unknown sampling {sampling!r}")
    return np.clip(start + np.arange(T) * stride, 0, n - 1)


def load_frames(path: Path, T: int, sampling: str = "uniform", stride: int = 1) -> np.ndarray:
    from PIL import Image
    if path.is_dir():
        files = sorted(f for f in path.iterdir() if f.suffix.lower() in IMG_EXT)
        if not files:
            raise FileNotFoundError(f"no image frames in {path}")
        idx = sample_indices(len(files), T, sampling, stride)
        return np.stack([np.asarray(Image.open(files[i]).convert("RGB")) for i in idx])
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        return np.ascontiguousarray(arr[sample_indices(len(arr), T, sampling, stride)])
    import cv2  # video file
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise IOError(f"could not decode {path}")
    return np.stack([frames[i] for i in sample_indices(len(frames), T, sampling, stride)])


# ------------------------------------------------------------------------- writers


def write_synthetic(out: str | Path, family: str, grid_overrides=None, n_starts=None, seed=0,
                    frames=16, render_res=256, floor="noise", view_m=stimuli.VIEW_M,
                    plugins=()) -> VideoDataset:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fam = stimuli.FAMILIES[family]
    df = stimuli.make_manifest(family, grid_overrides, n_starts, seed)
    df.to_csv(out / "manifest.csv", index=False)
    meta = {"type": "synthetic", "family": family, "frames": frames, "render_res": render_res,
            "floor": floor, "view_m": view_m, "fps": stimuli.FPS, "seed": seed,
            "targets": fam.targets, "group": fam.group,
            "plugins": [str(Path(p).resolve()) for p in plugins]}
    (out / "dataset.json").write_text(json.dumps(meta, indent=2))
    return VideoDataset(out)


def write_folder(out: str | Path, manifest: pd.DataFrame, targets: dict, group: str | None,
                 frames=16, sampling="uniform", stride=1, extra=None) -> VideoDataset:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = manifest.copy()
    if "video_id" not in manifest.columns:
        manifest.insert(0, "video_id", np.arange(len(manifest)))
    manifest.to_csv(out / "manifest.csv", index=False)
    meta = {"type": "folder", "frames": frames, "sampling": sampling, "stride": stride,
            "targets": targets, "group": group, **(extra or {})}
    (out / "dataset.json").write_text(json.dumps(meta, indent=2))
    return VideoDataset(out)


def write_shuffled(out: str | Path, source: str | Path, block: int = 1) -> VideoDataset:
    """Temporal-order control (paper Sec. 6.1): every source video appears twice,
    once intact and once with its frames in random order. Target: `shuffled`.
    The two versions of a video stay in the same CV group."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    src = VideoDataset(source)
    a = src.labels.copy()
    a["source_id"] = np.arange(len(a))
    df = pd.concat([a.assign(shuffled=0), a.assign(shuffled=1)], ignore_index=True)
    df["video_id"] = np.arange(len(df))
    df.to_csv(out / "manifest.csv", index=False)
    meta = {"type": "shuffled", "source": str(Path(source).resolve()),
            "frames": src.meta.get("frames", 16), "block": block, "targets": {"shuffled": "binary"},
            "group": "source_id"}
    (out / "dataset.json").write_text(json.dumps(meta, indent=2))
    return VideoDataset(out)


def _find_is_possible(obj):
    if isinstance(obj, dict):
        if "is_possible" in obj:
            return bool(obj["is_possible"])
        for v in obj.values():
            r = _find_is_possible(v)
            if r is not None:
                return r
    return None


def index_intphys(root: str | Path) -> pd.DataFrame:
    """Index an IntPhys (2019) dev download.

    Expected layout (dev set): <root>/<O1|O2|O3>/<scene>/<1..4>/{scene/*.png, status.json}
    where status.json carries header.is_possible. The four videos under one <scene>
    form a matched quadruplet (2 possible, 2 impossible) and are kept together in
    cross-validation via the `scene` column. Anything with a status.json and a
    `scene` frame folder is picked up, so small layout differences are tolerated.
    """
    root = Path(root)
    rows = []
    for st in sorted(root.rglob("status.json")):
        vdir = st.parent
        frames = vdir / "scene"
        if not frames.is_dir():
            continue
        poss = _find_is_possible(json.loads(st.read_text()))
        if poss is None:
            continue
        rel = vdir.relative_to(root).parts
        block = next((p for p in rel if p.upper() in {"O1", "O2", "O3"}), "NA")
        rows.append({"path": str(frames.resolve()), "possible": int(poss), "block": block,
                     "scene": "/".join(rel[:-1]) or vdir.name})
    if not rows:
        raise FileNotFoundError(
            f"No <video>/status.json + <video>/scene/ pairs under {root}. Point --root at the "
            "extracted IntPhys dev folder.")
    return pd.DataFrame(rows)


def write_contact_sheet(ds: VideoDataset, path: str | Path, n: int = 12, seed: int = 0):
    """PNG preview: n random videos as rows, frames as columns."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(len(ds), size=min(n, len(ds)), replace=False))
    rows = []
    for i in ids:
        v = ds.get(int(i))
        cols = v[np.round(np.linspace(0, len(v) - 1, 8)).astype(int)]
        cols = [np.asarray(Image.fromarray(c).resize((96, 96), Image.BILINEAR)) for c in cols]
        rows.append(np.concatenate(cols, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)
    return ids


def write_gif(ds: VideoDataset, i: int, path: str | Path, fps: int = 12):
    from PIL import Image
    fr = [Image.fromarray(f) for f in ds.get(i)]
    fr[0].save(path, save_all=True, append_images=fr[1:], duration=int(1000 / fps), loop=0)
