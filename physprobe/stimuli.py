"""Controlled-physics video stimuli.

A small deterministic numpy renderer (top-down view of balls on a floor) plus a
registry of *stimulus families*. A family is a factorial design: a grid of
factors, and a function that turns one row of that grid into a `Scene`.

Geometry follows the paper's toy-ball set (App. A.1): 8 x 8 m floor seen from an
overhead camera, sphere radius 0.3 m, 16 frames at 24 fps, zero friction. With
zero friction and an overhead camera the physics is closed-form, so no physics
engine is needed: x(t) = x0 + v t (velocity set) or x0 + a t^2 / 2 (acceleration
set). What differs from the paper is the *look* (Kubric/Blender shading and
perspective vs. this flat-shaded orthographic renderer).

Conventions: world x points right, world y points UP on screen, angles are
counter-clockwise from +x, so 0 deg = rightward and 90 deg = upward motion.
"""
from __future__ import annotations

import colorsys
import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

FPS = 24.0
VIEW_M = 8.0  # side of the visible square, metres
BALL_RADIUS = 0.3
DEFAULT_COLOR = (0.82, 0.16, 0.14)

# --------------------------------------------------------------------------- scene


@dataclass
class Ball:
    pos: np.ndarray  # [T, 2] metres
    radius: float | np.ndarray = BALL_RADIUS  # scalar or [T]
    color: tuple | np.ndarray = DEFAULT_COLOR  # rgb in 0..1, or [T, 3]
    visible: np.ndarray | None = None  # [T] bool; None = always


@dataclass
class Occluder:
    """Opaque axis-aligned panel hovering above the floor (drawn over balls)."""

    rect: tuple  # (x0, y0, x1, y1) metres
    color: tuple = (0.25, 0.27, 0.32)


@dataclass
class Scene:
    balls: list
    occluders: list = field(default_factory=list)
    floor: str = "noise"  # noise | plain | checker
    floor_seed: int = 0
    view_m: float = VIEW_M


_FLOOR_CACHE: dict = {}
_LIGHT = np.array([-0.45, -0.55, 0.70])  # image coords (x right, y down, z to camera)
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)


def _floor(res: int, style: str, seed: int) -> np.ndarray:
    key = (res, style, seed)
    if key in _FLOOR_CACHE:
        return _FLOOR_CACHE[key]
    base = np.array([0.62, 0.62, 0.60], dtype=np.float32)
    img = np.ones((res, res, 3), dtype=np.float32) * base
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32) / res
    if style == "noise":
        # Smooth, fixed low-frequency texture: gives the model a static reference
        # frame without adding anything that moves.
        rng = np.random.default_rng(seed)
        tex = np.zeros((res, res), dtype=np.float32)
        for f in (2, 4, 8, 16):
            ph = rng.uniform(0, 2 * np.pi, size=4)
            tex += (np.sin(2 * np.pi * f * xx + ph[0]) * np.sin(2 * np.pi * f * yy + ph[1])
                    + np.sin(2 * np.pi * f * (xx + yy) / 1.4 + ph[2])
                    * np.sin(2 * np.pi * f * (xx - yy) / 1.4 + ph[3])) / f
        tex = tex / np.abs(tex).max()
        img = img * (1.0 + 0.10 * tex[..., None])
    elif style == "checker":
        chk = ((np.floor(xx * 8) + np.floor(yy * 8)) % 2).astype(np.float32)
        img = img * (0.93 + 0.14 * chk[..., None])
    elif style != "plain":
        raise ValueError(f"unknown floor style {style!r}")
    img = np.clip(img, 0, 1)
    _FLOOR_CACHE[key] = img
    return img


def _at(v, t, n=None):
    """Value of a possibly time-varying attribute at frame t."""
    a = np.asarray(v, dtype=np.float32)
    if n is None:  # scalar attr
        return float(a[t]) if a.ndim == 1 else float(a)
    return a[t] if a.ndim == 2 else a


def _draw_disc(img, cx, cy, r, color, shade=True):
    res = img.shape[0]
    pad = 2.5 * r + 3
    x0, x1 = int(max(0, np.floor(cx - pad))), int(min(res, np.ceil(cx + pad)))
    y0, y1 = int(max(0, np.floor(cy - pad))), int(min(res, np.ceil(cy + pad)))
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    patch = img[y0:y1, x0:x1]
    if shade:  # soft contact shadow, offset away from the light
        sx, sy = cx - _LIGHT[0] * 0.55 * r, cy - _LIGHT[1] * 0.55 * r
        ds = np.sqrt((xs - sx) ** 2 + (ys - sy) ** 2) / (1.15 * r)
        patch *= (1.0 - 0.38 * np.exp(-(ds ** 4)))[..., None]
    u, v = (xs - cx) / r, (ys - cy) / r
    d2 = u * u + v * v
    alpha = np.clip((r - np.sqrt(d2) * r) + 0.5, 0.0, 1.0)  # 1 px anti-aliased edge
    col = np.asarray(color, dtype=np.float32)
    if shade:
        nz = np.sqrt(np.clip(1.0 - d2, 0.0, 1.0))
        lam = np.clip(u * _LIGHT[0] + v * _LIGHT[1] + nz * _LIGHT[2], 0.0, 1.0)
        spec = np.clip(lam, 0, 1) ** 24
        rgb = col * (0.38 + 0.62 * lam)[..., None] + 0.35 * spec[..., None]
    else:
        rgb = np.broadcast_to(col, patch.shape)
    patch[:] = patch * (1 - alpha[..., None]) + np.clip(rgb, 0, 1) * alpha[..., None]


def render(scene: Scene, T: int, res: int = 256) -> np.ndarray:
    """Render a scene to uint8 [T, res, res, 3]."""
    ppm = res / scene.view_m  # pixels per metre
    floor = _floor(res, scene.floor, scene.floor_seed)
    out = np.empty((T, res, res, 3), dtype=np.uint8)
    for t in range(T):
        img = floor.copy()
        for b in scene.balls:
            if b.visible is not None and not bool(b.visible[t]):
                continue
            x, y = b.pos[t]
            cx = (x / scene.view_m + 0.5) * res
            cy = (0.5 - y / scene.view_m) * res
            _draw_disc(img, cx, cy, _at(b.radius, t) * ppm, _at(b.color, t, 3))
        for o in scene.occluders:
            x0, y0, x1, y1 = o.rect
            px0, px1 = sorted(((x0 / scene.view_m + 0.5) * res, (x1 / scene.view_m + 0.5) * res))
            py0, py1 = sorted(((0.5 - y0 / scene.view_m) * res, (0.5 - y1 / scene.view_m) * res))
            a, b_, c, d = (int(round(v)) for v in (py0, py1, px0, px1))
            a, b_, c, d = max(a, 0), min(b_, res), max(c, 0), min(d, res)
            img[a:b_, c:d] = np.asarray(o.color, dtype=np.float32)
            img[a:b_, c:min(c + 1, d)] *= 0.8  # hairline edge so the panel reads as an object
            img[a:min(a + 1, b_), c:d] *= 0.8
        out[t] = np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------------ families


@dataclass
class Family:
    name: str
    doc: str
    grid: dict  # factor -> list of levels (crossed)
    build: Callable  # (row: dict, T: int) -> Scene
    targets: dict  # column -> probe kind: circular | scalar | binary | vector
    group: str  # default grouping column for cross-validation
    n_starts: int = 0  # >0: cross the grid with this many random start positions
    start_range: float = 2.0
    derive: Callable | None = None  # (row) -> dict of extra label columns
    row_seeded: bool = False  # give each row its own rng seed column


FAMILIES: dict[str, Family] = {}


def register_family(fam: Family) -> Family:
    FAMILIES[fam.name] = fam
    return fam


def _times(T):
    return np.arange(T, dtype=np.float64) / FPS


def _unit(deg):
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)])


def _hue_rgb(h_deg, s=0.85, v=0.9):
    return colorsys.hsv_to_rgb((h_deg % 360) / 360.0, s, v)


def motion_labels(vx, vy, view_m=VIEW_M, res=256):
    """Velocity labels in m/s and in px/frame at `res` (the paper quotes both units)."""
    k = (res / view_m) / FPS
    sp = float(np.hypot(vx, vy))
    return {"vx": vx, "vy": vy, "speed_mps": sp, "vx_pxf": vx * k, "vy_pxf": -vy * k,
            "speed_pxf": sp * k}


# -- 1. velocity (paper, App. A.1.2): 8 directions x 7 speeds x 7 starts = 392 videos
def _build_velocity(row, T):
    v = _unit(row["direction_deg"]) * row["speed"]
    p0 = np.array([row["start_x"], row["start_y"]])
    return Scene([Ball(p0 + _times(T)[:, None] * v)])


def _derive_velocity(row):
    v = _unit(row["direction_deg"]) * row["speed"]
    return motion_labels(v[0], v[1])


register_family(Family(
    name="velocity",
    doc="Paper's velocity set: constant-velocity ball. direction x speed x start position.",
    grid={"direction_deg": list(range(0, 360, 45)), "speed": [1, 2, 3, 4, 5, 6, 7]},
    build=_build_velocity, derive=_derive_velocity, n_starts=7,
    targets={"direction_deg": "circular", "speed": "scalar", "vx,vy": "vector"},
    group="start_id",
))


# -- 1b. velocity_centered (control, not in the paper): same design, but each trajectory
#        passes through its position mid-clip instead of starting there. In the paper's
#        design the ball's mean position (start + v * 0.33 s) depends on direction, so a
#        probe could read direction from position alone; here it cannot.
def _build_velocity_centered(row, T):
    v = _unit(row["direction_deg"]) * row["speed"]
    t = _times(T)
    p0 = np.array([row["start_x"], row["start_y"]])
    return Scene([Ball(p0 + (t - t.mean())[:, None] * v)])


register_family(Family(
    name="velocity_centered",
    doc="Velocity set with trajectories centred on the position (no mean-position shortcut).",
    grid={"direction_deg": list(range(0, 360, 45)), "speed": [1, 2, 3, 4, 5, 6, 7]},
    build=_build_velocity_centered, derive=_derive_velocity, n_starts=7,
    targets={"direction_deg": "circular", "speed": "scalar", "vx,vy": "vector"},
    group="start_id",
))


# -- 1c. velocity_stepped (control, not in the paper): V-JEPA 2 turns each pair of frames
#        (one tubelet) into one token, so in ordinary videos motion is visible inside a
#        single token. Here the ball is frozen for the two frames of each tubelet and
#        jumps between tubelets: direction can only be computed by comparing tokens.
TUBELET = 2


def _build_velocity_stepped(row, T):
    v = _unit(row["direction_deg"]) * row["speed"]
    t = (np.arange(T) // TUBELET * TUBELET) / FPS  # frozen inside each frame pair
    p0 = np.array([row["start_x"], row["start_y"]])
    return Scene([Ball(p0 + (t - t.mean())[:, None] * v)])


register_family(Family(
    name="velocity_stepped",
    doc="Velocity set with the ball frozen inside each 2-frame tubelet: motion is only "
        "visible across tokens, never within one. Centred trajectories.",
    grid={"direction_deg": list(range(0, 360, 45)), "speed": [1, 2, 3, 4, 5, 6, 7]},
    build=_build_velocity_stepped, derive=_derive_velocity, n_starts=7,
    targets={"direction_deg": "circular", "speed": "scalar", "vx,vy": "vector"},
    group="start_id",
))


# -- 2. acceleration (paper): 8 directions x 5 accelerations x 7 starts = 280 videos
def _build_accel(row, T):
    a = _unit(row["direction_deg"]) * row["accel"]
    p0 = np.array([row["start_x"], row["start_y"]])
    return Scene([Ball(p0 + 0.5 * (_times(T) ** 2)[:, None] * a)])


def _derive_accel(row):
    a = _unit(row["direction_deg"]) * row["accel"]
    return {"ax": a[0], "ay": a[1]}


register_family(Family(
    name="acceleration",
    doc="Paper's acceleration set: ball starts at rest under a constant force. "
        "direction x acceleration magnitude x start position.",
    grid={"direction_deg": list(range(0, 360, 45)), "accel": [2, 4, 6, 8, 10]},
    build=_build_accel, derive=_derive_accel, n_starts=7,
    targets={"direction_deg": "circular", "accel": "scalar", "ax,ay": "vector"},
    group="start_id",
))


# -- 3. position: where is the ball? Expect a 2-D sheet (plane) manifold.
def _build_position(row, T):
    p = np.array([row["x"], row["y"]])
    drift = _unit(row["drift_deg"]) * 0.5  # slow drift so it is still a video, not a still
    return Scene([Ball(p + (_times(T) - _times(T).mean())[:, None] * drift)])


register_family(Family(
    name="position",
    doc="Ball at a grid of floor positions with a slow drift. Tests for a planar (x, y) map.",
    grid={"x": list(np.linspace(-3, 3, 7).round(3)), "y": list(np.linspace(-3, 3, 7).round(3)),
          "drift_deg": [0, 90, 180, 270]},
    build=_build_position,
    targets={"x": "scalar", "y": "scalar", "x,y": "vector"},
    group="drift_deg",
))


# -- 4. hue: a circular variable that has nothing to do with motion.
def _build_hue(row, T):
    v = _unit(row["direction_deg"]) * 3.0
    p0 = np.array([row["start_x"], row["start_y"]])
    return Scene([Ball(p0 + _times(T)[:, None] * v, color=_hue_rgb(row["hue_deg"]))])


register_family(Family(
    name="hue",
    doc="Ball colour sweeps the hue wheel while motion varies as a nuisance. A circular "
        "control variable: is the direction ring special, or does any circular variable ring?",
    grid={"hue_deg": list(range(0, 360, 20)), "direction_deg": [0, 90, 180, 270]},
    build=_build_hue, n_starts=4,
    targets={"hue_deg": "circular", "direction_deg": "circular"},
    group="start_id",
))


# -- 5. size: a 1-D magnitude variable (expect a line / curve).
def _build_size(row, T):
    v = _unit(row["direction_deg"]) * 3.0
    p0 = np.array([row["start_x"], row["start_y"]])
    return Scene([Ball(p0 + _times(T)[:, None] * v, radius=row["radius"])])


register_family(Family(
    name="size",
    doc="Ball radius sweep with motion as a nuisance. A scalar (non-circular) control.",
    grid={"radius": [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7], "direction_deg": [0, 90, 180, 270]},
    build=_build_size, n_starts=4,
    targets={"radius": "scalar", "direction_deg": "circular"},
    group="start_id",
))


# -- 6. orbit: circular motion. Phase is circular, handedness is binary.
def _build_orbit(row, T):
    ph = np.deg2rad(row["phase_deg"]) + np.deg2rad(row["omega_dps"]) * _times(T)
    c = np.array([row["start_x"], row["start_y"]]) * 0.5
    return Scene([Ball(c + row["orbit_r"] * np.stack([np.cos(ph), np.sin(ph)], 1))])


def _derive_orbit(row):
    return {"ccw": int(row["omega_dps"] > 0), "ang_speed": abs(row["omega_dps"]),
            "lin_speed": abs(np.deg2rad(row["omega_dps"])) * row["orbit_r"]}


register_family(Family(
    name="orbit",
    doc="Ball on a circular path: start phase x angular velocity (sign = handedness) x "
        "orbit radius. Direction changes within the clip, so 'direction' is no longer one "
        "number per video; look at phase rings and time trajectories instead.",
    grid={"phase_deg": list(range(0, 360, 30)), "omega_dps": [-240, -120, 120, 240],
          "orbit_r": [1.2, 2.0]},
    build=_build_orbit, derive=_derive_orbit, n_starts=4, start_range=1.5,
    targets={"phase_deg": "circular", "ccw": "binary", "ang_speed": "scalar"},
    group="start_id",
))


# -- 7. two_balls: two independent directions. Factorised code => torus (two rings in
#       orthogonal planes); conjunctive code => power at mixed harmonics.
def _build_two(row, T):
    rng = np.random.default_rng(int(row["seed"]))
    t = _times(T)[:, None]
    va = _unit(row["dir_a_deg"]) * row["speed"]
    vb = _unit(row["dir_b_deg"]) * row["speed"]
    for _ in range(500):  # rejection-sample starts so the balls never overlap
        pa = rng.uniform(-2.5, 2.5, 2)
        pb = rng.uniform(-2.5, 2.5, 2)
        d = np.linalg.norm((pa + t * va) - (pb + t * vb), axis=1)
        if d.min() > 2 * BALL_RADIUS + 0.4:
            break
    return Scene([Ball(pa + t * va, color=(0.82, 0.16, 0.14)),
                  Ball(pb + t * vb, color=(0.13, 0.36, 0.80))])


register_family(Family(
    name="two_balls",
    doc="Red and blue ball with independent directions (8 x 8) and non-overlapping paths. "
        "Tests binding / factorisation: is (dir_a, dir_b) a torus?",
    grid={"dir_a_deg": list(range(0, 360, 45)), "dir_b_deg": list(range(0, 360, 45)),
          "speed": [2.0, 3.5], "rep": [0, 1, 2]},
    build=_build_two, row_seeded=True,
    targets={"dir_a_deg": "circular", "dir_b_deg": "circular"},
    group="rep",
))


# -- 8. permanence: a home-made violation-of-expectation set (NOT IntPhys).
_VIOLATIONS = ("vanish", "teleport", "morph")


def _build_perm(row, T):
    # Seeded per (scene, violation) so the possible/impossible pair shares every
    # parameter, while different violation types get different scenes.
    rng = np.random.default_rng([int(row["scene_seed"]), _VIOLATIONS.index(row["violation"])])
    axis = int(rng.integers(0, 2))  # 0: moves along x, 1: along y
    sign = float(rng.choice([-1.0, 1.0]))
    speed = float(rng.uniform(5.0, 6.5))
    lateral = float(rng.uniform(-1.5, 1.5))
    occ_c = float(rng.uniform(-0.4, 0.4))
    hue = float(rng.uniform(0, 360))
    t = _times(T)
    t_mid = t[T // 2 - 1] + 0.5 / FPS  # ball centre crosses the occluder centre mid-clip
    along = occ_c + sign * speed * (t - t_mid)
    lat = np.full(T, lateral)
    radius = np.full(T, BALL_RADIUS)
    visible = np.ones(T, dtype=bool)
    after = t > t_mid
    if not row["possible"]:
        if row["violation"] == "vanish":
            visible[after] = False
        elif row["violation"] == "teleport":
            lat[after] += 1.0 if lateral < 0 else -1.0
        elif row["violation"] == "morph":
            radius[after] = BALL_RADIUS * 1.4
    pos = np.stack([along, lat], 1) if axis == 0 else np.stack([lat, along], 1)
    hw, hh = 0.85, 3.2  # half-width along the motion axis, half-height across it
    rect = (occ_c - hw, -hh, occ_c + hw, hh) if axis == 0 else (-hh, occ_c - hw, hh, occ_c + hw)
    return Scene([Ball(pos, radius=radius, color=_hue_rgb(hue), visible=visible)],
                 occluders=[Occluder(rect)])


def _derive_perm(row):
    return {"scene": f"{row['violation']}_{int(row['scene_id'])}"}


register_family(Family(
    name="permanence",
    doc="Violation-of-expectation proxy: a ball passes behind a panel and either continues "
        "(possible) or vanishes / jumps sideways / changes size while hidden (impossible). "
        "Matched pairs are identical up to the mid-clip breakpoint. A stand-in for IntPhys.",
    grid={"violation": list(_VIOLATIONS), "scene_id": list(range(40)), "possible": [1, 0]},
    build=_build_perm, derive=_derive_perm,
    targets={"possible": "binary"},
    group="scene",
))


# ------------------------------------------------------------------------ manifests


def parse_levels(text: str) -> list:
    """'1,2,3' or 'a:b:step' (numeric range, end exclusive) -> list of levels."""
    if ":" in text and "," not in text:
        a, b, s = (float(v) for v in text.split(":"))
        vals = list(np.arange(a, b, s))
    else:
        vals = []
        for tok in text.split(","):
            try:
                vals.append(float(tok))
            except ValueError:
                vals.append(tok)
    return [int(v) if isinstance(v, float) and float(v).is_integer() else v for v in vals]


def make_manifest(family: str, grid_overrides: dict | None = None, n_starts: int | None = None,
                  seed: int = 0) -> pd.DataFrame:
    """Cross the family's factors (and start positions) into one row per video."""
    fam = FAMILIES[family]
    grid = dict(fam.grid)
    for k, v in (grid_overrides or {}).items():
        if k not in grid:
            raise KeyError(f"{family!r} has no factor {k!r}; factors: {list(grid)}")
        grid[k] = v
    ns = fam.n_starts if n_starts is None else n_starts
    rng = np.random.default_rng(seed)
    starts = rng.uniform(-fam.start_range, fam.start_range, size=(max(ns, 1), 2))
    rows = []
    for combo in itertools.product(*grid.values()):
        base = dict(zip(grid.keys(), combo))
        for sid in range(max(ns, 1)):
            row = dict(base)
            if ns > 0:
                row.update(start_id=sid, start_x=float(starts[sid, 0]), start_y=float(starts[sid, 1]))
            rows.append(row)
    df = pd.DataFrame(rows)
    if fam.row_seeded:
        df["seed"] = np.arange(len(df)) + 1000 * (seed + 1)
    if "scene_id" in df.columns:  # scene-level randomness shared by matched videos
        df["scene_seed"] = df["scene_id"].astype(int) + 7919 * (seed + 1)
    if fam.derive is not None:
        extra = pd.DataFrame([fam.derive(r) for r in df.to_dict("records")])
        df = pd.concat([df, extra], axis=1)
    df.insert(0, "video_id", np.arange(len(df)))
    return df


def render_row(family: str, row: dict, T: int = 16, res: int = 256, floor: str = "noise",
               view_m: float = VIEW_M) -> np.ndarray:
    scene = FAMILIES[family].build(row, T)
    scene.floor, scene.view_m = floor, view_m
    return render(scene, T, res)
