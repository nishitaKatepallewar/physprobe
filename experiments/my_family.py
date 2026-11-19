"""Template: add your own stimulus family.

A family = a grid of factors (crossed into one video per combination) + a function
that turns one row of that grid into a Scene. Register it, then use it like the
built-ins:

    python -m physprobe make bounce --out data/bounce --plugin experiments/my_family.py
    python -m physprobe extract --data data/bounce --out runs/bounce/acts --time-layers thirds
    python -m physprobe probe --acts runs/bounce/acts --out runs/bounce/probe
    python -m physprobe manifold --acts runs/bounce/acts --out runs/bounce/manifold --by direction_deg

This example: a ball heads rightward at some angle and either bounces off a wall
or (no wall) carries on. Question it lets you ask: once direction is decodable, is
it the incoming direction, the outgoing one, or neither?
"""
import numpy as np

from physprobe.stimuli import BALL_RADIUS, FPS, Ball, Family, Occluder, Scene, register_family

WALL_X = 1.5


def build(row, T):
    """row: dict of this video's factor levels (+ start_x, start_y). Return a Scene."""
    t = np.arange(T)[:, None] / FPS
    ang = np.deg2rad(row["direction_deg"])
    v = row["speed"] * np.array([np.cos(ang), np.sin(ang)])
    pos = np.array([row["start_x"] - 1.5, row["start_y"]]) + t * v
    walls = []
    if row["wall"]:
        limit = WALL_X - BALL_RADIUS
        hit = pos[:, 0] > limit
        pos[hit, 0] = 2 * limit - pos[hit, 0]  # elastic reflection
        walls = [Occluder((WALL_X, -4.0, WALL_X + 0.15, 4.0))]
    return Scene([Ball(pos)], occluders=walls)


def derive(row):
    """Extra label columns computed from the row (optional)."""
    out_deg = (180 - row["direction_deg"]) % 360 if row["wall"] else row["direction_deg"] % 360
    return {"direction_out_deg": out_deg}


register_family(Family(
    name="bounce",
    doc="Ball heading right at an angle; with a wall it reflects mid-clip.",
    grid={"direction_deg": [-60, -40, -20, 0, 20, 40, 60], "speed": [5.0], "wall": [0, 1]},
    build=build,
    derive=derive,
    n_starts=6,            # cross the grid with 6 random start positions ...
    start_range=1.0,       # ... drawn from [-1, 1]^2 metres
    # what `probe` decodes by default: column -> circular | scalar | binary | "a,b" -> vector
    targets={"direction_deg": "scalar", "direction_out_deg": "circular", "wall": "binary"},
    group="start_id",      # cross-validation never splits a group across train and test
))
