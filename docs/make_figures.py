"""Regenerate the figures in docs/figures/ from the tables in docs/results/.

    python docs/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physprobe import plots  # noqa: E402  (sets the shared matplotlib style)
from physprobe.plots import AXIS, CAT, INK, INK2, MUTED, NEUTRAL, SURFACE  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

RES = ROOT / "docs" / "results"
FIG = ROOT / "docs" / "figures"
PEZ = 8  # one-third depth of the 24-layer ViT-L


def curve(name, target, col="r2_mean"):
    d = pd.read_csv(RES / name)
    d = d[d.target == target].sort_values("layer")
    return d["layer"].to_numpy(), d[col].to_numpy()


def line(ax, x, y, color, label, ls="-", marker="o", lw=2.0):
    ax.plot(x, y, color=color, ls=ls, marker=marker, lw=lw, label=label, markeredgecolor=SURFACE,
            markeredgewidth=0.8, markersize=4.5)


def finish(ax, ylabel, ylim=(-0.1, 1.03), pez=True):
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(0, 24, 2))
    if pez:
        ax.axvline(PEZ, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("->", FIG / name)


def fig_shortcut():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0), sharey=True)
    ax = axes[0]
    line(ax, *curve("velocity_probe.csv", "direction_deg"), CAT[0], "velocity, pretrained")
    line(ax, *curve("velocity_probe_random.csv", "direction_deg"), CAT[0], "velocity, random weights", ls=(0, (4, 2)))
    line(ax, *curve("velocity_centered_probe.csv", "direction_deg"), CAT[1], "centred, pretrained")
    line(ax, *curve("velocity_centered_probe_random.csv", "direction_deg"), CAT[1], "centred, random weights",
         ls=(0, (4, 2)))
    finish(ax, "direction, cross-validated R²", (-0.15, 1.03))
    ax.set_title("Ordinary videos: motion visible inside each token")
    ax.legend(loc="lower right")
    ax = axes[1]
    s = pd.read_csv(RES / "velocity_stepped_probe_r2.csv")
    r = pd.read_csv(RES / "velocity_stepped_probe_random_r2.csv")
    line(ax, s.layer, s.direction_deg, CAT[0], "direction, pretrained")
    line(ax, s.layer, s.speed, CAT[2], "speed, pretrained")
    ax.plot(r.layer, r.direction_deg, ls="none", marker="s", color=CAT[0], markerfacecolor=SURFACE, markersize=6,
            label="direction, random weights")
    ax.plot(r.layer, r.speed.clip(lower=-0.1), ls="none", marker="s", color=CAT[2], markerfacecolor=SURFACE,
            markersize=6, label="speed, random weights (clipped at −0.1)")
    finish(ax, "", (-0.15, 1.03))
    ax.set_title("Stepped videos: motion only across tokens")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42))
    fig.text(0.01, -0.02, "Dotted line: one-third depth (layer 8). Ridge or AdamW linear probes, grouped 5-fold "
             "cross-validation; see docs/results/README.md.", fontsize=8, color=MUTED)
    save(fig, "fig1_shortcut.png")


def fig_shuffle():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.9), sharey=True)
    ax = axes[0]
    line(ax, *curve("shuffled_frames_probe.csv", "shuffled", "accuracy_mean"), CAT[3], "single frames shuffled")
    line(ax, *curve("shuffled_pairs_probe.csv", "shuffled", "accuracy_mean"), CAT[0], "tubelet pairs shuffled")
    finish(ax, "intact vs shuffled, accuracy", (0.4, 1.03))
    ax.axhline(0.5, color=AXIS, lw=1)
    ax.set_title("Detecting temporal order")
    ax.legend(loc="lower right")
    ax = axes[1]
    line(ax, *curve("ablate_shuffle_L5-6_base.csv", "shuffled", "accuracy_mean"), INK, "no ablation")
    line(ax, *curve("ablate_shuffle_L5-6_combined.csv", "shuffled", "accuracy_mean"), CAT[0],
         "local attention removed at layers 5–6")
    ax.axvspan(4.5, 6.5, color=NEUTRAL, alpha=0.3, lw=0, zorder=0)
    finish(ax, "", (0.4, 1.03))
    ax.axhline(0.5, color=AXIS, lw=1)
    ax.set_title("Tubelet-pair shuffling, with and without ablation")
    ax.legend(loc="lower right")
    save(fig, "fig2_temporal_order.png")


def fig_ablations():
    a = pd.read_csv(RES / "ablation_readouts.csv")
    rows = []
    for name, f in [("ordinary, layers 7–8, combined (AND)", "ablate_velocity_L7-8_and_combined_s3_t1.csv"),
                    ("ordinary, layers 7–8, combined (OR)", "ablate_velocity_L7-8_or_combined_s3_t1.csv")]:
        d = pd.read_csv(RES / f)
        b = pd.read_csv(RES / "ablate_velocity_L7-8_and_base.csv")
        g = lambda df, t: float(df[(df.target == t) & (df.layer == 10)]["r2_mean"].iloc[0])
        rows.append((name, g(b, "direction_deg"), g(d, "direction_deg"), g(b, "speed"), g(d, "speed")))
    sel = lambda stim, lay, cond, t: float(a[(a.stimuli == stim) & (a.ablated_layers == lay) & (a.condition == cond)
                                              & (a.target == t) & (a.read_layer == 10)].score.iloc[0])
    rows.append(("ordinary, layers 1–4, combined (OR)", sel("velocity", "1-4", "base", "direction_deg"),
                 sel("velocity", "1-4", "combined s=3 t=1", "direction_deg"), sel("velocity", "1-4", "base", "speed"),
                 sel("velocity", "1-4", "combined s=3 t=1", "speed")))
    base_d, base_s = sel("velocity_stepped", "7-8", "base", "direction_deg"), sel("velocity_stepped", "7-8", "base", "speed")
    for cond, label in (("spatial s=7", "spatial s=7"), ("temporal t=3", "temporal t=3"),
                        ("combined s=3 t=1", "combined (OR)")):
        rows.append((f"stepped, layers 7–8, {label}", base_d,
                     sel("velocity_stepped", "7-8", cond, "direction_deg"), base_s,
                     sel("velocity_stepped", "7-8", cond, "speed")))
    rows.append(("stepped, layers 4–6, combined (OR)", base_d, sel("velocity_stepped", "4-6", "combined s=3 t=1",
                                                                  "direction_deg"), base_s,
                 sel("velocity_stepped", "4-6", "combined s=3 t=1", "speed")))
    df = pd.DataFrame(rows, columns=["run", "dir_base", "dir_abl", "spd_base", "spd_abl"])
    df.to_csv(RES / "ablation_summary_L10.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2), sharey=True)
    y = np.arange(len(df))[::-1]
    for ax, (b, v, title) in zip(axes, (("dir_base", "dir_abl", "Direction (read at layer 10)"),
                                        ("spd_base", "spd_abl", "Speed (read at layer 10)"))):
        ax.barh(y + 0.18, df[b], height=0.34, color=NEUTRAL, label="no ablation")
        ax.barh(y - 0.18, df[v], height=0.34, color=CAT[0], label="ablated")
        for yi, val in zip(y, df[v]):
            ax.annotate(f"{val:.2f}", (val, yi - 0.18), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK2)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("cross-validated R²")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, df["run"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.04))
    save(fig, "fig3_ablations.png")


def fig_attention():
    a = pd.read_csv(RES / "attention_distance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.6))
    for ax, (m, lo, title, unit) in zip(axes, (("spatial_mean", "spatial_min_head", "Spatial", "patches"),
                                               ("temporal_mean", "temporal_min_head", "Temporal", "tubelets"))):
        line(ax, a.layer, a[m], INK, "layer mean")
        line(ax, a.layer, a[lo], CAT[0], "most local head", ls=(0, (4, 2)))
        ax.axvspan(4.5, 6.5, color=NEUTRAL, alpha=0.3, lw=0, zorder=0)
        ax.set_xlabel("layer")
        ax.set_ylabel(f"attention distance ({unit})")
        ax.set_xticks(range(0, 24, 2))
        ax.set_ylim(bottom=0)
        ax.set_title(f"{title} attention distance")
        ax.legend(loc="lower right")
    fig.text(0.01, -0.03, "Shaded: layers 5–6, where tubelet-order detection jumps (Fig. 2). 64 `velocity` videos.",
             fontsize=8, color=MUTED)
    save(fig, "fig4_attention_distance.png")


def fig_dimensions():
    v = pd.read_csv(RES / "velocity_dimension_by_layer.csv")
    s = pd.read_csv(RES / "velocity_stepped_dimension.csv")
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    for df, color, name in ((v, CAT[0], "ordinary"), (s, CAT[1], "stepped")):
        d = df[df.task == "direction"]
        ax.plot(d.layer, d.probe_sequence_dim, color=color, marker="o", lw=2, label=f"{name}: probe-sequence count",
                markeredgecolor=SURFACE)
    ax.axhline(2, color=INK, lw=1.4)
    ax.annotate("2 dimensions: erasing them leaves R² = 0.00 at every layer, both stimulus sets", (14, 2),
                xytext=(0, 14), textcoords="offset points", ha="center", fontsize=8.5, color=INK2,
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"))
    ax.set_xlabel("layer")
    ax.set_ylabel("dimensions for direction")
    ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(0, None)
    ax.set_title("Direction subspace size: two estimates")
    ax.legend(loc="upper left")
    save(fig, "fig5_dimensions.png")


def fig_steering():
    c = pd.read_csv(RES / "steering_curves.csv")
    p = pd.read_csv(RES / "velocity_steering_propagation_L8.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.8))
    for ax, (stim, layer) in zip(axes[:2], (("velocity", 8), ("velocity_stepped", 10))):
        d = c[c.stimuli == stim]
        ps = d[d.method == "probe subspace"]
        line(ax, ps.dims, ps.mae_to_target_deg, CAT[0], "orthogonal-probe subspace")
        for i, m in enumerate(("mean shift top-2 axes", "mean shift all axes")):
            r = d[d.method == m].iloc[0]
            ax.axhline(r.mae_to_target_deg, color=CAT[1 + i], lw=1.4,
                       label=f"difference of means, {int(r.dims)} dims: {r.mae_to_target_deg:.1f}°")
        none = d[d.method == "none"].iloc[0].mae_to_target_deg
        ax.axhline(none, color=MUTED, lw=1, ls=(0, (3, 3)), label=f"no steering: {none:.1f}°")
        ax.set_xlabel("dimensions steered")
        ax.set_ylabel("decoded error to 90° (deg)")
        ax.set_ylim(0, 100)
        ax.set_title(f"{'Ordinary' if stim == 'velocity' else 'Stepped'} videos, layer {layer}")
        ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62), fontsize=7.5)
    ax = axes[2]
    line(ax, p.layer, p.not_steered, MUTED, "not steered")
    line(ax, p.layer, p.difference_of_means, CAT[1], "difference of means (7 dims)")
    line(ax, p.layer, p.probe_subspace_39, CAT[0], "probe subspace (78 dims)")
    ax.axvline(8, color=INK, lw=1)
    ax.set_xlabel("layer read out")
    ax.set_ylabel("decoded error to 90° (deg)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(0, 100)
    ax.set_title("Ordinary videos: steered at layer 8, read downstream")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "fig6_steering.png")


def fig_shape():
    m = pd.read_csv(RES / "stepped24_manifold_direction.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.8))
    ax = axes[0]
    line(ax, m.layer, m.reliability, INK, "split-half reliability")
    line(ax, m.layer, m.circ_corr, CAT[0], "ring order (PC1–2)")
    ax.axhline(0.3, color=AXIS, lw=1)
    finish(ax, "", (-0.05, 1.03))
    ax.set_title("Is there an ordered ring?")
    ax.legend(loc="upper left")
    ax = axes[1]
    for i, (col, name) in enumerate((("harmonic1", "harmonic 1: direction (once per cycle)"),
                                     ("harmonic2", "harmonic 2: axis (opposite directions alike)"),
                                     ("harmonic4", "harmonic 4: four-fold pattern"))):
        line(ax, m.layer, m[col], CAT[[0, 1, 3][i]], name)
    finish(ax, "share of across-direction variance", (-0.02, 1.0))
    ax.set_title("What kind of loop (blank: reliability < 0.3)")
    ax.legend(loc="upper left", fontsize=8)
    fig.text(0.01, -0.03, "Stepped videos, 24 directions × 3 speeds × 5 positions (360 videos).", fontsize=8,
             color=MUTED)
    save(fig, "fig7_direction_shape.png")


if __name__ == "__main__":
    fig_shortcut()
    fig_shuffle()
    fig_ablations()
    fig_attention()
    fig_dimensions()
    fig_steering()
    fig_shape()
