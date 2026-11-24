"""Figures. One visual system for every plot: light surface, recessive grid, a fixed
categorical colour order (colour follows the series, never its rank), one-hue
sequential maps for magnitudes, and a cyclic map only for genuinely circular
variables (always paired with direct angle labels)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#b9b8b0"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef4fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SEQ.set_bad("#efeee9")
# Cyclic map for angles: the categorical hues arranged as a wheel, all mid-lightness
# so every angle stays visible on the light surface (and wraps without a seam).
CYCLIC = LinearSegmentedColormap.from_list(
    "wheel", ["#2a78d6", "#6250d6", "#e87ba4", "#e34948", "#eb6834", "#eda100", "#1baf7a", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "legend.frameon": False, "font.family": "DejaVu Sans", "lines.linewidth": 2.0,
    "lines.markersize": 4.5, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def _save(fig, path):
    from matplotlib.ticker import MaxNLocator
    for ax in fig.axes:  # layers and patch indices are integers
        if ax.get_xlabel().startswith(("layer", "patch", "probe index", "orthogonal probes")):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if ax.get_ylabel().startswith(("layer", "patch")):
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _mark_pez(ax, pez, top=True):
    if pez is None:
        return
    ax.axvline(pez, color=MUTED, lw=1.0, ls=(0, (3, 3)), zorder=1)
    if top:
        ax.annotate("1/3 depth", (pez, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(3, 4), textcoords="offset points", va="bottom", fontsize=8, color=MUTED)


def _series(ax, x, mean, sd, color, label):
    ax.plot(x, mean, color=color, marker="o", label=label, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=0.8)
    if sd is not None:
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.16, lw=0, zorder=2)


LINESTYLES = ["-", (0, (4, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2))]


def layer_curves(summary, path, title="Layerwise linear probes", pez=None, extra=None):
    """Probe score vs layer. R^2 targets and accuracy targets get separate panels
    (never a second y-axis). `extra`: {label: summary} overlays, e.g. a control run;
    colour always identifies the target, line style identifies the run."""
    runs = {"": summary, **(extra or {})}
    kinds = [k for k in ("r2", "accuracy")
             if any(f"{k}_mean" in s and s[f"{k}_mean"].notna().any() for s in runs.values())]
    targets = list(dict.fromkeys(t for s in runs.values() for t in s["target"]))
    fig, axes = plt.subplots(1, len(kinds), figsize=(6.2 * len(kinds), 3.7), squeeze=False)
    for ax, k in zip(axes[0], kinds):
        for ri, (rname, s) in enumerate(runs.items()):
            if f"{k}_mean" not in s:
                continue
            for tgt, g in s[s[f"{k}_mean"].notna()].groupby("target", sort=False):
                g = g.sort_values("layer")
                color = CAT[targets.index(tgt) % len(CAT)]
                x, mean = g["layer"].to_numpy(), g[f"{k}_mean"].to_numpy()
                ax.plot(x, mean, color=color, ls=LINESTYLES[ri % 4], marker="o" if ri == 0 else None,
                        label=f"{tgt} ({rname})" if rname else tgt, zorder=3 - 0.1 * ri,
                        markeredgecolor=SURFACE, markeredgewidth=0.8, lw=2.0 if ri == 0 else 1.5)
                if ri == 0:
                    sd = g[f"{k}_std"].fillna(0).to_numpy()
                    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.16, lw=0, zorder=2)
        _mark_pez(ax, pez)
        ax.set_xlabel("layer (block index)")
        _score_axis(ax, k)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5)) if len(runs) > 1 else ax.legend(loc="best")
    axes[0][0].set_title(title)
    return _save(fig, path)


def _score_axis(ax, k):
    if k == "r2":
        ax.set_ylabel("cross-validated R²")
        ax.set_ylim(max(min(-0.05, ax.get_ylim()[0]), -0.5), 1.03)
    else:
        ax.set_ylabel("cross-validated accuracy")
        ax.axhline(0.5, color=AXIS, lw=1.0)
        ax.annotate("chance", (1, 0.5), xycoords=("axes fraction", "data"), xytext=(-3, 3),
                    textcoords="offset points", ha="right", fontsize=8, color=MUTED)
        ax.set_ylim(0.4, 1.02)


def condition_curves(summaries: dict, path, title, pez=None, shade=None):
    """One panel per probe target; one line per condition (first = reference, drawn
    in ink). shade = layers to mark (e.g. the ablated ones)."""
    first = next(iter(summaries.values()))
    targets = list(dict.fromkeys(first["target"]))
    fig, axes = plt.subplots(1, len(targets), figsize=(4.6 * len(targets), 3.6), squeeze=False)
    for ax, tgt in zip(axes[0], targets):
        k = "accuracy" if first[first.target == tgt]["kind"].iloc[0] == "binary" else "r2"
        for ci, (name, s) in enumerate(summaries.items()):
            g = s[s.target == tgt].sort_values("layer")
            ax.plot(g["layer"], g[f"{k}_mean"], color=INK if ci == 0 else CAT[(ci - 1) % len(CAT)],
                    marker="o", label=name, markeredgecolor=SURFACE, markeredgewidth=0.8,
                    lw=2.0 if ci == 0 else 1.6, zorder=3 if ci == 0 else 2)
        for l in shade or ():
            ax.axvspan(l - 0.5, l + 0.5, color=NEUTRAL, alpha=0.25, lw=0, zorder=0)
        ax.set_xlabel("layer (block index)")
        _score_axis(ax, k)
        ax.set_title(tgt, fontsize=10)
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                       title="shaded = ablated layers" if shade else None, title_fontsize=8)
    fig.suptitle(title, x=0.02, y=1.03, ha="left", fontweight="bold", fontsize=11)
    return _save(fig, path)


def probe_sequence(metrics_by_name: dict, thresholds: dict, path, title="Orthogonal probe sequence"):
    """Validation score of each successive probe after projecting out the earlier ones."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (name, m) in enumerate(metrics_by_name.items()):
        ax.plot(np.arange(len(m)), m, color=CAT[i], marker="o", label=name,
                markeredgecolor=SURFACE, markeredgewidth=0.8)
        if name in thresholds:
            ax.axhline(thresholds[name], color=CAT[i], lw=1.0, ls=(0, (3, 3)))
    ax.set_xlabel("probe index (earlier probes projected out)")
    ax.set_ylabel("validation score (R² or accuracy)")
    ax.set_title(title)
    ax.legend(title="dashed = stop threshold", title_fontsize=8)
    return _save(fig, path)


def subspace_layers(df, path, pez=None):
    """Per layer: subspace dimensions, and overlap vs the random-subspace baseline."""
    pairs = list(df["pair"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.6))
    dims = df.melt(id_vars=["layer"], value_vars=[c for c in df.columns if c.startswith("dim[")])
    for i, (name, g) in enumerate(dims.drop_duplicates(["layer", "variable"]).groupby("variable", sort=False)):
        axes[0].plot(g["layer"], g["value"], color=CAT[i], marker="o", label=name[4:-1],
                     markeredgecolor=SURFACE, markeredgewidth=0.8)
    axes[0].set_ylabel("subspace dimension (orthogonal probes)")
    axes[0].set_title("How many dimensions carry each variable")
    for i, p in enumerate(pairs):
        g = df[df.pair == p].sort_values("layer")
        axes[1].plot(g["layer"], g["mean_angle_deg"], color=CAT[i], marker="o", label=p,
                     markeredgecolor=SURFACE, markeredgewidth=0.8)
        axes[1].plot(g["layer"], g["mean_angle_random_deg"], color=CAT[i], lw=1.0, ls=(0, (3, 3)))
        axes[2].plot(g["layer"], g["overlap"], color=CAT[i], marker="o", label=p,
                     markeredgecolor=SURFACE, markeredgewidth=0.8)
        axes[2].plot(g["layer"], g["overlap_random"], color=CAT[i], lw=1.0, ls=(0, (3, 3)))
    axes[1].set_ylabel("mean principal angle (deg)")
    axes[1].set_title("Angle between subspaces (dashed = random)")
    axes[2].set_ylabel("projection overlap")
    axes[2].set_title("Overlap (dashed = random baseline)")
    for ax in axes:
        ax.set_xlabel("layer")
        _mark_pez(ax, pez, top=False)
        ax.legend()
    return _save(fig, path)


def steering_curve(res, path, title="Steering direction through its subspace"):
    c = res["curve"]
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    x = [r["n_probes"] for r in c]
    y = [r["mae_to_target_deg"] for r in c]
    ax.plot(x, y, color=CAT[0], marker="o", markeredgecolor=SURFACE, markeredgewidth=0.8)
    b = res["baseline_mae_to_target_deg"]
    ax.axhline(b, color=MUTED, lw=1.0, ls=(0, (3, 3)))
    ax.annotate(f"no steering: {b:.1f}°", (1, b), xycoords=("axes fraction", "data"),
                xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=8.5, color=INK2)
    for xi, yi in ((x[0], y[0]), (x[-1], y[-1])):
        ax.annotate(f"{yi:.1f}°", (xi, yi), xytext=(5, 6), textcoords="offset points", fontsize=8.5,
                    color=INK2)
    for i, (name, r) in enumerate(res.get("controls", {}).items()):
        ax.axhline(r["mae_to_target_deg"], color=CAT[1 + i], lw=1.4,
                   label=f"{name} ({r['dims']} dims): {r['mae_to_target_deg']:.1f}°")
    ax.set_xlabel("orthogonal probes steered jointly (2 dimensions each)")
    ax.set_ylabel("decoded error to target (deg)")
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    if res.get("controls"):
        ax.legend(loc="best", title="control: difference of means", title_fontsize=8)
    return _save(fig, path)


def tuning_overview(tun, curves, path, layer, kind="fc1", thr=0.1):
    """Preferred-direction histogram (polar) + tuning curves of the best-tuned units."""
    angs, C = curves
    fig = plt.figure(figsize=(11.5, 3.9))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    sel = tun[tun.delta_r2 > thr]
    bins = np.linspace(0, 2 * np.pi, 25)
    h, _ = np.histogram(np.deg2rad(sel.pref_deg), bins)
    ax.bar(bins[:-1] + np.pi / 24, h, width=2 * np.pi / 24 * 0.9, color=CAT[0], edgecolor=SURFACE, lw=0.6)
    ax.set_title(f"Preferred directions, layer {layer} {kind}\n{len(sel)} of {len(tun)} units with ΔR² > {thr}",
                 loc="left", fontsize=10)
    ax.set_yticklabels([])
    ax.grid(color=GRID)
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(C, aspect="auto", cmap=SEQ, interpolation="nearest")
    ax2.set_xticks(range(len(angs)), [f"{a:g}°" for a in angs])
    ax2.set_xlabel("motion direction")
    ax2.set_ylabel("units, sorted by preferred direction")
    ax2.set_title("Tuning curves (mean z-scored response)", fontsize=10)
    ax2.grid(False)
    fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
    return _save(fig, path)


def tiling_heatmap(M, path, title, xlabel="preferred direction (deg)", lo=0, hi=360, pez=None):
    """[layers, bins] best ΔR² among units preferring each bin. Gaps = untiled."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.imshow(M, aspect="auto", cmap=SEQ, origin="lower", vmin=0, vmax=max(np.nanmax(M), 1e-6),
                   extent=(lo, hi, -0.5, M.shape[0] - 0.5), interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("layer")
    ax.set_title(title)
    ax.grid(False)
    if pez is not None:
        ax.axhline(pez, color=INK, lw=0.8, ls=(0, (3, 3)))
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="max ΔR² in bin")
    return _save(fig, path)


def fraction_tuned(layers, frac_by_name: dict, path, pez=None, thr=0.1):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for i, (name, f) in enumerate(frac_by_name.items()):
        ax.plot(layers, f, color=CAT[i], marker="o", label=name, markeredgecolor=SURFACE,
                markeredgewidth=0.8)
    _mark_pez(ax, pez)
    ax.set_xlabel("layer")
    ax.set_ylabel(f"fraction of units with ΔR² > {thr}")
    ax.set_title("Tuned MLP units by layer")
    ax.set_ylim(bottom=0)
    ax.legend()
    return _save(fig, path)


def ablation_bars(table, path, title="Attention ablation at the emergence layers"):
    """table: DataFrame indexed by condition, one column per metric. One panel per
    metric so each has its own honest scale (all start at zero)."""
    cols = list(table.columns)
    ncol = min(3, len(cols))
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.3 * nrow), squeeze=False)
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    for ax, c in zip(axes.ravel(), cols):
        colors = [NEUTRAL] + [CAT[0]] * (len(table) - 1)
        ax.bar(range(len(table)), table[c], color=colors, width=0.62)
        for i, v in enumerate(table[c]):
            ax.annotate(f"{v:.2f}", (i, max(v, 0)), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK2)
        ax.set_xticks(range(len(table)), table.index, rotation=30, ha="right")
        ax.set_ylim(min(0, table[c].min() * 1.15), max(1.0 if table[c].max() <= 1 else table[c].max(), 0.05) * 1.08)
        ax.axhline(0, color=AXIS, lw=1.0)
        ax.set_title(c, fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def attention_distance(sp, tp, path, pez=None):
    """Mean attention distance of every head, by layer (dots) and the layer mean (line)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.7))
    for ax, M, lab in ((axes[0], sp, "spatial distance (patches)"),
                       (axes[1], tp, "temporal distance (tubelets)")):
        L, H = M.shape
        ax.scatter(np.repeat(np.arange(L), H), M.ravel(), s=14, color=CAT[0], alpha=0.45, lw=0,
                   label="one head")
        ax.plot(np.arange(L), M.mean(1), color=INK, lw=1.6, label="layer mean")
        _mark_pez(ax, pez)
        ax.set_xlabel("layer")
        ax.set_ylabel(f"attention-weighted {lab}")
        ax.legend()
    axes[0].set_title("How far each head looks")
    return _save(fig, path)


def patch_map(M, path, title):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(M, cmap=SEQ, vmin=min(0, np.nanmin(M)), vmax=max(np.nanmax(M), 1e-6))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("patch column")
    ax.set_ylabel("patch row")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return _save(fig, path)


# ----------------------------------------------------------------------- manifolds


def manifold_scatter(scores, values, path, title, circular=False, var_ratio=None, name="value",
                     second=None):
    """Condition means in PC space. Consecutive levels are joined so the path
    (ring, arc, line) is visible; each point is labelled with its level.
    second: optional array; points sharing a level of `second` are joined instead."""
    values = np.asarray(values, dtype=float)
    cmap = CYCLIC if circular else SEQ
    lo, hi = (0.0, 360.0) if circular else (values.min(), values.max())
    norm = lambda v: (v - lo) / max(hi - lo, 1e-12) if circular else 0.25 + 0.75 * (v - lo) / max(hi - lo, 1e-12)
    npc = scores.shape[1]
    panels = [(0, 1)] + ([(1, 2)] if npc >= 3 else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.5), squeeze=False)
    groups = [np.arange(len(values))] if second is None else \
        [np.where(np.asarray(second) == s)[0] for s in np.unique(second)]
    for ax, (a, b) in zip(axes[0], panels):
        for gi in groups:
            o = gi[np.argsort(values[gi])]
            loop = np.r_[o, o[0]] if circular and len(o) > 2 else o
            ax.plot(scores[loop, a], scores[loop, b], color=NEUTRAL, lw=1.0, zorder=1)
        ax.scatter(scores[:, a], scores[:, b], c=[cmap(norm(v)) for v in values], s=58, zorder=3,
                   edgecolors=SURFACE, linewidths=1.2)
        if len(values) <= 40:
            for i, v in enumerate(values):
                ax.annotate(f"{v:g}", (scores[i, a], scores[i, b]), xytext=(5, 5),
                            textcoords="offset points", fontsize=7.5, color=INK2)
        vr = (lambda i: f" ({100 * var_ratio[i]:.0f}%)") if var_ratio is not None else (lambda i: "")
        ax.set_xlabel(f"PC{a + 1}{vr(a)}")
        ax.set_ylabel(f"PC{b + 1}{vr(b)}")
        ax.set_aspect("equal", adjustable="datalim")
    axes[0][0].set_title(title)
    fig.text(0.01, -0.01, f"points = condition means, labelled and coloured by {name}",
             fontsize=8, color=MUTED)
    return _save(fig, path)


def stacked_by_layer(layers, parts: dict, path, title, ylabel, pez=None):
    """Stacked bars per layer. Keys keep a fixed colour; 'residual' is neutral."""
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bottom = np.zeros(len(layers))
    ci = 0
    for name, v in parts.items():
        v = np.asarray(v)
        color = NEUTRAL if name == "residual" else CAT[ci % len(CAT)]
        ci += name != "residual"
        ax.bar(layers, v, bottom=bottom, color=color, width=0.8, label=name, edgecolor=SURFACE, lw=0.8)
        bottom += v
    _mark_pez(ax, pez, top=False)
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    return _save(fig, path)


def heat_by_layer(M, path, title, xlabels, xlabel, cbar):
    fig, ax = plt.subplots(figsize=(1.2 + 0.7 * M.shape[1], 4.4))
    im = ax.imshow(M, aspect="auto", cmap=SEQ, origin="lower", vmin=0, vmax=max(np.nanmax(M), 1e-6),
                   interpolation="nearest")
    ax.set_xticks(range(M.shape[1]), xlabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("layer")
    ax.set_title(title)
    ax.grid(False)
    if M.shape[1] <= 8:
        for (i, j), v in np.ndenumerate(M):
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if v > 0.55 * np.nanmax(M) else INK2)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, label=cbar)
    return _save(fig, path)


def lines_by_layer(layers, series: dict, path, title, ylabel, pez=None):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for i, (name, v) in enumerate(series.items()):
        ax.plot(layers, v, color=CAT[i], marker="o", label=name, markeredgecolor=SURFACE,
                markeredgewidth=0.8)
    _mark_pez(ax, pez)
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend()
    return _save(fig, path)


def rdm_plot(D, ticks, path, title):
    fig, ax = plt.subplots(figsize=(4.9, 4.2))
    im = ax.imshow(D, cmap=SEQ, interpolation="nearest")
    step = max(1, len(ticks) // 12)
    ax.set_xticks(range(0, len(ticks), step), [f"{t:g}" if isinstance(t, (int, float, np.number)) else str(t)
                                               for t in ticks[::step]], rotation=90)
    ax.set_yticks(range(0, len(ticks), step), [f"{t:g}" if isinstance(t, (int, float, np.number)) else str(t)
                                               for t in ticks[::step]])
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="distance between condition means")
    return _save(fig, path)


def trajectories(scores, groups, path, title, labels=None, circular=True):
    """Space-averaged temporal tokens projected to PCs: one line per condition,
    marker size grows with time."""
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    C, Tt, _ = scores.shape
    cmap = CYCLIC if circular else SEQ
    for c in range(C):
        g = groups[c] if circular else 0.25 + 0.75 * groups[c]
        col = cmap(g) if groups is not None else CAT[c % len(CAT)]
        ax.plot(scores[c, :, 0], scores[c, :, 1], color=col, lw=1.4, zorder=2)
        ax.scatter(scores[c, :, 0], scores[c, :, 1], s=np.linspace(8, 46, Tt), color=col, zorder=3,
                   edgecolors=SURFACE, linewidths=0.6)
        if labels is not None and C <= 16:
            ax.annotate(labels[c], scores[c, -1, :2], xytext=(4, 4), textcoords="offset points",
                        fontsize=7.5, color=INK2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    fig.text(0.01, -0.01, "one line per condition; marker grows with time in the clip",
             fontsize=8, color=MUTED)
    return _save(fig, path)
