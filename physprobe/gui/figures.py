"""Plotly figures for the GUI, in the same visual system as physprobe.plots: light
surface, recessive grid, colour follows the series (fixed order), one-hue scale for
magnitudes, a cyclic wheel only for circular variables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..plots import CAT, INK, INK2, MUTED, NEUTRAL

SURFACE = "#ffffff"
GRID = "#e3e8eb"
AXIS = "#c5cfd5"
FONT = "Atkinson Hyperlegible, system-ui, -apple-system, Segoe UI, sans-serif"
WHEEL = ["#2a78d6", "#6250d6", "#e87ba4", "#e34948", "#eb6834", "#eda100", "#1baf7a", "#2a78d6"]
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DASHES = ["solid", "dash", "dot", "dashdot"]


def _base(fig, height=420, legend=True):
    fig.update_layout(
        template="none", height=height, margin=dict(l=56, r=16, t=28, b=48),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK2),
        hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(family=FONT, color=INK)),
        showlegend=legend, legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="outside", tickcolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="outside", tickcolor=AXIS)
    return fig


def _lerp_hex(stops, t):
    t = float(np.clip(t, 0, 1)) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    f = t - i
    a = np.array([int(stops[i][j:j + 2], 16) for j in (1, 3, 5)])
    b = np.array([int(stops[i + 1][j:j + 2], 16) for j in (1, 3, 5)])
    c = np.round(a + (b - a) * f).astype(int)
    return "#%02x%02x%02x" % tuple(c)


def level_colors(levels, circular: bool):
    v = np.asarray(levels, dtype=float)
    if circular:
        return [_lerp_hex(WHEEL, (x % 360) / 360) for x in v]
    lo, hi = v.min(), v.max()
    return [_lerp_hex(SEQ, (x - lo) / max(hi - lo, 1e-12)) for x in v]


def _fmt(v):
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------------- probes


def probe_curves(runs: dict, targets: list, pez: int | None = None):
    """runs: {label: summary DataFrame}. Colour = target, line style = run.
    R^2 and accuracy targets go in separate panels (never a second y-axis)."""
    kinds = []
    for s in runs.values():
        for t in targets:
            g = s[s.target == t]
            if len(g):
                kinds.append("accuracy" if g["kind"].iloc[0] == "binary" else "r2")
    kinds = [k for k in ("r2", "accuracy") if k in kinds]
    if not kinds:
        return None
    fig = make_subplots(rows=1, cols=len(kinds), horizontal_spacing=0.09,
                        subplot_titles=["Cross-validated R²" if k == "r2" else "Cross-validated accuracy"
                                        for k in kinds])
    for ri, (rname, s) in enumerate(runs.items()):
        for ti, t in enumerate(targets):
            g = s[s.target == t].sort_values("layer")
            if not len(g):
                continue
            k = "accuracy" if g["kind"].iloc[0] == "binary" else "r2"
            col = kinds.index(k) + 1
            color = CAT[ti % len(CAT)]
            m, sd = g[f"{k}_mean"].to_numpy(), g[f"{k}_std"].fillna(0).to_numpy()
            x = g["layer"].to_numpy()
            name = f"{t} ({rname})" if len(runs) > 1 else t
            if ri == 0 and sd.any():
                fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[m + sd, (m - sd)[::-1]], fill="toself",
                                         fillcolor=color, opacity=0.15, line=dict(width=0), hoverinfo="skip",
                                         showlegend=False), row=1, col=col)
            fig.add_trace(go.Scatter(x=x, y=m, mode="lines+markers", name=name,
                                     line=dict(color=color, width=2.2 if ri == 0 else 1.8, dash=DASHES[ri % 4]),
                                     marker=dict(size=6, line=dict(color="white", width=1)),
                                     hovertemplate=f"{name}<br>layer %{{x}}: %{{y:.3f}}<extra></extra>"),
                          row=1, col=col)
    for i, k in enumerate(kinds, start=1):
        fig.update_xaxes(title_text="layer", dtick=2, row=1, col=i)
        if k == "accuracy":
            fig.add_hline(y=0.5, line=dict(color=AXIS, width=1), row=1, col=i)
            fig.update_yaxes(range=[0.4, 1.02], row=1, col=i)
        else:
            fig.update_yaxes(range=[-0.1, 1.02], row=1, col=i)
        if pez is not None:
            fig.add_vline(x=pez, line=dict(color=MUTED, width=1, dash="dot"), row=1, col=i)
    for a in fig.layout.annotations:
        a.font = dict(family=FONT, size=13, color=INK)
        a.x = a.x - 0.5 / len(kinds) + 0.005
        a.xanchor = "left"
    fig = _base(fig, 470)
    fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0, yanchor="top"), margin=dict(b=40))
    return fig


# -------------------------------------------------------------------- manifold


def _closed(idx, circular):
    return np.r_[idx, idx[:1]] if circular and len(idx) > 2 else idx


def _norm_layer(S):
    r = np.sqrt((S ** 2).sum(1).mean())
    return S / max(r, 1e-12)


def _pad_axes(X, k=2):
    """Append zero columns so X has at least k axes. A variable with two values has a
    single axis of variation; its points are drawn along that axis at zero height."""
    if X is None or X.shape[1] >= k:
        return X
    return np.hstack([X, np.zeros((X.shape[0], k - X.shape[1]), dtype=X.dtype)])


def manifold_frame(scores, levels, circular, dims=3, videos=None, video_levels=None, label=""):
    """Traces for one layer: optional faint videos, the path through condition means
    (closed for circular variables), and the labelled means."""
    scores, videos = _pad_axes(scores), _pad_axes(videos)
    order = np.argsort(levels.astype(float)) if np.issubdtype(levels.dtype, np.number) else np.arange(len(levels))
    path = _closed(order, circular)
    cols = level_colors(levels, circular) if np.issubdtype(levels.dtype, np.number) else \
        [CAT[i % len(CAT)] for i in range(len(levels))]
    # labels: all of them in 2-D; in 3-D only when few enough not to collide (hover shows the rest)
    text = [_fmt(v) for v in levels] if len(levels) <= (36 if dims == 2 else 12) else None
    tr = []
    if dims == 3 and scores.shape[1] >= 3:
        if videos is not None:
            vc = dict(zip(levels.tolist(), cols))
            tr.append(go.Scatter3d(x=videos[:, 0], y=videos[:, 1], z=videos[:, 2], mode="markers",
                                   marker=dict(size=2.2, color=[vc.get(v, NEUTRAL) for v in video_levels],
                                               opacity=0.35), hoverinfo="skip", name="videos"))
        tr.append(go.Scatter3d(x=scores[path, 0], y=scores[path, 1], z=scores[path, 2], mode="lines",
                               line=dict(color=NEUTRAL, width=4), hoverinfo="skip", name="path"))
        tr.append(go.Scatter3d(x=scores[:, 0], y=scores[:, 1], z=scores[:, 2],
                               mode="markers+text" if text else "markers", text=text, textposition="top center",
                               textfont=dict(size=11, color=INK2),
                               marker=dict(size=6, color=cols, line=dict(color="white", width=1.5)),
                               customdata=[_fmt(v) for v in levels],
                               hovertemplate=f"{label} %{{customdata}}<extra></extra>", name="condition means"))
    else:
        if videos is not None:
            vc = dict(zip(levels.tolist(), cols))
            tr.append(go.Scatter(x=videos[:, 0], y=videos[:, 1], mode="markers",
                                 marker=dict(size=4, color=[vc.get(v, NEUTRAL) for v in video_levels],
                                             opacity=0.3), hoverinfo="skip", name="videos"))
        tr.append(go.Scatter(x=scores[path, 0], y=scores[path, 1], mode="lines",
                             line=dict(color=NEUTRAL, width=1.6), hoverinfo="skip", name="path"))
        tr.append(go.Scatter(x=scores[:, 0], y=scores[:, 1], mode="markers+text" if text else "markers",
                             text=text, textposition="top right", textfont=dict(size=11, color=INK2),
                             marker=dict(size=11, color=cols, line=dict(color="white", width=1.5)),
                             customdata=[_fmt(v) for v in levels],
                             hovertemplate=f"{label} %{{customdata}}<extra></extra>", name="condition means"))
    return tr


def _scene_layout(fig, dims, lim, var=None):
    def t(i):
        if var is None:
            return f"PC{i + 1}"
        if i >= len(var):
            return ""  # padding axis: the variable has fewer axes of variation than shown
        return f"PC{i + 1} ({100 * var[i]:.0f}%)"
    if dims == 3:
        ax = dict(range=[-lim, lim], gridcolor=GRID, zerolinecolor=AXIS, backgroundcolor=SURFACE,
                  showspikes=False, tickfont=dict(size=10, color=MUTED))
        fig.update_layout(scene=dict(xaxis=dict(ax, title=t(0)), yaxis=dict(ax, title=t(1)),
                                     zaxis=dict(ax, title=t(2)), aspectmode="cube",
                                     camera=dict(eye=dict(x=1.45, y=1.45, z=0.9))))
    else:
        fig.update_xaxes(range=[-lim, lim], title_text=t(0))
        fig.update_yaxes(range=[-lim, lim], title_text=t(1), scaleanchor="x", scaleratio=1)


def manifold_view(traj, layer, circular, dims=3, normalise=True, show_videos=False, label=""):
    S = traj["scores"][layer]
    V = traj["videos"][layer] if show_videos and traj["videos"] is not None else None
    if normalise:
        r = np.sqrt((S ** 2).sum(1).mean())
        S = S / max(r, 1e-12)
        V = V / max(r, 1e-12) if V is not None else None
    lim = 1.25 * (np.abs(S).max() if V is None else np.percentile(np.abs(V), 99))
    dims = min(dims, S.shape[1])  # a variable with 3 values has only 2 axes of variation
    fig = go.Figure(manifold_frame(S, traj["levels"], circular, dims, V, traj["video_levels"], label))
    _scene_layout(fig, dims, lim, traj["var"][layer])
    return _base(fig, 560, legend=False)


def manifold_animation(traj, circular, dims=3, label=""):
    """All layers as animation frames, with a play button and a layer slider."""
    L = traj["scores"].shape[0]
    dims = min(dims, traj["scores"].shape[2])
    frames = []
    for l in range(L):
        S = _norm_layer(traj["scores"][l])
        frames.append(go.Frame(data=manifold_frame(S, traj["levels"], circular, dims, label=label), name=str(l)))
    fig = go.Figure(frames[0].data, frames=frames)
    _scene_layout(fig, dims, 1.25 * max(np.abs(_norm_layer(traj["scores"][l])).max() for l in range(L)))
    steps = [dict(method="animate", label=str(l), args=[[str(l)], dict(mode="immediate",
             frame=dict(duration=0, redraw=dims == 3), transition=dict(duration=0))]) for l in range(L)]
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.02, xanchor="left", yanchor="top",
                          showactive=False, bgcolor="white", bordercolor=AXIS,
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, dict(frame=dict(duration=450, redraw=dims == 3),
                                                         transition=dict(duration=200), fromcurrent=True)]),
                                   dict(label="Pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(active=0, steps=steps, x=0.14, y=-0.02, len=0.86, yanchor="top",
                      currentvalue=dict(prefix="layer ", font=dict(size=13, color=INK)),
                      bordercolor=AXIS, tickcolor=AXIS, font=dict(size=10, color=MUTED))])
    fig = _base(fig, 620, legend=False)
    fig.update_layout(margin=dict(b=90))
    return fig


def filmstrip(traj, circular, cols=6, current=None):
    """Every layer's 2-D shape side by side (each layer scaled to the same size)."""
    L = traj["scores"].shape[0]
    rows = int(np.ceil(L / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"layer {l}" for l in range(L)],
                        horizontal_spacing=0.02, vertical_spacing=0.06)
    for l in range(L):
        S = _norm_layer(traj["scores"][l])
        for t in manifold_frame(S, traj["levels"], circular, dims=2):
            t.update(text=None, mode="lines" if t.mode == "lines" else "markers")
            if t.marker is not None and t.mode == "markers":
                t.marker.size = 6
            fig.add_trace(t, row=l // cols + 1, col=l % cols + 1)
    fig.update_xaxes(range=[-2.1, 2.1], showticklabels=False, ticks="", showgrid=False, linecolor=GRID, mirror=True)
    fig.update_yaxes(range=[-2.1, 2.1], showticklabels=False, ticks="", showgrid=False, linecolor=GRID, mirror=True)
    for i, a in enumerate(fig.layout.annotations):
        a.font = dict(family=FONT, size=11, color=INK if i == current else MUTED)
    if current is not None:
        fig.update_xaxes(linecolor=INK, linewidth=2, row=current // cols + 1, col=current % cols + 1)
        fig.update_yaxes(linecolor=INK, linewidth=2, row=current // cols + 1, col=current % cols + 1)
    fig = _base(fig, 150 * rows + 40, legend=False)
    fig.update_layout(margin=dict(l=8, r=8, t=24, b=8))
    fig.update_xaxes(ticks="")
    fig.update_yaxes(ticks="")
    return fig


def metric_lines(df: pd.DataFrame, cols: list, current: int | None = None, title="", ytitle="", yrange=None,
                 pez=None, height=240):
    fig = go.Figure()
    for i, c in enumerate(cols):
        if c not in df:
            continue
        fig.add_trace(go.Scatter(x=df["layer"], y=df[c], mode="lines+markers", name=c,
                                 line=dict(color=CAT[i % len(CAT)], width=2),
                                 marker=dict(size=5, line=dict(color="white", width=1)),
                                 hovertemplate=f"{c}<br>layer %{{x}}: %{{y:.2f}}<extra></extra>"))
    if current is not None:
        fig.add_vline(x=current, line=dict(color=INK, width=1.2))
    if pez is not None:
        fig.add_vline(x=pez, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_xaxes(dtick=2)
    fig.update_yaxes(title_text=ytitle, range=yrange)
    fig = _base(fig, height, legend=len(cols) > 1)
    fig.update_layout(title=dict(text=title, font=dict(size=13, color=INK), x=0, xanchor="left", y=0.98),
                      margin=dict(t=34, b=30), legend=dict(orientation="h", y=-0.18, x=0, yanchor="top"))
    return fig


# ------------------------------------------------------------- ablation, attention


def ablation_curves(curves: dict, targets: list, shaded=None):
    fig = make_subplots(rows=1, cols=len(targets), subplot_titles=targets, horizontal_spacing=0.06)
    names = list(curves)
    for ti, t in enumerate(targets, start=1):
        for ci, n in enumerate(names):
            s = curves[n]
            g = s[s.target == t].sort_values("layer")
            if not len(g):
                continue
            k = "accuracy" if g["kind"].iloc[0] == "binary" else "r2"
            color = INK if n == "base" else CAT[(ci - 1) % len(CAT)]
            fig.add_trace(go.Scatter(x=g["layer"], y=g[f"{k}_mean"], mode="lines+markers", name=n,
                                     legendgroup=n, showlegend=ti == 1,
                                     line=dict(color=color, width=2.4 if n == "base" else 1.8),
                                     marker=dict(size=5, line=dict(color="white", width=1)),
                                     hovertemplate=f"{n}<br>layer %{{x}}: %{{y:.3f}}<extra></extra>"),
                          row=1, col=ti)
        for l in shaded or ():
            fig.add_vrect(x0=l - 0.5, x1=l + 0.5, fillcolor=NEUTRAL, opacity=0.25, line_width=0, row=1, col=ti)
        fig.update_xaxes(title_text="layer", dtick=4, row=1, col=ti)
    for a in fig.layout.annotations:
        a.font = dict(family=FONT, size=13, color=INK)
    fig = _base(fig, 380)
    fig.update_layout(legend=dict(orientation="h", y=-0.22, x=0))
    return fig


def attention_heads(spatial, temporal, pez=None):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=["Spatial distance (patches)", "Temporal distance (tubelets)"])
    for c, M in enumerate((spatial, temporal), start=1):
        L, H = M.shape
        fig.add_trace(go.Scatter(x=np.repeat(np.arange(L), H) + np.tile(np.linspace(-0.2, 0.2, H), L),
                                 y=M.ravel(), mode="markers", name="one head", showlegend=c == 1,
                                 marker=dict(size=5, color=CAT[0], opacity=0.45),
                                 customdata=np.tile(np.arange(H), L),
                                 hovertemplate="layer %{x:.0f}, head %{customdata}: %{y:.2f}<extra></extra>"),
                      row=1, col=c)
        fig.add_trace(go.Scatter(x=np.arange(L), y=M.mean(1), mode="lines", name="layer mean",
                                 showlegend=c == 1, line=dict(color=INK, width=2)), row=1, col=c)
        fig.update_xaxes(title_text="layer", dtick=2, row=1, col=c)
        if pez is not None:
            fig.add_vline(x=pez, line=dict(color=MUTED, width=1, dash="dot"), row=1, col=c)
    for a in fig.layout.annotations:
        a.font = dict(family=FONT, size=13, color=INK)
    fig = _base(fig, 400)
    fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0))
    return fig


def attention_table(df: pd.DataFrame, pez=None):
    fig = go.Figure()
    for i, (lo, mid, name) in enumerate((("spatial_min_head", "spatial_mean", "spatial"),
                                         ("temporal_min_head", "temporal_mean", "temporal"))):
        fig.add_trace(go.Scatter(x=df.layer, y=df[mid], mode="lines+markers", name=f"{name}: layer mean",
                                 line=dict(color=CAT[i], width=2)))
        fig.add_trace(go.Scatter(x=df.layer, y=df[lo], mode="lines", name=f"{name}: most local head",
                                 line=dict(color=CAT[i], width=1.4, dash="dot")))
    if pez is not None:
        fig.add_vline(x=pez, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_xaxes(title_text="layer", dtick=2)
    fig.update_yaxes(title_text="attention distance")
    fig = _base(fig, 380)
    fig.update_layout(legend=dict(orientation="h", y=-0.22, x=0))
    return fig
