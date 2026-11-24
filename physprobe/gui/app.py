"""physprobe explorer. Started with `python -m physprobe gui`, which runs
`streamlit run` on this file with a results folder and a datasets folder."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from physprobe.gui import data as D
from physprobe.gui import figures as F


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--hosted", action="store_true",
                    help="read-only mode for public hosting: no dataset creation or local probe runs")
    a, _ = ap.parse_known_args(sys.argv[1:])
    return Path(a.runs).resolve(), Path(a.data).resolve(), a.hosted or os.environ.get("PHYSPROBE_HOSTED") == "1"


RUNS, DATA, HOSTED = _args()
st.set_page_config(page_title="physprobe", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {
  font-family: 'Atkinson Hyperlegible', system-ui, -apple-system, 'Segoe UI', sans-serif; }
h1, h2, h3 { letter-spacing: -0.01em; color: #17212b; }
h1 { font-size: 1.9rem; font-weight: 700; margin-bottom: .2rem; }
h2 { font-size: 1.35rem; font-weight: 700; }
h3 { font-size: 1.08rem; font-weight: 700; }
.block-container { padding-top: 2.2rem; max-width: 1400px; }
[data-testid="stMetricValue"] { font-size: 1.35rem; font-variant-numeric: tabular-nums; }
[data-testid="stCaptionContainer"], .stCaption { color: #5b6875; }
.lede { color: #3c4955; font-size: 1.02rem; max-width: 70ch; margin-bottom: 1.1rem; }
.readout { font-variant-numeric: tabular-nums; border-left: 3px solid #0e6f78; padding: .15rem 0 .15rem .8rem;
  margin: .2rem 0 .9rem 0; }
.readout b { color: #17212b; font-size: 1.3rem; }
.readout span { color: #5b6875; font-size: .9rem; display: block; }
code { font-size: .86em; }
</style>""", unsafe_allow_html=True)


def lede(text):
    st.markdown(f"<p class='lede'>{text}</p>", unsafe_allow_html=True)


def readout(value, label):
    st.markdown(f"<div class='readout'><b>{value}</b><span>{label}</span></div>", unsafe_allow_html=True)


def sources():
    """Results sources: the configured folder plus any bundles uploaded in this session."""
    base = [("", str(RUNS), str(DATA) if DATA.exists() else None)]
    return tuple(base + [(u["label"], u["path"], None) for u in st.session_state.get("uploads", [])])


@st.cache_data(ttl=30, show_spinner=False)
def _found(srcs):
    return D.discover_many(list(srcs)).__dict__


def found():
    return _found(sources())


@st.cache_resource(show_spinner=False)
def cache_of(path, kind="live"):
    return D.BundleSet(path) if kind == "bundle" else D.load_cache(path)


@st.cache_data(show_spinner="Computing the shape at every layer...", max_entries=16)
def trajectory(path, kind, by, where, videos):
    if kind == "bundle":
        tr = cache_of(path, kind).trajectory(by)
        if not videos:
            tr["videos"] = None
        return tr
    return D.manifold_trajectory(cache_of(path), by, where or None, k=3, with_videos=videos)


@st.cache_data(show_spinner="Measuring the shape at every layer...", max_entries=16)
def metrics(path, kind, by, where, circular):
    if kind == "bundle":
        return cache_of(path, kind).metrics(by)
    return D.manifold_metrics(cache_of(path), by, where or None, circular)


@st.cache_data(show_spinner=False)
def summary(path):
    return D.read_probe_summary(path)


def parse_specs(text):
    """'a:circular, vx,vy:vector, b:binary' -> ['a:circular', 'vx,vy:vector', 'b:binary']"""
    import re
    return [f"{c.strip(' ,')}:{k}" for c, k in re.findall(r"([^:]+?):(circular|scalar|binary|vector)", text)
            if c.strip(" ,")]


def cmd(*parts):
    return " ".join(shlex.quote(str(p)) for p in parts)


# ------------------------------------------------------------------- overview


REPO = os.environ.get("PHYSPROBE_REPO", "https://github.com/nishitaKatepallewar/physprobe")

CONTEXT = f"""
Probing and geometry results for frozen video encoders, produced with **physprobe**, an independent
reimplementation of the analyses in Joseph et al. (2026), *Interpreting Physics in Video World Models*
([arXiv:2602.07050](https://arxiv.org/abs/2602.07050)). The code and stimuli are not those of the original
authors; settings the paper leaves unspecified are documented in the [repository README]({REPO}#readme).
Results and their comparison with the paper: [FINDINGS.md]({REPO}/blob/main/FINDINGS.md).
"""

DEFINITIONS = """
**Stimuli.** Synthetic top-down videos of a ball on an 8 × 8 m floor, 16 frames at 24 fps.
- `velocity`: constant velocity; 8 directions × 7 speeds × 7 start positions (the paper's design).
- `velocity_centered`: as `velocity`, but each trajectory passes through its position at mid-clip, so the
  ball's mean position carries no information about direction.
- `velocity_stepped`: the ball is static within each two-frame tubelet (one token in time) and moves
  between tubelets. Motion is therefore not visible inside any single token.
- `shuffled_pairs`: each video twice, intact and with its tubelets in random order (frames within a
  tubelet keep their order).
- `permanence`: a ball passes behind a panel and continues, vanishes, jumps sideways or changes size.
  A stand-in for IntPhys, not IntPhys itself.

**Activations.** The residual stream after each transformer block, mean-pooled over all tokens.
Layers are numbered from 0. "Untrained" runs use the same architecture with random weights.

**Probes.** Linear read-outs with grouped 5-fold cross-validation (groups: start position, or matched
scene). Circular variables are regressed as (sin θ, cos θ) and scored by R².

**Shapes.** For each value of a variable, the condition mean is the average activation over all videos
with that value. Reliability is the correlation between condition-mean patterns computed from two
disjoint halves of the videos; harmonic *k* is the share of across-value variance that varies as
cos(kθ), sin(kθ) for a circular variable; ring order is 1 when the values appear around the first two
principal axes in their true circular order.

**Ablations.** At the chosen layers, attention from each query to nearby keys is removed (logits masked
before the softmax, remaining attention renormalised). Conditions: spatial s = 7 (keys within 7
patches), temporal t = 3 (keys within 3 tubelets), combined s = 3, t = 1 (keys within 3 patches or 1
tubelet).
"""


def page_overview():
    st.title("physprobe")
    st.markdown(CONTEXT)
    with st.expander("Definitions: stimuli, probes, shapes, ablations"):
        st.markdown(DEFINITIONS)
    f = found()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Activation sets", sum(not (c["ablated"] and c["kind"] == "missing") for c in f["caches"]))
    c2.metric("Probe runs", len(f["probes"]))
    c3.metric("Ablation runs", len(f["ablations"]))
    c4.metric("Datasets", len(f["datasets"]))
    if not HOSTED:
        st.caption(f"Results folder: `{RUNS}`  \nData folder: `{DATA}`")
    upload_section()

    if not (f["caches"] or f["probes"]):
        st.info("No results in this folder. Results are created by making a dataset and extracting activations:\n\n"
                "```\npython -m physprobe make velocity --out data/velocity\n"
                "python -m physprobe extract --data data/velocity --out runs/vjepa2-l/velocity/acts\n```")
        return
    # conditions of an ablation run are listed on the ablation page, not here, unless they can be explored
    listed = [c for c in f["caches"] if not (c["ablated"] and c["kind"] == "missing")]
    if listed:
        st.subheader("Activation sets")
        st.caption("Each set: one stimulus dataset passed through one model; residual-stream activations at every layer.")
        df = pd.DataFrame(listed)
        df["extras"] = [", ".join(n for n, k in (("MLP units", "fc1"), ("over time", "time"), ("ablated", "ablated"))
                                  if r.get(k)) for r in df.to_dict("records")]
        df["shapes"] = df["kind"].map({"live": "computed on demand", "bundle": "precomputed",
                                       "missing": "not available"})
        st.dataframe(df[["name", "dataset", "model", "videos", "layers", "extras", "shapes"]], hide_index=True,
                     width="stretch")
        if (df["kind"] == "missing").any():
            st.caption("Shapes “not available”: the probe results exist but neither the activation arrays (.npy) "
                       "nor an exported bundle of their shapes is present. Probe and ablation pages are unaffected.")
    if f["probes"]:
        with st.expander(f"{len(f['probes'])} probe runs"):
            st.dataframe(pd.DataFrame(f["probes"])[["name"]], hide_index=True, width="stretch")
    if not HOSTED and st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()


def upload_section():
    with st.expander("Load a results bundle"):
        st.markdown("A bundle holds probe results, ablations, attention statistics and precomputed shapes for one "
                    "results folder, without the activation arrays. It is created where the activations are, with\n\n"
                    "```\npython -m physprobe export --runs runs --out bundle --zip\n```\n\n"
                    "Uploaded bundles are added to every page for the rest of this browser session and are not "
                    "stored on the server.")
        up = st.file_uploader("Bundle (.zip)", type=["zip"], key="bundle_upload")
        if up is not None and up.file_id not in {u["id"] for u in st.session_state.get("uploads", [])}:
            from physprobe.export import safe_extract
            try:
                root = safe_extract(up, tempfile.mkdtemp(prefix="physprobe_bundle_"))
            except Exception as e:  # noqa: BLE001
                st.error(f"Bundle not loaded: {e}")
            else:
                label = Path(up.name).stem
                st.session_state.setdefault("uploads", []).append({"id": up.file_id, "label": label,
                                                                   "path": str(root)})
                st.success(f"Loaded “{label}”. Its entries are prefixed with “{label}:”.")
                st.rerun()
        for u in st.session_state.get("uploads", []):
            st.caption(f"Loaded: {u['label']}")


# --------------------------------------------------------------------- shapes


def page_shapes():
    st.title("Shape explorer")
    lede("The condition means of one variable (for each value, the average activation over all videos with "
         "that value) projected onto their principal axes, layer by layer. A closed ring indicates a circular "
         "code; a line or open curve indicates a magnitude code.")
    f = found()
    ready = [c for c in f["caches"] if c["kind"] in ("live", "bundle")]
    if not ready:
        st.warning("No activation sets with shapes available. Shapes need the activation arrays (created by "
                   "`python -m physprobe extract`) or an exported bundle (`python -m physprobe export`).")
        return
    names = [c["name"] for c in ready]
    default = next((i for i, n in enumerate(names) if "stepped24" in n and n.endswith("acts")),
                   next((i for i, n in enumerate(names) if n.endswith("acts")), 0))
    top = st.columns([3, 2, 2])
    name = top[0].selectbox("Activation set", names, index=default)
    entry = ready[names.index(name)]
    kind = entry["kind"]
    c = cache_of(entry["path"], kind)
    cols = c.shape_labels() if kind == "bundle" else D.label_columns(c.labels)
    if not cols:
        st.warning("This set has no label column with between 2 and 60 distinct values.")
        return
    by = top[1].selectbox("Variable to trace", cols,
                          index=cols.index("direction_deg") if "direction_deg" in cols else 0)
    where = top[2].text_input("Video subset (optional)", placeholder="speed >= 4", disabled=kind == "bundle",
                              help="A pandas expression over the label columns, for example `speed == 4` or "
                                   "`possible == 1`. Not available for precomputed (bundled) shapes.")
    if where:
        try:
            n = int(c.labels.eval(where).sum())
        except Exception as e:  # noqa: BLE001 - show the parse error to the user
            st.error(f"Invalid subset expression: {e}")
            return
        if n < 4:
            st.error(f"The subset contains {n} videos; at least 4 are required.")
            return
    circular = D.is_circular(c.labels, by, c.targets)
    opts = st.columns([1.3, 2.2, 2.8, 2])
    dims = 3 if opts[0].segmented_control("View", ["3-D", "2-D"], default="3-D") != "2-D" else 2
    show_videos = opts[1].toggle("Show individual videos", value=False,
                                 help="Each faint point is one video, projected onto the same axes.")
    normalise = opts[2].toggle("Normalise scale per layer", value=True,
                               help="Activation norms grow with depth. Rescaling each layer to unit size makes "
                                    "shapes comparable across layers; off shows the original scale.")
    try:
        traj = trajectory(entry["path"], kind, by, where, show_videos)
    except Exception as e:  # noqa: BLE001
        st.error(f"Shape computation failed: {e}")
        return
    L = traj["scores"].shape[0]
    tab1, tab2, tab3 = st.tabs(["Layer by layer", "Play through layers", "All layers at once"])

    with tab1:
        layer = st.slider("Layer", 0, L - 1, value=min(c.pez, L - 1), key="shape_layer")
        left, right = st.columns([3, 1.15])
        with left:
            st.plotly_chart(F.manifold_view(traj, layer, circular, dims, normalise, show_videos, by),
                            width="stretch", key=f"mf_{layer}_{dims}")
        with right:
            m = metrics(entry["path"], kind, by, where, circular)
            row = m[m.layer == layer].iloc[0]
            var = traj["var"][layer]
            readout(f"{row.reliability:.2f}", "split-half reliability of the condition means "
                                              f"(below {D.MIN_RELIABILITY}: no reproducible structure)")
            readout(f"{int(row.dimensions)}" if pd.notna(row.dimensions) else "n/a",
                    "dimensions holding 90% of the condition-mean variance (noise-corrected)")
            readout(f"{100 * var[:2].sum():.0f}%", "of the condition-mean variance in the first two principal axes")
            if circular and "harmonic 1" in m:
                ro = row["ring order"]
                readout(f"{ro:.2f}" if pd.notna(ro) else "n/a", "ring order (1: values in their circular order)")
                h = [row.get(f"harmonic {j}", np.nan) for j in (1, 2, 4)]
                readout(" / ".join(f"{x:.2f}" for x in h) if pd.notna(h[0]) else "n/a",
                        "harmonics 1 / 2 / 4: share of variance varying once per cycle (direction), twice "
                        "(axis: opposite values alike), four times (four-fold pattern)")
        g1, g2 = st.columns(2)
        g1.plotly_chart(F.metric_lines(m, ["reliability"] + (["ring order"] if circular else []), layer,
                                       "Reliability and ring order, by layer", "", [-0.3, 1.05],
                                       c.pez, height=270), width="stretch")
        if circular:
            g2.plotly_chart(F.metric_lines(m, ["harmonic 1", "harmonic 2", "harmonic 3", "harmonic 4"], layer,
                                           "Harmonic content, by layer", "share of the shape", [0, 1.02], c.pez,
                                           height=270),
                            width="stretch")
        else:
            g2.plotly_chart(F.metric_lines(m, ["dimensions"], layer, "Dimensionality, by layer",
                                           "dimensions", None, c.pez, height=270), width="stretch")
        st.caption("Solid vertical line: selected layer. Dotted line: one-third depth. Shape statistics are omitted "
                   f"where reliability is below {D.MIN_RELIABILITY}.")

    with tab2:
        st.caption("All layers as animation frames. Each layer is rescaled to unit size and rotated (orthogonal "
                   "Procrustes) to align with the preceding layer; distances within a layer are unchanged.")
        st.plotly_chart(F.manifold_animation(traj, circular, dims, by), width="stretch")

    with tab3:
        st.caption("One panel per layer: first two principal axes, unit scale. Outlined: the layer selected "
                   "under “Layer by layer”.")
        st.plotly_chart(F.filmstrip(traj, circular, current=st.session_state.get("shape_layer")),
                        width="stretch")

    if kind == "live":
        with st.expander("Equivalent command"):
            st.code(cmd("python", "-m", "physprobe", "manifold", "--acts", entry["path"], "--out",
                        str(Path(entry["path"]).parent / f"manifold_{by}"), "--by", by,
                        *(["--where", where] if where else [])), language="bash")


# --------------------------------------------------------------------- probes


def _pretty(n):
    """Run label for legends: 'vjepa2-l/velocity/probe_random' -> 'vjepa2-l/velocity (random weights)'."""
    for suffix, tag in (("/probe_random", " (random weights)"), ("/probe_vs_random", " (ridge)"),
                        ("/probe_gui", " (explorer run)"), ("/probe", "")):
        if n.endswith(suffix):
            return n[: -len(suffix)] + tag
    return n


def page_probes():
    st.title("Probes")
    lede("Cross-validated linear decoding of each variable from the mean-pooled residual stream after each "
         "block. Several runs can be overlaid, for example pretrained against randomly initialised weights, or "
         "two stimulus sets.")
    f = found()
    if not f["probes"]:
        st.info("No probe results found. They are created by `python -m physprobe probe --acts ... --out ...` "
                "or with the section below.")
    else:
        names = [p["name"] for p in f["probes"] if "/base" not in p["name"] and "ablate" not in p["name"]]
        names = names or [p["name"] for p in f["probes"]]
        default = [n for n in names if n.endswith("velocity/probe")][:1] or names[:1]
        rand = [n for n in names if n.endswith("velocity/probe_random")]
        chosen = st.multiselect("Probe runs", names, default=default + rand[:1], format_func=_pretty)
        runs = {_pretty(n): summary(f["probes"][[p["name"] for p in f["probes"]].index(n)]["path"])
                for n in chosen}
        if runs:
            targets = list(dict.fromkeys(t for s in runs.values() for t in s["target"]))
            pick = st.pills("Variables", targets, selection_mode="multi", default=targets)
            layers = int(max(s["layer"].max() for s in runs.values())) + 1
            fig = F.probe_curves(runs, pick or [], layers // 3)
            if fig is not None:
                st.plotly_chart(fig, width="stretch")
                st.caption(f"Dotted line: one-third depth (layer {layers // 3}). Band: ±1 SD across "
                           "cross-validation folds (first run). Colour: variable. Line style: run.")
            with st.expander("Table"):
                for rname, s in runs.items():
                    col = [c for c in ("r2_mean", "accuracy_mean") if c in s]
                    st.markdown(f"**{rname}**")
                    st.dataframe(s.pivot_table(index="layer", columns="target", values=col[0]).round(3).T,
                                 width="stretch")

    st.subheader("New probe run")
    ready = [c for c in f["caches"] if c["kind"] == "live"]
    if not ready:
        st.caption("Requires an activation set with activation arrays on disk." +
                   (" Not available in the hosted explorer." if HOSTED else ""))
        return
    name = st.selectbox("Activation set", [c["name"] for c in ready], key="probe_now")
    c = cache_of(ready[[x["name"] for x in ready].index(name)]["path"])
    tspec = st.text_input("Variables (column:kind, comma-separated)",
                          value=", ".join(f"{k}:{v}" for k, v in c.targets.items()),
                          help="kind is circular, scalar, binary or vector (vector: `vx,vy:vector`).")
    if st.button("Run probes", type="primary"):
        specs = parse_specs(tspec)
        with st.spinner(f"Ridge probes at all {c.n_layers} layers..."):
            try:
                s = D.quick_probe(c, specs)
            except Exception as e:  # noqa: BLE001
                st.error(f"Probe run failed: {e}")
                return
        st.plotly_chart(F.probe_curves({name: s}, list(dict.fromkeys(s["target"])), c.pez),
                        width="stretch")
        out = Path(ready[[x["name"] for x in ready].index(name)]["path"]).parent / "probe_gui"
        out.mkdir(exist_ok=True)
        s.to_csv(out / "probe_summary.csv", index=False)
        st.success(f"Saved to {out}; listed under “Probe runs”.")
        st.cache_data.clear()


# ------------------------------------------------------------------- ablation


def page_ablations():
    st.title("Attention ablations")
    lede("Attention to nearby tokens is removed at the shaded layers and every layer is probed again. "
         "A drop after the shaded layers indicates that local attention there contributes to computing the variable.")
    f = found()
    if not f["ablations"]:
        st.info("No ablation runs found. Example command:\n\n```\npython -m physprobe ablate --data data/velocity_stepped "
                "--out runs/vjepa2-l/ablate_stepped_L4-6 --layers 4,5,6 --conditions combined "
                "--base-acts runs/vjepa2-l/velocity_stepped/acts --solver ridge\n```")
        return
    names = [a["name"] for a in f["ablations"]]
    default = next((i for i, n in enumerate(names) if "stepped_L4-6" in n), 0)
    name = st.selectbox("Ablation run", names, index=default)
    res = D.read_ablation(f["ablations"][names.index(name)]["path"])
    targets = list(dict.fromkeys(t for s in res["curves"].values() for t in s["target"]))
    if res["layers"]:
        st.markdown(f"Local attention removed at layers **{', '.join(map(str, res['layers']))}** (shaded). "
                    "Black: unablated model.")
    st.plotly_chart(F.ablation_curves(res["curves"], targets, res["layers"]), width="stretch")
    st.markdown("**Scores at the read-out layers**")
    st.dataframe(res["table"].round(3), width="stretch")
    st.caption("Conditions: spatial s=7 removes keys within 7 patches; temporal t=3 removes keys within 3 "
               "tubelets; combined s=3,t=1 removes keys within 3 patches or 1 tubelet (within both, for runs "
               "made with --combine and).")


def page_attention():
    st.title("Attention distance")
    lede("Attention-weighted mean distance between query and key tokens for every head: spatial distance in "
         "patches, temporal distance in tubelets. Small values identify local heads.")
    f = found()
    if not f["attention"]:
        st.info("No attention statistics found. They are created by `python -m physprobe extract ... --attn-stats` "
                "(a few dozen videos suffice).")
        return
    names = [a["name"] for a in f["attention"]]
    name = st.selectbox("Run", names)
    a = D.read_attention(f["attention"][names.index(name)]["path"])
    if "spatial" in a:
        L = a["spatial"].shape[0]
        st.plotly_chart(F.attention_heads(a["spatial"], a["temporal"], L // 3), width="stretch")
    else:
        df = a["table"]
        st.plotly_chart(F.attention_table(df, len(df) // 3), width="stretch")
        st.caption("Per-head values require the attn_distance.npz file from the extraction folder.")


# ------------------------------------------------------------------- datasets


@st.cache_data(show_spinner=False, max_entries=32)
def preview_strip(path, n=4, frames=8):
    from physprobe.datasets import VideoDataset
    ds = VideoDataset(path)
    ids = np.linspace(0, len(ds) - 1, n).astype(int)
    rows = []
    for i in ids:
        v = ds.get(int(i))
        sel = v[np.round(np.linspace(0, len(v) - 1, frames)).astype(int)]
        rows.append(np.concatenate(list(sel), axis=1))
    return np.concatenate(rows, axis=0), ds.labels.iloc[ids]


def page_datasets():
    st.title("Datasets")
    lede("Stimulus sets behind the results shown here, with example frames and labels." if HOSTED else
         "Stimulus sets available for extraction: previews of existing sets, generation of synthetic sets "
         "with custom factor levels, and import of external videos with labels. Each step lists the "
         "extraction command.")
    f = found()
    tabs = st.tabs(["Browse"] + ([] if HOSTED else ["New synthetic set", "Import videos"]))
    tab1 = tabs[0]

    with tab1:
        if not f["datasets"]:
            st.info(f"No datasets in `{DATA}`.")
        else:
            df = pd.DataFrame(f["datasets"])
            st.dataframe(df[["name", "type", "family", "videos"]], hide_index=True, width="stretch")
            name = st.selectbox("Preview", df["name"].tolist())
            d = f["datasets"][df["name"].tolist().index(name)]
            try:
                if d.get("preview"):
                    st.image(d["preview"], caption="Example videos (rows), frames left to right", width="stretch")
                    lab = pd.read_csv(Path(d["path"]) / "manifest.csv", nrows=8)
                else:
                    img, lab = preview_strip(d["path"])
                    st.image(img, caption="Four videos (rows), eight frames each (left to right)", width="stretch")
                st.dataframe(lab.drop(columns=[c for c in ("seed", "scene_seed", "path") if c in lab]),
                             hide_index=True, width="stretch")
            except Exception as e:  # noqa: BLE001
                st.error(f"Preview failed: {e}")
            if not HOSTED:
                st.markdown("Extraction command:")
                st.code(cmd("python", "-m", "physprobe", "extract", "--data", d["path"], "--out",
                            str(RUNS / "vjepa2-l" / Path(d["path"]).name / "acts"), "--model", "vjepa2-l",
                            "--batch-size", "4"), language="bash")
    if HOSTED:
        st.caption("Creating and importing datasets requires a local installation: "
                   "`pip install -e \".[gui]\"`, then `python -m physprobe gui`.")
        return
    tab2, tab3 = tabs[1], tabs[2]

    with tab2:
        from physprobe import stimuli
        fams = list(stimuli.FAMILIES)
        fam = st.selectbox("Stimulus family", fams, index=fams.index("velocity_stepped")
                           if "velocity_stepped" in fams else 0)
        F_ = stimuli.FAMILIES[fam]
        st.caption(F_.doc)
        overrides = {}
        cols = st.columns(2)
        for i, (k, v) in enumerate(F_.grid.items()):
            txt = cols[i % 2].text_input(k, value=",".join(map(str, v)), key=f"grid_{fam}_{k}",
                                         help="Comma-separated levels, or start:stop:step")
            overrides[k] = stimuli.parse_levels(txt)
        starts = st.number_input("Random start positions per condition", 0, 50, F_.n_starts) if F_.n_starts else None
        out_name = st.text_input("Dataset name", value=f"{fam}_custom")
        n = int(np.prod([len(v) for v in overrides.values()])) * max(starts or 1, 1)
        st.caption(f"{n} videos")
        if st.button("Create dataset", type="primary"):
            from physprobe import datasets
            ds = datasets.write_synthetic(DATA / out_name, fam, overrides, starts)
            datasets.write_contact_sheet(ds, DATA / out_name / "preview.png")
            st.success(f"Created {len(ds)} videos in {DATA / out_name}.")
            st.image(str(DATA / out_name / "preview.png"), width="stretch")
            st.code(cmd("python", "-m", "physprobe", "extract", "--data", DATA / out_name, "--out",
                        RUNS / "vjepa2-l" / out_name / "acts", "--model", "vjepa2-l", "--batch-size", "4"),
                    language="bash")
            st.cache_data.clear()

    with tab3:
        st.markdown("Input: a CSV with one row per video, a `path` column (video file, folder of frames, or "
                    "`.npy` array) and one column per label to probe. Example:")
        st.code("path,angle,bounced,clip\nvideos/cube_001.mp4,30,1,a\nvideos/cube_002.mp4,210,0,a\n", language="text")
        csv = st.text_input("CSV file", placeholder="/path/to/videos/labels.csv")
        targets = st.text_input("Labels to probe (column:kind)", placeholder="angle:circular, bounced:binary",
                                help="kind is circular (angles in degrees), scalar, binary or vector.")
        group = st.text_input("Group column (optional)", placeholder="clip",
                              help="Videos with the same value are never split between training and test folds.")
        frames = st.number_input("Frames per clip", 2, 64, 16, step=2)
        out_name = st.text_input("Dataset name", value="imported_videos", key="own_name")
        if st.button("Import dataset", type="primary"):
            from physprobe import datasets
            try:
                df = pd.read_csv(csv)
                base = Path(csv).parent
                df["path"] = [str(p if Path(p).is_absolute() else (base / p).resolve()) for p in df["path"]]
                tg = dict(t.rsplit(":", 1) for t in parse_specs(targets))
                if not tg:
                    raise ValueError("at least one label is required, as column:kind (for example angle:circular)")
                missing = [c for t in tg for c in t.split(",") if c not in df.columns]
                if missing:
                    raise ValueError(f"columns not in the CSV: {missing}")
                ds = datasets.write_folder(DATA / out_name, df, tg, group or None, int(frames))
                ds.get(0)  # make sure the first video can be read
            except Exception as e:  # noqa: BLE001
                st.error(f"Import failed: {e}")
            else:
                st.success(f"Imported {len(ds)} videos as {DATA / out_name}.")
                st.code(cmd("python", "-m", "physprobe", "extract", "--data", DATA / out_name, "--out",
                            RUNS / "vjepa2-l" / out_name / "acts", "--model", "vjepa2-l", "--batch-size", "4"),
                        language="bash")
                st.cache_data.clear()


# ---------------------------------------------------------------- navigation

pages = [st.Page(page_overview, title="Overview", default=True),
         st.Page(page_shapes, title="Shape explorer", url_path="shapes"),
         st.Page(page_probes, title="Probes", url_path="probes"),
         st.Page(page_ablations, title="Attention ablations", url_path="ablations"),
         st.Page(page_attention, title="Attention distance", url_path="attention"),
         st.Page(page_datasets, title="Datasets", url_path="datasets")]
nav = st.navigation(pages)
st.sidebar.markdown(f"[Code on GitHub]({REPO}) · [Findings]({REPO}/blob/main/FINDINGS.md) · "
                    "[Paper](https://arxiv.org/abs/2602.07050)")
nav.run()
