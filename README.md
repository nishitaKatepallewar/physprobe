# physprobe

Probing, ablation and representational-geometry tools for frozen video encoders, built to
replicate and extend Joseph et al. (2026), *Interpreting Physics in Video World Models*
([arXiv:2602.07050](https://arxiv.org/abs/2602.07050)) on V-JEPA 2.

This is an independent reimplementation from the paper's text; the original code and stimuli
are not public. Results obtained with it on V-JEPA 2-L, including where they agree and disagree
with the paper, are in **[FINDINGS.md](FINDINGS.md)**.

Components:

- **Stimuli**: a deterministic renderer for controlled-physics videos (moving, accelerating,
  orbiting and occluded balls; colour, size and position sweeps; two-object scenes) plus
  loaders for IntPhys and for arbitrary labelled videos.
- **Model access**: V-JEPA 2 (L/H/G) and VideoMAE wrappers with hooks on the residual stream,
  MLP units and attention; attention ablations; residual-stream interventions. Results are
  reported only for V-JEPA 2-L.
- **Analyses**: layerwise linear and attentive probes, orthogonal probe sequences, subspace
  angles, rank-k erasure, steering, unit tuning, per-patch probes, attention distance, and
  manifold statistics (split-half reliability, noise-corrected dimensionality, harmonic
  content, variance partitioning).
- **Explorer**: a browser interface over the results, runnable locally or hosted.

## Install

```bash
pip install -e .              # analysis and model code
pip install -e ".[gui]"       # plus the explorer (streamlit, plotly)
python -m physprobe selftest  # 60 checks on planted structure; no downloads, about a minute
```

Python 3.10 or newer; developed with torch 2.15 and transformers 5.23 (V-JEPA 2 requires
transformers 4.53 or newer). Model weights download from Hugging Face on first use
(`facebook/vjepa2-vitl-fpc64-256` and siblings). A GPU is needed for extraction at practical
speed; all analyses run on CPU.

## Quick start

```bash
python -m physprobe make velocity --out data/velocity                      # 392 videos
python -m physprobe extract --data data/velocity --out runs/vjepa2-l/velocity/acts --model vjepa2-l
python -m physprobe probe --acts runs/vjepa2-l/velocity/acts --out runs/vjepa2-l/velocity/probe
python -m physprobe manifold --acts runs/vjepa2-l/velocity/acts --out runs/vjepa2-l/manifold --by direction_deg
python -m physprobe gui                                                    # explore runs/ and data/
```

`python -m physprobe demo --out runs/demo` writes a simulated activation set (planted
structure, not model output) for trying the analyses and the explorer without a GPU.
`notebooks/colab_quickstart.ipynb` runs the same steps on a Colab GPU.

## Full runs

```bash
bash scripts/run_replication.sh      # the paper's protocol
bash scripts/run_controls.sh         # the control experiments in FINDINGS.md
bash scripts/run_manifold_zoo.sh     # shapes of other variables (hue, size, position, orbits, two balls)
```

Environment variables: `MODEL` (`vjepa2-l`, `vjepa2-h`, `vjepa2-g`), `RES` (default 224),
`BS` (batch size), `DEVICE`, `OUT`, `DATA`, `INTPHYS` (path to the IntPhys
dev set), `QUICK=1` (ridge probes, six layers for subspaces). Finished extractions are reused
when a script is run again. Each script ends with `runs/<model>/report.html`.

### Paper sections and the commands that reproduce them

| Paper | Command | Main output | Reported in the paper (as read from the text) |
|---|---|---|---|
| Fig. 1, possible vs impossible | `probe` on IntPhys or `permanence` | `layer_curves.png` | chance early, ~85–95% at about one-third depth, peak mid-network, decline toward the output |
| Fig. 2, speed, acceleration, direction | `probe` on `velocity`, `acceleration` | `layer_curves.png` | magnitudes early; direction from the transition (layer ~8 of 24) |
| Sec. 6.1, shuffled videos | `make-shuffled --block 2`, `probe` | `layer_curves.png` | emergence similar to IntPhys |
| Fig. 3, attention distance | `extract --attn-stats`, `attn` | `attention_distance.png` | local heads at the transition |
| Table 2, attention ablation | `ablate` | `ablation_table.csv` | direction R² 0.97 base; 0.93 spatial s=7; 0.83 temporal t=3; 0.14 combined |
| App. C.11, orthogonal probes | `subspace` | `dimension_by_layer.csv` | direction 14–136 dimensions; IntPhys 1–15 |
| Table 3, subspace overlap | `subspace` | `subspace_by_layer.csv` | direction vs IntPhys ~69–75°; speed vs IntPhys ~80–83° |
| Fig. 4, tuned units | `tuning` | `direction_tiling_fc1.png` | direction-tuned MLP units tile 360°; organisation absent in early layers, emerging at the transition |
| Sec. 7.2, steering | `steer` | `steering_L8.png` | 82.9° without steering; 11.9° with 20 probes; one probe insufficient |
| App. C.5, per-patch probes | `extract --patch-layers`, `patch` | `patch_r2_L*.png` | direction decodable across patches after the transition |

Not implemented: the ImageNet, Something-Something-v2 and CLEVRER controls, VideoMAE-v2 G, and
the predictor-based violation-of-expectation analysis.

### Settings the paper does not specify

Each is a command-line flag or a few lines of code.

- **Rendering.** The paper uses Kubric/Blender with a perspective camera 10 m above the floor,
  rendered at 256 × 256. With zero friction and an overhead camera the motion is closed-form, so a
  flat-shaded orthographic renderer with the paper's geometry is used here (8 m floor, ball radius
  0.3 m, 24 fps, 16 frames, speeds 1–7 m/s, starts in [−2, 2]² m). Trajectories match the
  description; images do not (no perspective, simpler shading). `--view-m` sets the visible width.
- **Resolution.** 224 × 224 as stated in the paper (14 × 14 patches, 1568 tokens); the released
  checkpoints were trained at 256. The Hugging Face module derives rotary position ids from the
  checkpoint's 256 grid, which misplaces patches at other sizes; the wrapper resets the grid to
  the input size.
- **Preprocessing.** Frames are resized directly to the model resolution (bilinear, antialiased)
  and normalised with ImageNet statistics; non-square frames are centre-cropped first.
- **Layers.** Layer *i* is the residual stream after block *i* (0-based), mean-pooled over all
  tokens. One-third depth is layer `n_layers // 3`.
- **Probes.** The 20-configuration AdamW sweep is as described; epochs (100), batch size (64),
  feature standardisation and the validation rule (a grouped 20% of each training fold) are
  choices made here. `--solver ridge` is a closed-form alternative.
- **Cross-validation groups.** Start position for the synthetic sets, matched scene for IntPhys
  and `permanence` (`--group`).
- **IntPhys frames.** 16 frames spread uniformly over each clip (`--frames`, `--sampling`).
- **Attention ablation.** Attention to keys within patch distance *s* and/or tubelet distance
  *t* is masked before the softmax at layers L/3−1 and L/3 (`--layers`), all heads. "Combined"
  removes keys near in space *or* time (`--combine or`, the default); with *and* it would remove a
  subset of the spatial condition, which cannot reproduce the paper's ordering of effects.
  Read-out layers default to L/3+2, L/2 and L−1 (`--read-layers`).
- **Attention distance.** Attention-weighted mean query–key distance per head, averaged over
  queries.
- **Orthogonal probes.** Adam, learning rate 1e-3, weight decay 1e-4, 100/50 epochs and the stop
  thresholds as stated; batch size 32, zero initialisation, isotropic scaling and standardised
  scalar targets chosen here.
- **Steering.** 70/30 split with the evaluation probe trained on the held-out 30%, as in App. C.12.
  The paper's two steering figures (<0.5° in the text, 11.9° in the appendix) are not reconciled
  there. `--strict` scores with an evaluation probe that never saw the scored activations;
  `--propagate` injects the shift into the running network and reads every later layer.
- **Attentive probes.** Not specified in the paper; `patch --attentive` uses a small
  V-JEPA-style attentive pooler.

### Interpreting dimension counts

The orthogonal-probe count removes each probe's weight directions and retrains. When noise is
anisotropic those directions differ from the directions in which the variable moves the
activations, so signal leaks into later rounds: in the self-test, a ring planted in exactly two
dimensions is reported as 4–14 dimensions depending on the noise. `subspace` therefore also
reports the score after erasing the rank-*k* cross-covariance subspace (k = 2 for an angle),
`manifold` reports a noise-corrected dimensionality of the condition means, and `steer` compares
the probe-subspace intervention with a difference-of-means shift of 2 or 7 dimensions.

## Stimulus sets

| Family | Videos | Varies | Use |
|---|---|---|---|
| `velocity` | 392 | direction × speed × start position | the paper's velocity set |
| `velocity_centered` | 392 | as `velocity`; trajectories centred on the position | removes the mean-position cue |
| `velocity_stepped` | 392 | as `velocity_centered`; ball static within each 2-frame tubelet | removes within-token motion |
| `acceleration` | 280 | direction × acceleration × start | the paper's acceleration set |
| `permanence` | 240 | continue / vanish / teleport / grow behind an occluder | violation-of-expectation stand-in |
| `position` | 196 | floor position (with slow drift) | planar code |
| `hue` | 288 | colour around the hue wheel; direction as nuisance | non-motion circular variable |
| `size` | 128 | radius; direction as nuisance | magnitude code |
| `orbit` | 384 | phase × angular velocity × radius | phase rings, trajectories |
| `two_balls` | 384 | two independent directions | torus: factorised or conjunctive |

`python -m physprobe families` lists every factor. Factor levels can be overridden:
`make velocity_stepped --grid direction_deg=0:360:15 speed=2,4,6 --n-starts 5`.
`make-shuffled --data <set> --block 2` builds an intact-vs-shuffled set (blocks of 2 frames
keep each tubelet intact).

**New families**: `experiments/my_family.py` is a template (a ball that bounces off a wall);
`make <name> --plugin <file.py>` loads it.
**External videos**: a CSV with a `path` column (video file, frame folder or `.npy`) and label
columns: `make-folder --csv videos.csv --out data/mine --target angle:circular hit:binary --group clip`.
**IntPhys**: `make-intphys --root <dev folder> --out data/intphys`.

## Commands

| Command | Purpose |
|---|---|
| `make`, `make-shuffled`, `make-folder`, `make-intphys`, `families` | build datasets |
| `extract` | run an encoder and cache activations (`--fc1-layers`, `--patch-layers`, `--time-layers`, `--attn-stats`, `--random-init`) |
| `probe` | layerwise probes (`--holdout` for generalisation, `--where` for subsets, `--compare` to overlay runs) |
| `subspace` | orthogonal probe sequences, erasure, subspace angles |
| `tuning` | direction- and speed-tuned MLP units |
| `steer` | subspace and difference-of-means steering (`--strict`, `--propagate`) |
| `ablate` | attention ablations (`--layers`, `--conditions`, `--combine`); resumes finished conditions |
| `attn`, `patch` | attention distance; per-patch and attentive probes |
| `manifold` | shape statistics of a variable by layer (`--by`, `--by2` for tori, `--kind fc1`) |
| `report` | collect a run folder's figures and tables into one HTML page |
| `export` | results bundle without activation arrays (for sharing and hosting) |
| `gui` | the explorer |
| `selftest`, `demo` | checks on planted structure; a simulated activation set |

`python -m physprobe <command> --help` lists every option.

## Explorer

```bash
python -m physprobe gui                                   # http://localhost:8501
python -m physprobe gui --runs path/to/runs --data path/to/data
```

| Page | Content |
|---|---|
| Overview | context, definitions, inventory of activation sets, probe runs, ablations and datasets; bundle upload |
| Shape explorer | condition-mean shapes by layer (3-D or 2-D, individual videos optional) with reliability, dimensionality, ring order and harmonics; animation over layers; all layers side by side |
| Probes | overlaid probe curves from any runs; new ridge probe runs on any activation set |
| Attention ablations | probe curves with and without ablation, ablated layers shaded |
| Attention distance | per-head and layer-mean attention distance |
| Datasets | previews; new synthetic sets with custom factor levels; import of external videos |

The explorer reads results and never runs the model; each page lists the command for the
next step.

### Results bundles and hosting

```bash
python -m physprobe export --runs runs --data data --out results --zip
```

writes the probe, ablation and attention results, dataset previews and precomputed shapes
(condition-mean trajectories, video projections and shape statistics) without activation
arrays: a few MB instead of hundreds. Bundles can be uploaded on the explorer's Overview page.

`hosted/streamlit_app.py` runs the explorer read-only on the bundle in `results/`. On Streamlit
Community Cloud: create an app from this repository with main file `hosted/streamlit_app.py`
(its `requirements.txt` excludes torch). Details in [docs/HOSTING.md](docs/HOSTING.md).

## Compute

- V-JEPA 2-L, 16 frames at 224 px: about 7 s per video on 2 CPU cores, so the 392-video set
  takes about 45 minutes on CPU. The results in FINDINGS.md were produced on a 4 GB laptop GPU
  (GTX 1650) in float16; ViT-g needs a larger GPU.
- `extract --attn-stats` and `ablate` compute attention matrices explicitly (about 1.3 GB per
  layer per 8 videos in float32); lower `--batch-size` if memory is tight.
- Probes run on CPU by default (`--probe-device cuda` to move them). Ridge probes over 24 layers
  take seconds; the 20-configuration AdamW sweep takes minutes.
- Activation caches are small: `pooled.npy` for 392 videos × 24 layers × 1024 dimensions is 39 MB.

## Layout

```
physprobe/
  stimuli.py     renderer and stimulus families      datasets.py   datasets: synthetic, folder, IntPhys, shuffled
  models.py      encoder wrapper, hooks, attention   extract.py    activation cache
  probes.py      linear and attentive probes, CV     subspace.py   probe sequences, erasure, overlap, steering
  tuning.py      unit tuning                         manifold.py   shape statistics
  plots.py       figures                             export.py     results bundles
  cli.py         commands                            selftest.py   planted-structure tests, demo data
  gui/           explorer (data layer, figures, app)
hosted/          read-only explorer entry point and its requirements
results/         results bundle shown by the hosted explorer
docs/            result tables, figure script, figures, hosting notes
scripts/         run_replication.sh, run_controls.sh, run_manifold_zoo.sh
experiments/     template for new stimulus families
notebooks/       Colab quick start
tests/           pytest suite
```

## Citation

The analyses reimplemented here are from:

> Sonia Joseph, Quentin Garrido, Randall Balestriero, Matthew Kowal, Thomas Fel,
> Shahab Bakhtiari, Blake Richards, Mike Rabbat. *Interpreting Physics in Video World
> Models.* arXiv:2602.07050, 2026.

Licence: MIT (see `LICENSE`).
