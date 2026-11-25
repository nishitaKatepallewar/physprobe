# Result tables

Numbers behind [FINDINGS.md](../../FINDINGS.md). All runs: V-JEPA 2 ViT-L
(`facebook/vjepa2-vitl-fpc64-256`), 16 frames at 224 × 224, float16 on a GTX 1650 (4 GB),
November 2026. Figures in `../figures/` are regenerated from these files with
`python docs/make_figures.py`.

Two kinds of file:

**Copied** from the run folders (`probe_summary.csv` and friends, full precision):

| File | Run |
|---|---|
| `velocity_probe.csv`, `velocity_probe_random.csv` | `velocity` set: pretrained weights (ridge probes) / random weights (produced by `run_controls.sh`, which uses ridge; the solver is not stored in the file) |
| `velocity_centered_probe.csv`, `velocity_centered_probe_random.csv` | `velocity_centered`, pretrained / random weights (produced by `run_controls.sh`, which uses ridge; the solver is not stored in the file) |
| `acceleration_probe.csv`, `permanence_probe.csv` | paper protocol |
| `shuffled_frames_probe.csv` | intact vs single-frame-shuffled videos |
| `shuffled_pairs_probe.csv` | intact vs tubelet-pair-shuffled videos (AdamW probes) |
| `ablate_shuffle_L5-6_*.csv` | local attention removed at layers 5–6 (combined, OR), shuffle detection (AdamW probes) |
| `ablate_velocity_L7-8_and_*.csv` | layers 7–8, the three paper conditions, combined = AND (ridge probes) |
| `ablate_velocity_L7-8_or_combined_s3_t1.csv` | layers 7–8, combined = OR |
| `attention_distance.csv` | attention distance per layer, 64 `velocity` videos |
| `velocity_dimension_by_layer.csv`, `velocity_subspace_overlap.csv` | orthogonal probe sequences, erasure, subspace angles (direction, speed, `permanence`) |
| `tuning_by_layer_fc1.csv` | direction- and speed-tuned MLP units |
| `velocity_manifold_direction.csv` | direction manifold statistics, `velocity` |
| `velocity_patch_probe.csv` | per-patch direction probes |
| `velocity_steering_L8.json` | steering at layer 8, strict protocol (one run of the probe sequence) |
| `intphys_probe.csv`, `intphys_O1_probe.csv` | IntPhys dev set, all blocks / object permanence only, per-video accuracy (AdamW probes) |
| `intphys_subspace_overlap.csv`, `intphys_dimension_by_layer.csv` | `subspace` with direction and speed from `velocity` and possible vs impossible from IntPhys |
| `velocity_256px_probe.csv` | `velocity` at 256 × 256 input (ridge probes) |

**Transcribed** from the printed output of later runs, at the precision printed (2–3 decimals):

| File | Command output |
|---|---|
| `velocity_stepped_probe_r2.csv` | `probe` on `velocity_stepped`, ridge, all layers |
| `velocity_stepped_probe_random_r2.csv` | same, random weights (layers 0, 8, 12, 23 printed) |
| `stepped24_probe_r2.csv` | `probe` on `velocity_stepped` with 24 directions (layers 0, 8, 12, 23) |
| `stepped24_manifold_direction.csv`, `velocity_stepped_manifold_direction.csv` | `manifold --by direction_deg` |
| `velocity_stepped_dimension.csv` | `subspace` on `velocity_stepped` |
| `steering_curves.csv` | `steer --strict` on `velocity` (layer 8) and `velocity_stepped` (layer 10) |
| `velocity_steering_propagation_L8.csv` | `steer --strict --propagate`, 64 random held-out videos |
| `ablation_readouts.csv` | `ablate` read-out tables: `velocity` layers 1–4; `velocity_stepped` layers 7–8 and 4–6 |
| `intphys_scoring_check.csv` | IntPhys, logistic probe (C = 0.01), grouped 5-fold: per-video accuracy, accuracy with random folds that split matched scenes, and paired accuracy within matched scenes |
| `velocity_direction_L0_by_speed.csv` | `probe --where "speed == s"` on `velocity`, ridge, layer 0, pretrained and random weights (56 videos per speed); pixels per frame = speed / 24 fps × 28 px/m |

Probe solvers: "ridge" is closed-form ridge regression with the penalty chosen on a grouped
validation split; "AdamW" is the paper's 20-configuration sweep. Both use grouped 5-fold
cross-validation. Every result comes from a single run (one cross-validation split per fold, one seed).
Fold standard deviations are in the copied files; transcribed files have means only.
