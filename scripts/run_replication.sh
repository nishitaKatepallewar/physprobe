#!/usr/bin/env bash
# End-to-end replication run for "Interpreting Physics in Video World Models"
# (arXiv 2602.07050). Each block maps to a figure/table; see README.md.
#
#   bash scripts/run_replication.sh                       # V-JEPA 2-L, synthetic sets
#   MODEL=vjepa2-g bash scripts/run_replication.sh        # another size (vjepa2-h, vjepa2-g)
#   INTPHYS=/path/to/intphys/dev bash scripts/run_replication.sh   # add the real IntPhys dev set
#   QUICK=1 bash scripts/run_replication.sh               # ridge probes, fewer layers (CPU-friendly)
#   BS=4 bash scripts/run_replication.sh                  # smaller batches (4 GB GPUs)
#
# This follows the paper's protocol. The controls and follow-up experiments that test
# when its findings hold are in scripts/run_controls.sh (see FINDINGS.md).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-vjepa2-l}
RES=${RES:-224}
BS=${BS:-8}
OUT=${OUT:-runs/$MODEL}
DATA=${DATA:-data}
PY="python -m physprobe"
M="--model $MODEL --res $RES --batch-size $BS ${DEVICE:+--device $DEVICE} ${EXTRA:-}"
if [ "${QUICK:-0}" = 1 ]; then PROBE="--solver ridge"; SUBL="thirds"; else PROBE="--solver adamw"; SUBL="all"; fi

echo "== 1. stimuli (App. A.1)"
[ -d $DATA/velocity ]     || $PY make velocity     --out $DATA/velocity
[ -d $DATA/acceleration ] || $PY make acceleration --out $DATA/acceleration
[ -d $DATA/permanence ]   || $PY make permanence   --out $DATA/permanence
# Frames are shuffled in pairs (one tubelet each) so that motion inside every token stays
# natural and only the order across tokens is broken. Single-frame shuffling is detected
# trivially at layer 0 (see FINDINGS.md).
[ -d $DATA/shuffled_pairs ] || $PY make-shuffled --data $DATA/velocity --out $DATA/shuffled_pairs --block 2
VOE=permanence            # violation-of-expectation set used below
if [ -n "${INTPHYS:-}" ]; then
  [ -d $DATA/intphys ] || $PY make-intphys --root "$INTPHYS" --out $DATA/intphys
  VOE=intphys
fi

echo "== 2. activations"
[ -f $OUT/velocity/acts/meta.json ] || $PY extract --data $DATA/velocity --out $OUT/velocity/acts $M \
    --fc1-layers all --fc2-layers all --patch-layers thirds --time-layers thirds
for d in acceleration shuffled_pairs $VOE; do
  [ -f $OUT/$d/acts/meta.json ] || $PY extract --data $DATA/$d --out $OUT/$d/acts $M
done

echo "== 3. layerwise probes (Fig. 1, Fig. 2, Sec. 6.1 shuffled control)"
for d in velocity acceleration shuffled_pairs $VOE; do
  $PY probe --acts $OUT/$d/acts --out $OUT/$d/probe $PROBE
done

echo "== 4. subspaces: orthogonal probe sequences + overlap (App. C.11, C.4, Table 3)"
$PY subspace --out $OUT/subspace --layers $SUBL \
    --task direction=$OUT/velocity/acts:direction_deg:circular \
           speed=$OUT/velocity/acts:speed:scalar \
           voe=$OUT/$VOE/acts:possible:binary

echo "== 5. direction / speed tuned MLP units (Fig. 4, App. C.7-C.8)"
$PY tuning --acts $OUT/velocity/acts --out $OUT/tuning --kind fc1
$PY tuning --acts $OUT/velocity/acts --out $OUT/tuning --kind fc2

echo "== 6. steering at one-third depth (Sec. 7.2, App. C.12), paper protocol then strict"
$PY steer --acts $OUT/velocity/acts --out $OUT/steer
$PY steer --acts $OUT/velocity/acts --out $OUT/steer_strict --strict --propagate $M

echo "== 7. attention: head distances (Fig. 3) and ablations (Table 2)"
[ -f $OUT/velocity/acts_attn/meta.json ] || $PY extract --data $DATA/velocity --out $OUT/velocity/acts_attn \
    $M --batch-size 2 --limit 64 --attn-stats
$PY attn --acts $OUT/velocity/acts_attn --out $OUT/attention
$PY ablate --data $DATA/velocity --out $OUT/ablate_velocity $M --base-acts $OUT/velocity/acts $PROBE
$PY ablate --data $DATA/$VOE     --out $OUT/ablate_$VOE     $M --base-acts $OUT/$VOE/acts     $PROBE

echo "== 8. per-patch probes (App. C.5)"
$PY patch --acts $OUT/velocity/acts --out $OUT/patch

echo "== 9. geometry of the direction and speed manifolds"
$PY manifold --acts $OUT/velocity/acts --out $OUT/manifold_direction --by direction_deg --embed isomap
$PY manifold --acts $OUT/velocity/acts --out $OUT/manifold_speed --by speed

$PY report --run $OUT
echo "done: open $OUT/report.html"
