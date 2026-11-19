#!/usr/bin/env bash
# Controls and follow-up experiments: when do the paper's findings hold?
# Reproduces every result in FINDINGS.md that goes beyond the paper's own protocol.
# Run scripts/run_replication.sh first (this reuses its velocity and shuffled-pairs
# activations).
#
#   bash scripts/run_controls.sh            # MODEL, RES, BS, DEVICE, OUT, DATA as in run_replication.sh
#
# Roughly 6-7 hours of model time on a 4 GB laptop GPU (GTX 1650); far less on a cloud GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-vjepa2-l}; RES=${RES:-224}; BS=${BS:-4}
OUT=${OUT:-runs/$MODEL}; DATA=${DATA:-data}
PY="python -m physprobe"
M="--model $MODEL --res $RES --batch-size $BS ${DEVICE:+--device $DEVICE} ${EXTRA:-}"
P="--solver ridge"

extract () {  # dataset, output dir, extra args...
  [ -f $2/meta.json ] || $PY extract --data $1 --out $2 $M "${@:3}"
}

echo "== 1. stimulus sets"
[ -d $DATA/velocity ]          || $PY make velocity          --out $DATA/velocity
[ -d $DATA/velocity_centered ] || $PY make velocity_centered --out $DATA/velocity_centered
[ -d $DATA/velocity_stepped ]  || $PY make velocity_stepped  --out $DATA/velocity_stepped
[ -d $DATA/stepped24 ]         || $PY make velocity_stepped  --out $DATA/stepped24 \
                                      --grid direction_deg=0:360:15 speed=2,4,6 --n-starts 5
[ -d $DATA/shuffled_pairs ]    || $PY make-shuffled --data $DATA/velocity --out $DATA/shuffled_pairs --block 2

echo "== 2. pretrained and untrained activations"
extract $DATA/velocity          $OUT/velocity/acts --fc1-layers all --fc2-layers all --patch-layers thirds --time-layers thirds
extract $DATA/shuffled_pairs    $OUT/shuffled_pairs/acts
for d in velocity velocity_centered velocity_stepped; do
  extract $DATA/$d $OUT/$d/acts
  extract $DATA/$d $OUT/$d/acts_random --random-init
done
extract $DATA/stepped24 $OUT/stepped24/acts

echo "== 3. probes: is the early signal learned, or already in the pixels?"
for d in velocity velocity_centered velocity_stepped; do
  $PY probe --acts $OUT/$d/acts        --out $OUT/$d/probe        $P
  $PY probe --acts $OUT/$d/acts_random --out $OUT/$d/probe_random $P
  $PY probe --acts $OUT/$d/acts        --out $OUT/$d/probe_vs_random $P --compare untrained=$OUT/$d/probe_random
done
$PY probe --acts $OUT/shuffled_pairs/acts --out $OUT/shuffled_pairs/probe
$PY probe --acts $OUT/stepped24/acts      --out $OUT/stepped24/probe $P

echo "== 4. attention ablations: where is local attention needed?"
$PY ablate --data $DATA/velocity         --out $OUT/ablate_velocity_L1-4  $M --base-acts $OUT/velocity/acts         $P --layers 1,2,3,4 --conditions combined
$PY ablate --data $DATA/velocity_stepped --out $OUT/ablate_stepped_L7-8   $M --base-acts $OUT/velocity_stepped/acts $P
$PY ablate --data $DATA/velocity_stepped --out $OUT/ablate_stepped_L4-6   $M --base-acts $OUT/velocity_stepped/acts $P --layers 4,5,6 --conditions combined
$PY ablate --data $DATA/velocity         --out $OUT/ablate_velocity_and   $M --base-acts $OUT/velocity/acts         $P --combine and --conditions combined
$PY ablate --data $DATA/shuffled_pairs   --out $OUT/ablate_shuffle_L5-6   $M --base-acts $OUT/shuffled_pairs/acts     --layers 5,6 --conditions combined

echo "== 5. geometry on the stepped videos"
$PY subspace --out $OUT/stepped_subspace --layers 4,6,8,10,12,23 \
    --task direction=$OUT/velocity_stepped/acts:direction_deg:circular speed=$OUT/velocity_stepped/acts:speed:scalar
$PY steer --acts $OUT/velocity_stepped/acts --out $OUT/stepped_steer --strict --layer 10
$PY manifold --acts $OUT/velocity_stepped/acts --out $OUT/stepped_manifold --by direction_deg
$PY manifold --acts $OUT/stepped24/acts        --out $OUT/stepped24/manifold --by direction_deg --embed isomap

$PY report --run $OUT
echo "done: open $OUT/report.html"
