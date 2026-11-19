#!/usr/bin/env bash
# Beyond the paper: which variables trace which shapes? One family per expected manifold.
#   bash scripts/run_manifold_zoo.sh            (MODEL, RES, BS, DEVICE as in run_replication.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-vjepa2-l}; RES=${RES:-224}; BS=${BS:-8}
OUT=${OUT:-runs/$MODEL/zoo}; DATA=${DATA:-data}
PY="python -m physprobe"
M="--model $MODEL --res $RES --batch-size $BS ${DEVICE:+--device $DEVICE} ${EXTRA:-}"

run () {  # family, extra make args..., then "--" and manifold args
  fam=$1; shift; mk=(); while [ "$1" != "--" ]; do mk+=("$1"); shift; done; shift
  [ -d $DATA/$fam ] || $PY make $fam --out $DATA/$fam "${mk[@]}"
  [ -f $OUT/$fam/acts/meta.json ] || $PY extract --data $DATA/$fam --out $OUT/$fam/acts $M --time-layers thirds
  $PY probe --acts $OUT/$fam/acts --out $OUT/$fam/probe --solver ridge
  $PY manifold --acts $OUT/$fam/acts --out $OUT/$fam/manifold "$@"
}

# ring, finely sampled: 24 directions instead of the paper's 8 (resolves harmonics up to 12)
[ -d $DATA/direction24 ] || $PY make velocity --out $DATA/direction24 --grid direction_deg=0:360:15 speed=2,4,6 --n-starts 5
[ -f $OUT/direction24/acts/meta.json ] || $PY extract --data $DATA/direction24 --out $OUT/direction24/acts $M --time-layers thirds
$PY manifold --acts $OUT/direction24/acts --out $OUT/direction24/manifold --by direction_deg --embed isomap

run hue        -- --by hue_deg --embed isomap     # a circular variable that is not motion
run size       -- --by radius                     # a line
run position   -- --by x                          # one axis of a plane (repeat with --by y)
run orbit      -- --by phase_deg                  # phase ring; see time_trajectories_*.png
run two_balls  -- --by dir_a_deg --by2 dir_b_deg  # torus: factorised or conjunctive?

$PY report --run $OUT
echo "done: open $OUT/report.html"
