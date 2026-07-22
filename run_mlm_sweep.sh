#!/bin/bash
# Step-2 sweep: depth-2 λ=0, vary MLM softcap and train mask-rate. Compares via the
# val-MLM hook (val_msk_top1 in each run's probe_trajectory.csv). Scratch driver.
set -u
cd /Users/leigh_greathouse/Documents/My_Code/HARMONY/harmony-autoresearch
source .venv/bin/activate 2>/dev/null || true
COMMON="--depth 2 --aspect-ratio 128 --lam 0 --max-steps 10000 --eval-every 2000 \
--log-every 1000 --samples-per-batch 4 --reads-per-sample 8 --reads-cap 200 \
--seq-len 64 --seed 42 --max-runtime-hours 2"
run () {
  name=$1; cap=$2; mask=$3
  echo "=== ARM $name cap=$cap mask=$mask $(date +%H:%M:%S) ==="
  python train_mlm.py --out-dir experiments/mlm_sweep_$name \
    --mlm-softcap $cap --mask-prob $mask $COMMON \
    > experiments/mlm_sweep_$name.log 2>&1
  echo "ARM $name done $(date +%H:%M:%S)"
}
run A_control 15 0.15
run B_capoff  0  0.15
run C_mask10  15 0.10
run D_both    0  0.10
echo "ALL DONE $(date +%H:%M:%S)"
