#!/bin/bash
# ── HARMONY depth-6 with SEQ-LEN CURRICULUM on Aurora ─────────────────────────
# Nick's design: don't cap context — let it grow, scheduled so it's never too hard.
# Start at 64 tokens (~1 read), and each time val CE plateaus, DOUBLE the window
# (packing more same-sample reads per row) up to a 512 ceiling (~8 reads of context).
# This is also our first look at cross-read / within-sample structure — the Harmonizer's substrate.
#   qsub run_ddp_curriculum.sh
#
#PBS -A BioReason-Aurora-Test
#PBS -q capacity
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -l filesystems=home:flare
#PBS -N harmony_curric

set -eo pipefail
cd "${PBS_O_WORKDIR:-.}"

module use /soft/modulefiles
module load frameworks

NRANKS_PER_NODE=12
NNODES=$(wc -l < "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || echo 1)
NRANKS=$(( NNODES * NRANKS_PER_NODE ))
export WORLD_SIZE=$NRANKS
export MASTER_ADDR=$(head -1 "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || echo 127.0.0.1)
export MASTER_PORT=29500
export CCL_ZE_IPC_EXCHANGE=${CCL_ZE_IPC_EXCHANGE:-pidfd}

echo "[ddp-curric] nodes=$NNODES ranks=$NRANKS master=$MASTER_ADDR"

# --seq-len 64 = start window; --seq-len-max 512 = ceiling (turns the curriculum on).
# Window doubles after --seq-len-patience evals with no val-CE gain > --seq-len-min-delta.
mpiexec -n "$NRANKS" --ppn "$NRANKS_PER_NODE" \
  python train_mlm.py \
    --out-dir experiments/ddp_curriculum \
    --train-txt output/train.txt \
    --val-txt output/val.txt \
    --depth 6 --aspect-ratio 192 --lam 0 \
    --reads-cap 5000 --samples-per-batch 4 --reads-per-sample 8 \
    --seq-len 64 --seq-len-max 512 --seq-len-patience 3 --seq-len-min-delta 0.01 \
    --max-steps 600000 --eval-every 5000 --seed 42 \
    --mlm-softcap 15 --mask-prob 0.15 --max-runtime-hours 23.5

# Continue past the wall (keeps the grown window via the checkpoint):
#   ... same command ... --resume experiments/ddp_curriculum/latest.pt
