#!/bin/bash
# ── HARMONY 2-NODE DDP validation on Aurora ───────────────────────────────────
# Plumbing test for MULTI-NODE data parallelism (never tested >1 node). Confirms the
# cross-node all-reduce works before we trust a long multi-node run. Short + cheap.
#   qsub run_ddp_2node.sh
# Expect in the .o log:  "[ddp] nodes=2 ranks=24",  "DDP world_size=24",
# step-0 loss ~8.3, and loss DESCENDING — same as single-node, just 24 ranks.
#
# debug queue allows up to 2 nodes / 1 h — perfect for this check.
#PBS -A BioReason-Aurora-Test
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -N harmony_2node

set -eo pipefail
cd "${PBS_O_WORKDIR:-.}"

module use /soft/modulefiles
module load frameworks

NRANKS_PER_NODE=12
NNODES=$(wc -l < "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || echo 1)
NRANKS=$(( NNODES * NRANKS_PER_NODE ))                  # 2 nodes × 12 = 24 ranks
export WORLD_SIZE=$NRANKS                                # multi-node MUST export this (PALS has no global var)
export MASTER_ADDR=$(head -1 "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || echo 127.0.0.1)
export MASTER_PORT=29500
export CCL_ZE_IPC_EXCHANGE=${CCL_ZE_IPC_EXCHANGE:-pidfd}

echo "[ddp-2node] nodes=$NNODES ranks=$NRANKS master=$MASTER_ADDR"

# Short shakedown: tiny data cap, no eval, ~2000 steps. We only care that 24 ranks across
# 2 nodes init the process group and the loss goes down (cross-node grads are being averaged).
mpiexec -n "$NRANKS" --ppn "$NRANKS_PER_NODE" \
  python train_mlm.py \
    --out-dir experiments/_ddp_2node_check \
    --train-txt output/train.txt \
    --val-txt output/val.txt \
    --depth 6 --aspect-ratio 192 --lam 0 \
    --reads-cap 500 --samples-per-batch 4 --reads-per-sample 8 --seq-len 64 \
    --max-steps 2000 --eval-every 0 --seed 42 \
    --mlm-softcap 15 --mask-prob 0.15 --max-runtime-hours 0.4
