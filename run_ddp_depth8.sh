#!/bin/bash
# ── HARMONY depth-8 model-scaling run on Aurora ───────────────────────────────
# Same recipe/data as the depth-6 24h run, but a BIGGER model (depth 8, width 1536).
# Answers: are we data-bound or capacity-bound? If val plateaus at ~0.245 like depth-6,
# capacity is not the limit; if it climbs higher, bigger models help.
#   qsub run_ddp_depth8.sh
#
#PBS -A BioReason-Aurora-Test
#PBS -q capacity
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -l filesystems=home:flare
#PBS -N harmony_d8

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

echo "[ddp-d8] nodes=$NNODES ranks=$NRANKS master=$MASTER_ADDR"

# depth 8 × aspect 192 -> width 1536 (vs depth-6's 1152). Bigger model, same data/recipe.
mpiexec -n "$NRANKS" --ppn "$NRANKS_PER_NODE" \
  python train_mlm.py \
    --out-dir experiments/ddp_depth8 \
    --train-txt output/train.txt \
    --val-txt output/val.txt \
    --depth 8 --aspect-ratio 192 --lam 0 \
    --reads-cap 5000 --samples-per-batch 4 --reads-per-sample 8 --seq-len 64 \
    --max-steps 600000 --eval-every 5000 --seed 42 \
    --mlm-softcap 15 --mask-prob 0.15 --max-runtime-hours 23.5

# To CONTINUE this run later (after the 24h wall), resume from its latest checkpoint:
#   ... same command ... --resume experiments/ddp_depth8/latest.pt
