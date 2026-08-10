#!/bin/bash
# ── HARMONY DDP on Aurora (ANL) — data-parallel training across GPU tiles ──────────────
# One rank per Intel GPU tile (12 tiles/node = 6 GPUs x 2). Manual gradient all-reduce
# (see device_utils.py). Submit with:  qsub run_ddp_aurora.sh
#
# PBS resources — edit -A / -q / walltime for your allocation:
#PBS -A BioReason-Aurora-Test
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -N harmony_ddp

set -euo pipefail
cd "${PBS_O_WORKDIR:-.}"

module use /soft/modulefiles
module load frameworks

# ---- rank topology (works single- or multi-node) ----
NRANKS_PER_NODE=12                                   # 12 tiles per Aurora node
NNODES=$(wc -l < "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || echo 1)
NRANKS=$(( NNODES * NRANKS_PER_NODE ))
export WORLD_SIZE=$NRANKS                             # PALS exposes no global-nranks var; set it ourselves
export MASTER_ADDR=$(head -1 "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || echo 127.0.0.1)
export MASTER_PORT=29500

# ---- backend: our code auto-picks 'ccl' on xpu; override here if your stack wants xccl ----
# export HARMONY_DDP_BACKEND=xccl
export CCL_ZE_IPC_EXCHANGE=${CCL_ZE_IPC_EXCHANGE:-drmfd}   # oneCCL hint (per ALCF docs)

echo "[ddp] nodes=$NNODES ranks=$NRANKS master=$MASTER_ADDR"

# mpiexec sets PALS_*/PMI_* per rank; device_utils.init_distributed() reads them.
# NOTE: if GPU-tile binding is off, wrap the python call with ALCF's gpu_tile_compact.sh.
mpiexec -n "$NRANKS" --ppn "$NRANKS_PER_NODE" \
  python train_mlm.py \
    --out-dir experiments/ddp_depth6 \
    --train-txt output/train.txt \
    --depth 6 --aspect-ratio 192 --lam 0 \
    --reads-cap 5000 --samples-per-batch 4 --reads-per-sample 8 --seq-len 64 \
    --max-steps 40000 --eval-every 2000 --seed 42 \
    --mlm-softcap 15 --mask-prob 0.15 --max-runtime-hours 3

# ── FIRST-TIME VALIDATION (run this smaller command before the depth-6 job above) ──
# Grab a node interactively and confirm 12-tile DDP works + step-0 loss is sane:
#   qsub -I -A BioReason-Aurora-Test -q debug -l select=1 -l walltime=00:20:00 -l filesystems=home:flare
#   module use /soft/modulefiles && module load frameworks
#   cd /flare/BioReason-Aurora-Test/klgreathouse/HARMONY
#   export MASTER_ADDR=$(hostname) MASTER_PORT=29500
#   mpiexec -n 12 --ppn 12 python train_mlm.py --out-dir experiments/_ddp_check \
#     --train-txt output/val.txt --depth 2 --aspect-ratio 128 --lam 0 \
#     --max-steps 200 --eval-every 0 --reads-cap 50 --seed 42 --max-runtime-hours 1
#   # expect: "DDP world_size=12", step-0 ~8.3, loss descending, printed once (rank 0).
