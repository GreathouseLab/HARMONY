#!/bin/bash
# ── HARMONY next run — DATA-WALL TEST ────────────────────────────────────────────────
# Recommendation from audit/NEXT_RUN_DECISION_2026-10-22.md (Deliverable 5).
# REVIEW BEFORE SUBMITTING. Not launched by the audit.
#
# ONE variable changes vs run_ddp_aurora.sh: --reads-cap 5000 -> 50000.
# Everything else (depth, aspect ratio, seq_len, batch recipe, mask rate, softcap, seed)
# is held identical so the comparison is clean.
#
# Why: the 0.248 run made ~305 passes over 630,000 distinct reads (2.5% of the 25.2M-read
# corpus). The ~4-pass repetition budget was exceeded ~76x. This run holds compute LOW and
# raises distinct data 10x instead.
#
# Submit with:  qsub launch/next_run_2026-10-22.sh

#PBS -A BioReason-Aurora-Test
#PBS -q capacity
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -l filesystems=home:flare
#PBS -N harmony_datawall

set -eo pipefail
cd "${PBS_O_WORKDIR:-.}"

module use /soft/modulefiles
module load frameworks

# ---- rank topology ----
NRANKS_PER_NODE=12
NNODES=$(wc -l < "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || echo 1)
NRANKS=$(( NNODES * NRANKS_PER_NODE ))
export WORLD_SIZE=$NRANKS
export MASTER_ADDR=$(head -1 "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || echo 127.0.0.1)
export MASTER_PORT=29500
export CCL_ZE_IPC_EXCHANGE=${CCL_ZE_IPC_EXCHANGE:-pidfd}

echo "[ddp] nodes=$NNODES ranks=$NRANKS master=$MASTER_ADDR"

# ---- MEMORY PRECONDITION — FALLBACK IS ACTIVE --------------------------------------
# Loader holds tokenized reads in Python lists: 1,844 bytes/read measured.
#   --reads-cap 50000 x 126 samples = 6,300,000 reads = 11.6 GB/rank x 12 = 139 GB/node
#   --reads-cap 20000 x 126 samples = 2,520,000 reads =  4.6 GB/rank x 12 =  56 GB/node
#
# The >= 160 GB free check COULD NOT BE PERFORMED: this script was prepared on the Mac,
# which has no PBS client (qsub/qstat/pbsnodes absent) and no configured ALCF access.
# "Unverified" is not "confirmed", so the SAFE FALLBACK is active below: cap 20000 at
# 26,250 steps, which is still exactly 4.0 passes and needs only ~56 GB/node.
#
# TO RESTORE THE FULL RUN after checking free memory on a compute node
# (e.g.  qsub -I -A BioReason-Aurora-Test -q debug -l select=1 -l walltime=00:10:00
#        then: free -g   ->  need >= 160 GB available):
#   change  --reads-cap 20000  ->  --reads-cap 50000
#   change  --max-steps 26250  ->  --max-steps 65625
# Both numbers must change together or the 4.0-pass property breaks.
# ------------------------------------------------------------------------------------

# ---- STEP BUDGET ------------------------------------------------------------------
# Active (fallback): 4.0 passes = 4 x 2,520,000 / 384 reads-per-step = 26,250 steps.
# Full run:          4.0 passes = 4 x 6,300,000 / 384 reads-per-step = 65,625 steps.
# Do not raise either without recomputing passes; beyond ~4 passes repetition stops
# paying (Muennighoff 2023). train_mlm.py now prints tokens_seen and passes at every
# eval and writes them to probe_trajectory.csv, so this is checkable while it runs.
# ------------------------------------------------------------------------------------

mpiexec -n "$NRANKS" --ppn "$NRANKS_PER_NODE" \
  python train_mlm.py \
    --out-dir experiments/ddp_datawall_cap50k \
    --train-txt output/train.txt \
    --depth 6 --aspect-ratio 192 --lam 0 \
    --reads-cap 20000 --samples-per-batch 4 --reads-per-sample 8 --seq-len 64 \
    --max-steps 26250 --eval-every 1250 --seed 42 \
    --mlm-softcap 15 --mask-prob 0.15 --max-runtime-hours 3.5

# ---- SUCCESS CRITERION ------------------------------------------------------------
# Pair (N, D): N = 119.48M params, D = 26,250 x 384 x 48.27 = 486.7M useful tokens
# (full run: 65,625 x 384 x 48.27 = 1.216B useful tokens).
# Compare against the 0.248 Aurora run AT ITS OWN 1.216B-token point (its step ~65,625
# in probe_trajectory.csv) -- a matched-token comparison, not matched-steps.
#   PASS  : val_msk_top1 at 1.216B tokens >= the old run's value at 1.216B tokens.
#           -> repetition was hurting; data scaling is the lever; re-plan the scale-up.
#   FAIL  : materially below it.
#           -> repetition was not the binding constraint; compute is the lever after all.
# Secondary: centered Evaluation 1 AUC (probe_sample_coherence.py --center) against the
# 0.5329 baseline measured on mlm_bigcohort_d6 in audit/centered_rerun.json.
#
# STOPPING RULE: stop at step 65,625 (4.0 passes), or earlier if val CE fails to improve
# by > 0.01 nats across 5 consecutive evals. Do NOT extend past 65,625 on this corpus cap.
# ------------------------------------------------------------------------------------
