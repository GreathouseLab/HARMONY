#!/usr/bin/env python3
"""score_candidates.py — AUDIT ONLY. Deliverable 4 arithmetic, all inputs explicit."""
import math

TOK_PER_READ   = 48.27      # audit 2026-10-22
ROWS_PER_STEP  = 384        # 12 ranks x (K=4 x M=8) rows, run_ddp_aurora.sh:41 + NRANKS_PER_NODE=12
SEC_PER_STEP64 = 0.170      # measured Aurora throughput at seq_len 64
BYTES_PER_READ = 1844       # measured, audit/score_candidates provenance: loader token lists
N_SAMPLES      = 126
CORPUS_READS   = 25_200_003

def reads_per_row(T):  return max(1, int(T // TOK_PER_READ))
def reads_per_step(T): return ROWS_PER_STEP * reads_per_row(T)
def distinct(cap):     return min(cap * N_SAMPLES, CORPUS_READS)
def gb_node(cap):      return distinct(cap) * BYTES_PER_READ / 1e9 * 12

def curriculum_reads_per_step(stages):
    return sum(reads_per_step(T) for T in stages) / len(stages)

def row(name, T_desc, cap, steps, rps, note):
    d = distinct(cap)
    passes = steps * rps / d
    # step time scales ~ with tokens/step; attention is superlinear, so this is a LOWER bound at T>64
    sec = SEC_PER_STEP64 * (rps / reads_per_step(64))
    nodeh = steps * sec / 3600
    flag = "DATA-WALL" if passes > 4 else "ok"
    print(f"{name:<4}{T_desc:<14}{cap:>8,}{d:>12,}{steps:>10,}{rps:>10,.0f}{passes:>9.1f}"
          f"{nodeh:>11.1f}{gb_node(cap):>10.0f}  {flag:<10}{note}")

print(f"{'':<4}{'seq_len':<14}{'cap':>8}{'distinct':>12}{'steps':>10}{'reads/st':>10}{'passes':>9}"
      f"{'node-h':>11}{'GB/node':>10}  {'flag':<10}closes")
print("-"*132)
row("A", "64", 5_000, 500_000, reads_per_step(64), "nothing - more steps only")
row("B", "64->512", 5_000, 100_000, curriculum_reads_per_step([64,128,256,512]), "padding + cross-read")
row("C", "64->512 scratch", 5_000, 500_000, curriculum_reads_per_step([64,128,256,512]), "padding + cross-read, discards 500k steps")
row("D", "64 depth-8", 5_000, 200_000, reads_per_step(64), "capacity only")
row("E", "= B (pack is in B)", 5_000, 100_000, curriculum_reads_per_step([64,128,256,512]), "identical to B - _pack_row already packs")
print()
row("F", "64->512", 50_000, 65_625, curriculum_reads_per_step([64,128,256,512]), "padding + cross-read + DATA WALL")
row("G", "64->512", 200_000, 65_625, curriculum_reads_per_step([64,128,256,512]), "same, full corpus (memory risk)")
row("H", "64", 50_000, 65_625, reads_per_step(64), "data wall only, no curriculum")

print()
print("4-pass step budgets (steps = 4 x distinct / reads_per_step):")
for cap in (5_000, 20_000, 50_000, 200_000):
    d = distinct(cap)
    for T, lbl in ((64, "T=64"), (512, "T=512")):
        print(f"  cap {cap:>7,} ({d:>11,} reads)  {lbl:<7} -> {4*d/reads_per_step(T):>10,.0f} steps"
              f"   ({gb_node(cap):>6.0f} GB/node)")
