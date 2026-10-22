#!/usr/bin/env python3
"""ab_compare.py — AUDIT ONLY. D2 comparison at the largest COMMON step (matched tokens)."""
import csv, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
TOK, RPS = 48.27, 32
ARMS = [("SMALL-CAP", "ab_smallcap", 100, 12_600), ("LARGE-CAP", "ab_largecap", 1_000, 126_000)]

def load(d):
    p = REPO / "experiments" / d / "probe_trajectory.csv"
    rows = {}
    if not p.exists(): return rows
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                rows[int(r["step"])] = (float(r["val_msk_top1"]), float(r["val_msk_ce"]),
                                        float(r["train_msk_top1"]), float(r["train_msk_ce"]))
            except (KeyError, ValueError): pass
    return rows

data = {name: load(d) for name, d, _, _ in ARMS}
for n, v in data.items():
    print(f"{n}: {len(v)} eval points, steps {sorted(v)[:1]}..{sorted(v)[-1:]}")
common = sorted(set(data["SMALL-CAP"]) & set(data["LARGE-CAP"]))
if not common:
    sys.exit("no common eval step yet")
S = common[-1]
toks = S * RPS * TOK
print(f"\nMatched comparison at the largest common step: {S:,}")
print(f"  tokens = {S:,} steps x {RPS} reads/step x {TOK} tok/read = {toks:,.0f} useful tokens  (IDENTICAL for both arms)\n")
print(f"{'arm':<12}{'cap':>7}{'distinct':>11}{'passes':>9}{'val top-1':>11}{'val CE':>10}{'train top-1':>13}{'train-val gap':>15}")
res = {}
for name, d, cap, distinct in ARMS:
    vt, vce, tt, tce = data[name][S]
    passes = S * RPS / distinct
    res[name] = (vt, vce)
    print(f"{name:<12}{cap:>7,}{distinct:>11,}{passes:>9.2f}{vt:>11.4f}{vce:>10.4f}{tt:>13.4f}{tt-vt:>+15.4f}")

d_top1 = res["LARGE-CAP"][0] - res["SMALL-CAP"][0]
d_ce = res["LARGE-CAP"][1] - res["SMALL-CAP"][1]
# jitter = max |step-to-step change| over the last three evals of either arm
jit = 0.0
for name in data:
    s = sorted(data[name])[-3:]
    for a, b in zip(s, s[1:]):
        jit = max(jit, abs(data[name][b][0] - data[name][a][0]))
print(f"\nLARGE-CAP minus SMALL-CAP : val top-1 {d_top1:+.4f}   val CE {d_ce:+.4f}")
print(f"run-to-run jitter (max |delta top-1| over last 3 evals, either arm) : {jit:.4f}")
if abs(d_top1) <= jit:
    print("\nVERDICT: NO DIFFERENCE (gap within jitter) -> slice is representative; compute-bound reading stands.")
elif d_top1 > 0:
    print("\nVERDICT: LARGE-CAP BETTER -> repetition is costing accuracy; data, not compute, was the limiter.")
else:
    print("\nVERDICT: SMALL-CAP BETTER -> reported as measured; the data-wall argument needs rethinking.")
