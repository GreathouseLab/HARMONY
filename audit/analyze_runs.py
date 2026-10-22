#!/usr/bin/env python3
"""
analyze_runs.py — AUDIT ONLY. Reads logs/CSVs already on disk; writes nothing but stdout.

D1: tokens-seen and passes-over-distinct-reads for every run with a local record.
D2: entropy-floor fit L(D) = E + B / D^beta over tokens-seen, with bootstrap CI.
"""
from __future__ import annotations
import csv, glob, os, re, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

BP_PER_TOKEN = 4.290          # audit/TOKENIZER_SEQLEN_AUDIT_2026-10-22.md
TOK_PER_READ = 48.27          # same audit, mean
SEQ_LEN = 64

LOG_RE = {
    "reads":  re.compile(r"parsed (\d+) samples, (\d+) reads"),
    "batch":  re.compile(r"K=(\d+) samples . M=(\d+) reads = batch size (\d+)"),
    "model":  re.compile(r"model: depth=(\d+) n_embd=(\d+) params=([0-9.]+)M"),
    "final":  re.compile(r"MAX_RUNTIME reached at step (\d+)"),
}

def run_config(log_path):
    cfg = {}
    for line in open(log_path, errors="ignore"):
        for k, rx in LOG_RE.items():
            m = rx.search(line)
            if m and k not in cfg:
                cfg[k] = m.groups()
    return cfg

def traj(run_dir):
    p = os.path.join(run_dir, "probe_trajectory.csv")
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                out.append((int(row["step"]), float(row["val_msk_ce"]), float(row["val_msk_top1"])))
            except (KeyError, ValueError):
                pass
    return sorted(out)

# ---------------- D1 ----------------
print("=" * 108)
print("D1 — tokens seen and passes over distinct reads")
print("=" * 108)
hdr = f"{'run':<22}{'depth':>6}{'params':>9}{'reads/st':>9}{'distinct':>11}{'steps':>9}{'useful tok':>14}{'padded tok':>14}{'passes':>9}  flag"
print(hdr); print("-" * len(hdr))

rows = []
for log in sorted(glob.glob("experiments/mlm_*.log")):
    run = os.path.basename(log)[:-4]
    cfg = run_config(log)
    if "reads" not in cfg or "batch" not in cfg:
        continue
    distinct = int(cfg["reads"][1])
    rps = int(cfg["batch"][2])
    depth = cfg.get("model", ("?", "?", "?"))[0]
    params = cfg.get("model", ("?", "?", "?"))[2]
    t = traj(os.path.join("experiments", run))
    steps = int(cfg["final"][0]) if "final" in cfg else (t[-1][0] if t else 0)
    useful = steps * rps * TOK_PER_READ
    padded = steps * rps * SEQ_LEN
    passes = steps * rps / distinct if distinct else float("nan")
    flag = "DATA-WALL" if passes > 4 else ""
    rows.append((run, depth, params, rps, distinct, steps, useful, padded, passes, flag))
    print(f"{run:<22}{depth:>6}{params:>9}{rps:>9}{distinct:>11,}{steps:>9,}{useful:>14,.0f}{padded:>14,.0f}{passes:>9.2f}  {flag}")

# Aurora run — no local record; estimated from run_ddp_aurora.sh
A_STEPS, A_RPS, A_DISTINCT = 500_000, 384, 630_000
a_passes = A_STEPS * A_RPS / A_DISTINCT
print(f"{'ddp_depth6 (ESTIMATED)':<22}{'6':>6}{'119.48':>9}{A_RPS:>9}{A_DISTINCT:>11,}{A_STEPS:>9,}"
      f"{A_STEPS*A_RPS*TOK_PER_READ:>14,.0f}{A_STEPS*A_RPS*SEQ_LEN:>14,.0f}{a_passes:>9.2f}  DATA-WALL")

print()
print(f"Corpus actually available in output/train.txt : 25,200,003 reads (200,000 x 126 samples)")
print(f"Corpus the loader uses at --reads-cap 5000    :    630,000 reads  = 2.50% of it")
print(f"4-pass budget at cap 5000  : {4*630_000:,} read-presentations = {4*630_000/384:,.0f} steps at 384 reads/step")
print(f"4-pass budget at FULL data : {4*25_200_003:,} read-presentations = {4*25_200_003/384:,.0f} steps at 384 reads/step")

# ---------------- D2 ----------------
print()
print("=" * 108)
print("D2 — entropy-floor fit  L(D) = E + B / D^beta")
print("=" * 108)

def fit(D, L):
    """Grid over E; for each, least-squares line in log(L-E) vs log(D). Returns best (E,B,beta,rmse)."""
    best = None
    for E in np.linspace(0.0, min(L) - 1e-3, 4000):
        y = np.log(L - E)
        x = np.log(D)
        b, a = np.polyfit(x, y, 1)
        pred = E + np.exp(a) * D ** b
        rmse = float(np.sqrt(np.mean((L - pred) ** 2)))
        if best is None or rmse < best[3]:
            best = (float(E), float(np.exp(a)), float(-b), rmse)
    return best

for run in ("mlm_bigcohort_d6", "mlm_bigcohort_d4", "mlm_big", "mlm_moredata"):
    t = traj(os.path.join("experiments", run))
    if len(t) < 4:
        print(f"{run}: only {len(t)} usable checkpoints — fewer than 4, no fit attempted.")
        continue
    cfg = run_config(f"experiments/{run}.log") if os.path.exists(f"experiments/{run}.log") else {}
    rps = int(cfg["batch"][2]) if "batch" in cfg else 32
    D = np.array([s * rps * TOK_PER_READ for s, _, _ in t], dtype=float)
    L = np.array([ce for _, ce, _ in t], dtype=float)
    E, B, beta, rmse = fit(D, L)
    pred = E + B * D ** (-beta)
    resid = L - pred

    boot = []
    rng = np.random.default_rng(42)
    for _ in range(400):
        idx = rng.choice(len(D), len(D), replace=True)
        if len(set(idx.tolist())) < 4:
            continue
        try:
            boot.append(fit(D[idx], L[idx])[0])
        except Exception:
            pass
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) > 20 else (float("nan"),) * 2)

    # Has the curve begun to flatten? Compare the last decade's slope to the first's.
    half = len(L) // 2
    slope_early = (np.log(L[half - 1]) - np.log(L[0])) / (np.log(D[half - 1]) - np.log(D[0]))
    slope_late = (np.log(L[-1]) - np.log(L[half])) / (np.log(D[-1]) - np.log(D[half]))
    flattening = slope_late > slope_early * 0.6   # late slope materially shallower

    print(f"\n--- {run}  (n={len(t)} eval points, depth={cfg.get('model',('?',))[0]}, "
          f"params={cfg.get('model',('?','?','?'))[2]}M)")
    print(f"  val CE range      : {L[0]:.4f} -> {L[-1]:.4f} nats/token  over D = {D[0]:,.0f} -> {D[-1]:,.0f} tokens")
    print(f"  fit               : E={E:.4f}  B={B:.4g}  beta={beta:.4f}   RMSE={rmse:.5f}")
    print(f"  bootstrap 95% CI for E : [{lo:.4f}, {hi:.4f}]")
    print(f"  residuals         : min={resid.min():+.5f}  max={resid.max():+.5f}  mean={resid.mean():+.6f}")
    print(f"  log-log slope     : early {slope_early:+.4f}  late {slope_late:+.4f}")
    if not flattening:
        print("  VERDICT: curve has NOT begun to flatten — E is UNIDENTIFIABLE. Do not quote the number above.")
    else:
        print(f"  VERDICT: some flattening. E = {E:.4f} nats/token = {E/np.log(2)/BP_PER_TOKEN:.4f} bits/base.")
        print(f"           best val CE {L[-1]:.4f} is {L[-1]-E:+.4f} nats ({100*(L[-1]-E)/max(E,1e-9):.1f}%) above E.")
