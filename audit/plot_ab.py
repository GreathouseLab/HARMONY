#!/usr/bin/env python3
"""plot_ab.py — AUDIT ONLY. Both D2 curves vs tokens-seen, matched token budget."""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
TOK_PER_READ, READS_PER_STEP = 48.27, 32
RUNS = [("SMALL-CAP  cap 100  (12,600 distinct)", "experiments/ab_smallcap", "#b4531f"),
        ("LARGE-CAP  cap 1000 (126,000 distinct)", "experiments/ab_largecap", "#0d7c86")]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 7.0), sharex=True)
any_data = False
for label, d, colour in RUNS:
    p = REPO / d / "probe_trajectory.csv"
    if not p.exists(): continue
    steps, top1, ce = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                steps.append(int(r["step"]) * READS_PER_STEP * TOK_PER_READ)
                top1.append(float(r["val_msk_top1"])); ce.append(float(r["val_msk_ce"]))
            except (KeyError, ValueError): pass
    if not steps: continue
    any_data = True
    ax1.plot(steps, top1, color=colour, lw=1.6, label=label)
    ax2.plot(steps, ce, color=colour, lw=1.6, label=label)

if not any_data:
    sys.exit("no trajectory data yet")

ax1.set_ylabel("val MSK-DNA top-1"); ax1.legend(frameon=False, fontsize=8, loc="lower right")
ax1.set_title("Matched-token A/B — identical compute, 10x difference in distinct reads", fontsize=10)
ax2.set_ylabel("val cross-entropy (nats)"); ax2.set_xlabel("useful tokens seen")
for a in (ax1, ax2):
    a.spines[["top", "right"]].set_visible(False); a.grid(alpha=.25, lw=.5)
fig.tight_layout()
out = REPO / "audit" / "ab_matched_tokens.png"
fig.savefig(out, dpi=150)
print(f"saved -> {out}")
