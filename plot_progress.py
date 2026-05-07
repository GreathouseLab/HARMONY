"""
Across-runs val_bpb progress chart.

Reads experiments/results.csv (38 historical autoresearch rows) plus any
experiments/r{N}_*/stdout.txt directories (post-leak-fix R4+ convention) and
renders experiments/progress.png with kept improvements (green), discarded
runs (gray), running-best step line, and a vertical line at the 2026-04-29
paired-end leak fix boundary.

Auto-invoked by train.py at the end of every run. Safe to run standalone.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
EXP_DIR = PROJECT_DIR / "experiments"
RESULTS_CSV = EXP_DIR / "results.csv"
OUT_PNG = EXP_DIR / "progress.png"

# 2026-04-29: paired-end leak fix landed in prepare_fastq.py. Numbers before
# this date are inflated by R1/R2-from-same-molecule landing on opposite sides
# of the train/val split.
LEAK_FIX_DATE = datetime(2026, 4, 29)

# Post-leak-fix experiment dirs use the rN_ prefix (r4_, r5_, ...).
ROUND_DIR_RE = re.compile(r"^r\d+_")

VAL_BPB_RE = re.compile(r"^val_bpb:\s+([\d.]+)", re.MULTILINE)


def _parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _load_csv_rows():
    rows = []
    if not RESULTS_CSV.exists():
        return rows
    with open(RESULTS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "OK":
                continue
            try:
                vb = float(r.get("val_bpb") or "")
            except ValueError:
                continue
            rows.append({
                "name": r.get("name") or "?",
                "val_bpb": vb,
                "timestamp": _parse_ts(r.get("timestamp", "")),
            })
    return rows


def _load_round_dir_rows():
    rows = []
    if not EXP_DIR.exists():
        return rows
    for d in sorted(EXP_DIR.iterdir()):
        if not d.is_dir() or not ROUND_DIR_RE.match(d.name):
            continue
        sp = d / "stdout.txt"
        if not sp.exists():
            continue
        try:
            text = sp.read_text()
        except OSError:
            continue
        m = VAL_BPB_RE.search(text)
        if not m:
            continue
        rows.append({
            "name": d.name,
            "val_bpb": float(m.group(1)),
            "timestamp": datetime.fromtimestamp(sp.stat().st_mtime),
        })
    return rows


def main():
    csv_rows = _load_csv_rows()
    dir_rows = _load_round_dir_rows()
    # Dedupe by name; csv (richer schema) wins on collision.
    by_name = {r["name"]: r for r in dir_rows}
    for r in csv_rows:
        by_name[r["name"]] = r
    rows = list(by_name.values())
    rows.sort(key=lambda r: r["timestamp"] or datetime.min)
    if not rows:
        print("plot_progress: no rows to plot")
        return 1

    xs = list(range(len(rows)))
    ys = [r["val_bpb"] for r in rows]

    leak_idx = None
    for i, r in enumerate(rows):
        if r["timestamp"] and r["timestamp"] >= LEAK_FIX_DATE:
            leak_idx = i
            break

    # Running-best resets at the leak boundary — pre/post-leak val_bpb numbers
    # are not comparable, so a single running-best line across the boundary
    # would falsely imply post-leak runs are "regressions."
    running = []
    kept = []
    cur_pre = float("inf")
    cur_post = float("inf")
    for i, v in enumerate(ys):
        if leak_idx is not None and i >= leak_idx:
            if v < cur_post:
                cur_post = v
                kept.append(True)
            else:
                kept.append(False)
            running.append(cur_post)
        else:
            if v < cur_pre:
                cur_pre = v
                kept.append(True)
            else:
                kept.append(False)
            running.append(cur_pre)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    dx = [x for x, k in zip(xs, kept) if not k]
    dy = [y for y, k in zip(ys, kept) if not k]
    kx = [x for x, k in zip(xs, kept) if k]
    ky = [y for y, k in zip(ys, kept) if k]

    ax.scatter(dx, dy, c="lightgray", s=22, label="Discarded", zorder=2)
    if leak_idx is not None and 0 < leak_idx < len(xs):
        ax.step(xs[:leak_idx], running[:leak_idx], where="post",
                color="tab:green", alpha=0.7, linewidth=1.5,
                label="Running best", zorder=3)
        ax.step(xs[leak_idx:], running[leak_idx:], where="post",
                color="tab:green", alpha=0.7, linewidth=1.5, zorder=3)
    else:
        ax.step(xs, running, where="post", color="tab:green", alpha=0.7,
                linewidth=1.5, label="Running best", zorder=3)
    ax.scatter(kx, ky, c="tab:green", s=80, edgecolors="white",
               linewidths=0.6, label="Kept", zorder=4)

    if leak_idx is not None:
        ax.axvline(leak_idx - 0.5, color="tab:red", linestyle="--",
                   alpha=0.7, linewidth=1.2,
                   label=f"Paired-end leak fix ({LEAK_FIX_DATE.date()})")

    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Validation BPB (lower is better)")
    ax.set_title(
        f"Autoresearch Progress: {len(rows)} Experiments, "
        f"{sum(kept)} Kept Improvements"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"progress_plot:    {OUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
