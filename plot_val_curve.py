#!/usr/bin/env python3
"""
plot_val_curve.py — convergence figure for a HARMONY MLM run.

Reads a run's probe_trajectory.csv and writes val_curve.png next to it. Two stacked
panels sharing the x-axis (training step):

  (1) Fill-in-the-blank accuracy  — val vs train top-1   (the headline metric)
  (2) Cross-entropy loss          — val vs train         (lower = better)

Why both curves on each panel: the gap between val and train IS the science. Val ABOVE
train = generalizing (the biomarker replicates in the held-out cohort). Val dipping
BELOW train = overfitting (memorizing the discovery cohort). The plot makes that
divergence obvious without reading a single number.

Tufte style: no chart junk, no legend box (curves are labeled directly at their right
end), light axes, the best val point marked. Safe to run mid-training — it just plots
whatever rows exist so far.

Usage:
    python plot_val_curve.py                                  # defaults to experiments/ddp_depth6
    python plot_val_curve.py experiments/ddp_depth6           # a run dir (finds probe_trajectory.csv)
    python plot_val_curve.py path/to/probe_trajectory.csv     # an explicit csv
    python plot_val_curve.py <dir-or-csv> -o some/where.png   # custom output path
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless (Aurora compute nodes have no display)
import matplotlib.pyplot as plt


INK = "#2b2b2b"       # near-black for text/axes
VAL = "#0E7C86"       # teal — validation (held-out cohort)
TRAIN = "#B0651A"     # ochre — train (discovery cohort)
BEST = "#0E7C86"      # marker for best val point


def _resolve_csv(target: str) -> Path:
    """Accept a run dir OR a direct csv path; return the csv Path."""
    p = Path(target)
    if p.is_dir():
        p = p / "probe_trajectory.csv"
    if not p.exists():
        sys.exit(f"[plot_val_curve] no trajectory csv at: {p}\n"
                 f"  (pass a run dir containing probe_trajectory.csv, or the csv itself)")
    return p


def _load(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"[plot_val_curve] {csv_path} has a header but no data rows yet "
                 f"(no eval has run — check back after the first --eval-every steps).")
    return rows


def _col(rows: list[dict], name: str) -> list[float]:
    """Pull a numeric column; tolerate missing/blank cells (partial mid-run writes)."""
    out = []
    for r in rows:
        v = r.get(name, "")
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def _label_end(ax, x, y, text, color):
    """Tufte-style direct label at the right end of a curve (no legend box)."""
    # find last finite point
    for xi, yi in zip(reversed(x), reversed(y)):
        if yi == yi:  # not NaN
            ax.annotate(text, xy=(xi, yi), xytext=(6, 0), textcoords="offset points",
                        va="center", ha="left", color=color, fontsize=9, fontweight="bold")
            return


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot a HARMONY run's val/train convergence curve.")
    ap.add_argument("target", nargs="?", default="experiments/ddp_depth6",
                    help="run dir or probe_trajectory.csv (default: experiments/ddp_depth6)")
    ap.add_argument("-o", "--out", default=None, help="output png (default: val_curve.png next to the csv)")
    ap.add_argument("--logx", action="store_true",
                    help="log-scale the x-axis (training step) — distinguishes a true plateau "
                         "(curve flattens) from slow log-linear improvement (curve stays a rising line)")
    args = ap.parse_args()

    csv_path = _resolve_csv(args.target)
    rows = _load(csv_path)
    default_name = "val_curve_logx.png" if args.logx else "val_curve.png"
    out_path = Path(args.out) if args.out else csv_path.with_name(default_name)

    step = _col(rows, "step")
    v_top1, t_top1 = _col(rows, "val_msk_top1"), _col(rows, "train_msk_top1")
    v_ce, t_ce = _col(rows, "val_msk_ce"), _col(rows, "train_msk_ce")

    # best val top-1 (highest) for the marker/annotation
    best_i = max((i for i in range(len(v_top1)) if v_top1[i] == v_top1[i]),
                 key=lambda i: v_top1[i], default=None)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True,
                                   gridspec_kw={"hspace": 0.12})

    # ---- Panel 1: accuracy (top-1) ----
    ax1.plot(step, t_top1, color=TRAIN, lw=1.6)
    ax1.plot(step, v_top1, color=VAL, lw=2.0)
    _label_end(ax1, step, v_top1, "val", VAL)
    _label_end(ax1, step, t_top1, "train", TRAIN)
    if best_i is not None:
        bx, by = step[best_i], v_top1[best_i]
        ax1.scatter([bx], [by], s=36, color=BEST, zorder=5)
        # place the callout BELOW-LEFT of the point so it never collides with the title
        # (the best point often sits top-right) or with the direct "val" end-label.
        ax1.annotate(f"best val {by:.3f} @ step {int(bx):,}", xy=(bx, by),
                     xytext=(-8, -14), textcoords="offset points", ha="right", va="top",
                     fontsize=8.5, color=INK)
    ax1.set_ylabel("fill-in-the-blank\ntop-1 accuracy", color=INK, fontsize=10)
    ax1.set_title("HARMONY MLM convergence — validation (held-out) vs train (discovery)",
                  color=INK, fontsize=11, loc="left", pad=10)

    # ---- Panel 2: loss (CE) ----
    ax2.plot(step, t_ce, color=TRAIN, lw=1.6)
    ax2.plot(step, v_ce, color=VAL, lw=2.0)
    _label_end(ax2, step, v_ce, "val", VAL)
    _label_end(ax2, step, t_ce, "train", TRAIN)
    ax2.set_ylabel("cross-entropy\nloss (nats)", color=INK, fontsize=10)
    ax2.set_xlabel("training step" + (" — log scale" if args.logx else ""), color=INK, fontsize=10)

    # smallest positive step, for a sane left bound on log scale (log(0) is undefined)
    pos_steps = [s for s in step if s == s and s > 0]
    xmin = min(pos_steps) if pos_steps else 1.0
    xmax = max(pos_steps) if pos_steps else 1.0

    # ---- Tufte de-junking on both panels ----
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(INK)
        ax.spines["bottom"].set_color(INK)
        ax.tick_params(colors=INK, labelsize=9)
        if args.logx:
            ax.set_xscale("log")
            # multiplicative padding on a log axis (additive would be wrong); right pad
            # leaves room for the direct "val"/"train" end-labels.
            ax.set_xlim(xmin * 0.85, xmax * 1.35)
        else:
            ax.margins(x=0.02)
            x0, x1 = ax.get_xlim()
            ax.set_xlim(x0, x1 + 0.08 * (x1 - x0))  # right pad for the end-labels

    fig.text(0.008, 0.008,
             f"source: {csv_path.name}  ·  {len(rows)} eval points  ·  "
             f"val above train = generalizing; val below train = overfitting",
             fontsize=7, color="#8a8a8a")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[plot_val_curve] wrote {out_path}  ({len(rows)} eval points, "
          f"through step {int(step[-1]):,})")
    if best_i is not None:
        print(f"[plot_val_curve] best val top-1 = {v_top1[best_i]:.4f} at step {int(step[best_i]):,} "
              f"(train there = {t_top1[best_i]:.4f}, gap = {v_top1[best_i]-t_top1[best_i]:+.4f})")


if __name__ == "__main__":
    main()
