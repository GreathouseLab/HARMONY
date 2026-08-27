#!/usr/bin/env python3
"""
combine_corpora.py — merge training-format corpora into one bigger train.txt for the
data-scaling experiment, while KEEPING THE ORIGINAL val.txt FROZEN.

Why frozen val: to test "does more UNIQUE training data keep val climbing the log-linear
slope?", val top-1 must be measured on the SAME held-out cohort as the 24h run. So we grow
ONLY the training pool and never touch the original val.txt.

Recommended flow for lab data:
  1) Put ALL lab reads into training (no lab val split):
       python prepare_fastq.py --input-dir lab_fastqs/ --output-dir lab_out/ --val-fraction 0
  2) Combine the original train pool + the lab train pool:
       python combine_corpora.py --out output/train_plus_lab.txt \
              output/train.txt lab_out/train.txt
  3) Train, pointing at the bigger train and the ORIGINAL (unchanged) val:
       ... train_mlm.py --train-txt output/train_plus_lab.txt --val-txt output/val.txt ...

The combined file is written by streaming (safe for multi-GB inputs). Sample/read counts
per input and the total data multiplier vs the 630K baseline are reported.
"""
from __future__ import annotations

import argparse
from pathlib import Path

BASELINE_READS = 630_000  # the original corpus, for the "×" multiplier


def counts(path: Path) -> tuple[int, int]:
    s = r = 0
    with open(path) as f:
        for line in f:
            s += line.count("<SAMPLE_START>")
            r += line.count("<READ_START>")
    return s, r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="training-format .txt files to concatenate (base first)")
    ap.add_argument("--out", required=True, help="combined output path (e.g. output/train_plus_lab.txt)")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.resolve() in {Path(p).resolve() for p in args.inputs}:
        ap.error("--out must not be one of the inputs (would corrupt while streaming)")

    tot_s = tot_r = 0
    print(f"{'file':<46}{'samples':>10}{'reads':>14}")
    print("-" * 70)
    with open(out_path, "w") as out:
        for p in args.inputs:
            s, r = counts(Path(p))
            tot_s += s
            tot_r += r
            print(f"{p:<46}{s:>10,}{r:>14,}")
            with open(p) as f:
                for line in f:
                    out.write(line)
            out.write("\n")  # guarantee a boundary between corpora
    print("-" * 70)
    print(f"{'COMBINED → ' + str(out_path):<46}{tot_s:>10,}{tot_r:>14,}")
    print(f"\nData multiplier vs {BASELINE_READS:,} baseline reads: {tot_r / BASELINE_READS:.1f}×")
    print(f"Train with:  --train-txt {out_path}  --val-txt output/val.txt   (val FROZEN — do not add val data here)")


if __name__ == "__main__":
    main()
