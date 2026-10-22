#!/usr/bin/env python3
"""
audit_tokenizer.py — settle two open questions with measured numbers, not estimates:

  (1) TOKENS PER READ — are reads actually longer than seq_len (i.e. silently truncated)?
      Our loader logged 0 reads dropped at seq_len=64, which says NO; this confirms it and
      shows the full length distribution.

  (2) BP PER TOKEN — how efficiently does the tokenizer pack DNA? The laptop readout guessed
      ~2 bp/token (barely a 2-mer), while BPE at a 4,096 vocab is usually ~5-6 bp/token. If it
      really is ~2, every token-count estimate on the project inherits a ~3x inefficiency.

Reads a training-format file, samples reads, tokenizes them exactly as the loader does
(`<READ_START> {body} <READ_END>`), and reports the length distribution plus DNA bp-per-token.

    module use /soft/modulefiles && module load frameworks
    python audit_tokenizer.py --txt output/train.txt --n 20000
"""
from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

BASES = set("ACGT")


def dna_only(body: str) -> str:
    """Concatenate the ACGT bases of a read body, dropping special tokens like <PAIRED_END>."""
    out = []
    for tok in body.split():
        if tok.startswith("<"):
            continue
        out.append("".join(c for c in tok.upper() if c in BASES))
    return "".join(out)


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", default="output/train.txt")
    ap.add_argument("--n", type=int, default=20000, help="max reads to sample")
    ap.add_argument("--seq-len", type=int, default=64, help="cap to test truncation against")
    args = ap.parse_args()

    from prepare_genomic import Tokenizer
    from probe_sample_coherence import parse_val_txt_grouped
    tk = Tokenizer.from_directory()

    grouped = parse_val_txt_grouped(Path(args.txt), n_per_sample=10**9, seed=42)
    bodies = [b for s in grouped.values() for b in s]
    if not bodies:
        raise SystemExit(f"no reads parsed from {args.txt} (wrong format or path?)")
    if len(bodies) > args.n:
        random.seed(0)
        bodies = random.sample(bodies, args.n)

    full_tok, bp, dna_tok = [], [], []
    for body in bodies:
        full_tok.append(len(tk.encode(f"<READ_START> {body} <READ_END>")))   # as the loader tokenizes
        dna = dna_only(body)
        if dna:
            bp.append(len(dna))
            dna_tok.append(len(tk.encode(dna)))                              # DNA-only tokens (no wrappers)

    over = sum(1 for t in full_tok if t > args.seq_len)
    total_bp, total_dt = sum(bp), sum(dna_tok)

    print(f"reads sampled: {len(full_tok):,} from {args.txt}")
    print("\n== (1) tokens per read  (wrapped: <READ_START> body <READ_END>) ==")
    print(f"  mean={statistics.mean(full_tok):.1f}  median={statistics.median(full_tok)}  "
          f"p95={_pct(full_tok, 95)}  max={max(full_tok)}")
    print(f"  reads > seq_len={args.seq_len}: {over:,} ({100*over/len(full_tok):.2f}%)  ->  "
          f"{'TRUNCATION is happening' if over else 'NO truncation at this cap'}")

    print("\n== (2) bp per token  (tokenizer DNA compression) ==")
    print(f"  mean bases per read   = {statistics.mean(bp):.1f}")
    print(f"  mean DNA tokens/read  = {statistics.mean(dna_tok):.1f}")
    print(f"  bp per token (pooled) = {total_bp/total_dt:.2f}   "
          f"(BPE-4096 typical ~5-6; the laptop estimate was ~2)")
    verdict = ("looks INEFFICIENT (near 2-mer) — worth re-examining the tokenizer"
               if total_bp/total_dt < 3.5 else "looks healthy — no 3x inefficiency to recover")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
