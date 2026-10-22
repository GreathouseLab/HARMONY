#!/usr/bin/env python3
"""
measure_bp_per_token.py — AUDIT ONLY. Reads nothing but data; writes nothing but stdout.

Samples complete <READ_START> ... <READ_END> records from the pre-tokenization training
text, tokenizes them with the EXACT tokenizer train_mlm.py loads
(prepare_genomic.Tokenizer.from_directory -> ~/.cache/autoresearch/tokenizer/tokenizer.pkl),
and reports the token-length distribution and DNA bp/token.

Tokenization matches the loader convention exactly (paired_data_loader.py:56):
    text = f"<READ_START> {body} <READ_END>"

Sampling is by random byte offset across the whole file, so the sample is not biased
toward the head of a 6 GB corpus.

Usage:
    .venv/bin/python3.12 audit/measure_bp_per_token.py --txt output/train.txt --n 10000 --seed 42
"""
from __future__ import annotations

import argparse
import os
import random
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BASES = set("ACGTN")
RECORD_RE = re.compile(r"<READ_START>\s+(.*?)\s+<READ_END>", re.DOTALL)
# Window read at each random offset; comfortably larger than one record (~220 chars).
WINDOW = 4096


def dna_bp(body: str) -> int:
    """Count DNA bases in a record body, ignoring special tokens like <PAIRED_END>."""
    cleaned = re.sub(r"<[^>]*>", " ", body)
    return sum(1 for ch in cleaned if ch in BASES)


def sample_records(path: Path, n: int, seed: int) -> list[str]:
    """Draw n complete read bodies from random byte offsets across the file."""
    rng = random.Random(seed)
    size = os.path.getsize(path)
    out: list[str] = []
    attempts = 0
    max_attempts = n * 20
    with open(path, "r", errors="ignore") as f:
        while len(out) < n and attempts < max_attempts:
            attempts += 1
            f.seek(rng.randrange(0, max(1, size - WINDOW)))
            window = f.read(WINDOW)
            # Drop the leading partial record, then take the first COMPLETE one.
            m = RECORD_RE.search(window)
            if m:
                out.append(m.group(1))
    if len(out) < n:
        raise RuntimeError(f"only sampled {len(out)}/{n} records after {attempts} attempts")
    return out


def pct(sorted_vals: list[int], p: float) -> float:
    """Nearest-rank percentile on an already-sorted list."""
    if not sorted_vals:
        return float("nan")
    k = max(0, min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", default="output/train.txt")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    path = Path(args.txt)
    if not path.is_absolute():
        path = REPO / path

    from prepare_genomic import Tokenizer, TOKENIZER_DIR

    tk = Tokenizer.from_directory()
    vocab = tk.get_vocab_size()

    bodies = sample_records(path, args.n, args.seed)

    tok_lens: list[int] = []
    bp_lens: list[int] = []
    for body in bodies:
        ids = tk.encode(f"<READ_START> {body} <READ_END>")
        tok_lens.append(len(ids))
        bp_lens.append(dna_bp(body))

    s = sorted(tok_lens)
    total_bp = sum(bp_lens)
    total_tok = sum(tok_lens)
    # bp/token measured over DNA only: subtract the 2 wrapper tokens per record so the
    # ratio reflects sequence compression, not the special-token overhead.
    dna_tok = total_tok - 2 * len(tok_lens)

    print("=" * 72)
    print("HARMONY tokenizer audit — bp/token and tokens/read")
    print("=" * 72)
    print(f"corpus          : {path}")
    print(f"tokenizer       : {os.path.join(TOKENIZER_DIR, 'tokenizer.pkl')}")
    print(f"vocab_size      : {vocab}")
    print(f"records sampled : {len(tok_lens):,}   (seed={args.seed})")
    print()
    print("tokens per read (incl. <READ_START>/<READ_END>)")
    print(f"  mean   : {statistics.mean(tok_lens):.2f}")
    print(f"  median : {statistics.median(tok_lens):.1f}")
    print(f"  p5     : {pct(s, 5):.0f}")
    print(f"  p95    : {pct(s, 95):.0f}")
    print(f"  min/max: {s[0]} / {s[-1]}")
    print()
    print("read length (DNA bases, both mates, excl. special tokens)")
    print(f"  mean   : {statistics.mean(bp_lens):.2f} bp")
    print(f"  median : {statistics.median(bp_lens):.1f} bp")
    print()
    print("compression")
    print(f"  bp/token (DNA tokens only) : {total_bp / dna_tok:.3f}")
    print(f"  bp/token (incl. wrappers)  : {total_bp / total_tok:.3f}")
    print()
    print("truncation exposure")
    for thr in (64, 96, 128):
        over = sum(1 for t in tok_lens if t > thr)
        print(f"  reads > {thr:>3} tokens : {over:,} / {len(tok_lens):,} = {100.0 * over / len(tok_lens):.2f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
