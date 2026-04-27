"""
FASTQ → plain text preprocessor for BPE tokenization.

Reads demultiplexed FASTQ files (one file per sample), extracts sequences,
wraps them with special tokens, and outputs a single concatenated text stream
split into train/validation sets (90/10 by sample).

Usage:
    python prepare_fastq.py --input-dir /path/to/fastq_dir --output-dir /path/to/output

Input layout (one FASTQ per sample):
    fastq_dir/
        sample_001.fastq
        sample_002.fastq.gz
        ...

Output:
    output/
        train.txt          # concatenated text stream for training
        val.txt            # concatenated text stream for validation
        manifest.txt       # which samples went where

Special tokens inserted:
    <SAMPLE_START>  ... reads ...  <SAMPLE_END>
    <READ_START> ACGTACGT... <READ_END>

Quality scores are dropped (sequence only).
Split is 90/10 by SAMPLE to prevent data leakage between train and val.
"""

import argparse
import gzip
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

SAMPLE_START = "<SAMPLE_START>"
SAMPLE_END = "<SAMPLE_END>"
READ_START = "<READ_START>"
READ_END = "<READ_END>"

SPECIAL_TOKENS = [SAMPLE_START, SAMPLE_END, READ_START, READ_END]


# ---------------------------------------------------------------------------
# FASTQ parsing
# ---------------------------------------------------------------------------

def open_fastq(filepath):
    """Open a FASTQ file, handling gzip transparently by checking magic bytes."""
    with open(filepath, "rb") as probe:
        magic = probe.read(2)
    if magic == b"\x1f\x8b":  # actual gzip magic bytes
        return gzip.open(filepath, "rt")
    return open(filepath, "r")


def read_fastq_sequences(filepath):
    """
    Yield sequence strings from a FASTQ file.

    FASTQ format (4 lines per record):
        Line 1: @header
        Line 2: sequence
        Line 3: +
        Line 4: quality scores (dropped)
    """
    with open_fastq(filepath) as f:
        while True:
            header = f.readline().strip()
            if not header:
                break
            if not header.startswith("@"):
                raise ValueError(f"Expected FASTQ header starting with '@', got: {header[:50]}")
            sequence = f.readline().strip()
            plus_line = f.readline().strip()
            quality = f.readline().strip()  # noqa: F841 — dropped intentionally

            if not sequence:
                continue
            yield sequence


# ---------------------------------------------------------------------------
# Sample → text stream conversion
# ---------------------------------------------------------------------------

def sample_to_text(filepath):
    """
    Convert one FASTQ file (one sample) into a token-annotated text block.

    Returns:
        str: <SAMPLE_START> <READ_START>ACGT...<READ_END> ... <SAMPLE_END>
    """
    parts = [SAMPLE_START]
    read_count = 0
    for seq in read_fastq_sequences(filepath):
        parts.append(f" {READ_START} {seq} {READ_END}")
        read_count += 1
    parts.append(f" {SAMPLE_END}")

    if read_count == 0:
        print(f"  WARNING: no reads in {filepath}", file=sys.stderr)
        return None, 0

    return "".join(parts), read_count


# ---------------------------------------------------------------------------
# Discover FASTQ files
# ---------------------------------------------------------------------------

def find_fastq_files(input_dir):
    """Find all FASTQ files in a directory (non-recursive)."""
    extensions = (".fastq", ".fq", ".fastq.gz", ".fq.gz")
    files = []
    for f in sorted(Path(input_dir).iterdir()):
        if f.is_file() and any(str(f).endswith(ext) for ext in extensions):
            files.append(f)
    return files


# ---------------------------------------------------------------------------
# Train/val split by sample
# ---------------------------------------------------------------------------

def split_samples(fastq_files, val_fraction=0.1, seed=42):
    """
    Split FASTQ files into train and val sets BY SAMPLE.

    Uses deterministic shuffle so splits are reproducible.
    """
    import random
    rng = random.Random(seed)

    indices = list(range(len(fastq_files)))
    rng.shuffle(indices)

    n_val = max(1, int(len(fastq_files) * val_fraction))
    val_indices = set(indices[:n_val])

    train_files = [fastq_files[i] for i in range(len(fastq_files)) if i not in val_indices]
    val_files = [fastq_files[i] for i in range(len(fastq_files)) if i in val_indices]

    return train_files, val_files


# ---------------------------------------------------------------------------
# Write text stream
# ---------------------------------------------------------------------------

def write_stream(fastq_files, output_path):
    """
    Process a list of FASTQ files and write concatenated text stream.

    Returns:
        dict with stats (num_samples, num_reads, num_chars)
    """
    total_reads = 0
    total_samples = 0

    with open(output_path, "w") as out:
        for i, fpath in enumerate(fastq_files):
            text, read_count = sample_to_text(fpath)
            if text is None:
                continue
            if total_samples > 0:
                out.write("\n")
            out.write(text)
            total_reads += read_count
            total_samples += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(fastq_files):
                print(f"  Processed {i + 1}/{len(fastq_files)} samples "
                      f"({total_reads:,} reads so far)")

    file_size = os.path.getsize(output_path)
    return {
        "num_samples": total_samples,
        "num_reads": total_reads,
        "file_size_mb": file_size / (1024 * 1024),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess FASTQ files into a plain text stream for BPE tokenization."
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing FASTQ files (one per sample)."
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for train.txt and val.txt."
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction of samples for validation (default: 0.1)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split (default: 42)."
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover FASTQ files
    fastq_files = find_fastq_files(input_dir)
    if not fastq_files:
        print(f"ERROR: no FASTQ files found in {input_dir}", file=sys.stderr)
        print("  Looked for: .fastq, .fq, .fastq.gz, .fq.gz", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(fastq_files)} FASTQ files in {input_dir}")

    # Split by sample
    train_files, val_files = split_samples(fastq_files, args.val_fraction, args.seed)
    print(f"Split: {len(train_files)} train / {len(val_files)} val samples")
    print()

    # Create output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write train stream
    print("Writing train.txt ...")
    train_stats = write_stream(train_files, output_dir / "train.txt")
    print(f"  → {train_stats['num_samples']} samples, "
          f"{train_stats['num_reads']:,} reads, "
          f"{train_stats['file_size_mb']:.1f} MB")
    print()

    # Write val stream
    print("Writing val.txt ...")
    val_stats = write_stream(val_files, output_dir / "val.txt")
    print(f"  → {val_stats['num_samples']} samples, "
          f"{val_stats['num_reads']:,} reads, "
          f"{val_stats['file_size_mb']:.1f} MB")
    print()

    # Write manifest
    manifest_path = output_dir / "manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("# Train/val split manifest\n")
        f.write(f"# seed={args.seed} val_fraction={args.val_fraction}\n")
        f.write(f"# Special tokens: {' '.join(SPECIAL_TOKENS)}\n\n")
        f.write("## TRAIN\n")
        for fp in train_files:
            f.write(f"{fp.name}\n")
        f.write("\n## VAL\n")
        for fp in val_files:
            f.write(f"{fp.name}\n")
    print(f"Manifest written to {manifest_path}")

    # Summary
    print()
    print("=" * 60)
    print("DONE")
    print(f"  Train: {train_stats['num_reads']:,} reads from {train_stats['num_samples']} samples ({train_stats['file_size_mb']:.1f} MB)")
    print(f"  Val:   {val_stats['num_reads']:,} reads from {val_stats['num_samples']} samples ({val_stats['file_size_mb']:.1f} MB)")
    print(f"  Output: {output_dir}")
    print()
    print("Output format example:")
    print(f"  {SAMPLE_START} {READ_START} ACGTACGT... {READ_END} {READ_START} TGCA... {READ_END} {SAMPLE_END}")
    print()
    print("Ready for BPE tokenization.")


if __name__ == "__main__":
    main()
