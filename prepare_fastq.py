"""
FASTQ → plain text preprocessor for BPE tokenization.

Reads demultiplexed FASTQ files (paired-end or single-end), groups them by
**sample stem** so paired files always stay together, and outputs a single
concatenated text stream split 90/10 by sample.

Usage:
    python prepare_fastq.py --input-dir /path/to/fastq_dir --output-dir /path/to/output

Input layout (paired-end Illumina, single-end, or mixed):
    fastq_dir/
        SRR6915093_1.fastq.gz       # paired-end mate 1
        SRR6915093_2.fastq.gz       # paired-end mate 2
        SampleA_R1_001.fastq.gz     # paired-end Illumina BCL2FASTQ
        SampleA_R2_001.fastq.gz
        SampleB.fastq               # single-end
        ...

Output:
    output/
        train.txt          # concatenated text stream for training
        val.txt            # concatenated text stream for validation
        manifest.txt       # which sample stems went where (PE / SE)

Per-sample text format
    Paired-end (one read block per molecule, R1 + reverse_complement(R2)):
        <SAMPLE_START> <READ_START> <r1_seq> <PAIRED_END> <revcomp_r2_seq> <READ_END> ... <SAMPLE_END>

    Single-end (unchanged from prior pipeline):
        <SAMPLE_START> <READ_START> <seq> <READ_END> ... <SAMPLE_END>

Quality scores are dropped (sequence only).

Important: the train/val split is over **sample stems**, not file paths. This
prevents R1 of a sample landing in train while R2 lands in val (the paired-end
leakage discovered 2026-04-27 in `prepare_fastq.py`'s previous version).
"""

import argparse
import gzip
import os
import random
import re
import sys
from itertools import zip_longest
from pathlib import Path


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

SAMPLE_START = "<SAMPLE_START>"
SAMPLE_END = "<SAMPLE_END>"
READ_START = "<READ_START>"
READ_END = "<READ_END>"
PAIRED_END = "<PAIRED_END>"

SPECIAL_TOKENS = [SAMPLE_START, SAMPLE_END, READ_START, READ_END, PAIRED_END]


# ---------------------------------------------------------------------------
# FASTQ parsing
# ---------------------------------------------------------------------------

def open_fastq(filepath):
    """Open a FASTQ file, handling gzip transparently by checking magic bytes."""
    with open(filepath, "rb") as probe:
        magic = probe.read(2)
    if magic == b"\x1f\x8b":  # gzip magic
        return gzip.open(filepath, "rt")
    return open(filepath, "r")


def read_fastq_sequences(filepath):
    """Yield sequence strings from a FASTQ file (4 lines per record; quality dropped)."""
    with open_fastq(filepath) as f:
        while True:
            header = f.readline().strip()
            if not header:
                break
            if not header.startswith("@"):
                raise ValueError(f"Expected FASTQ header starting with '@', got: {header[:50]}")
            sequence = f.readline().strip()
            f.readline()  # plus line
            f.readline()  # quality (dropped)
            if not sequence:
                continue
            yield sequence


# ---------------------------------------------------------------------------
# Sample-stem grouping
# ---------------------------------------------------------------------------

# Match the suffix portion of paired-end FASTQ filenames.
# Supports: _1.fastq, _2.fastq, _R1.fastq, _R2.fastq,
#           _R1_001.fastq, _R2_001.fastq (Illumina BCL2FASTQ),
#           with optional .gz, optional lane prefix _L001 etc.
PAIR_SUFFIX_RE = re.compile(
    r"(?:_L\d{3})?_(?:R)?([12])(?:_\d{3})?\.(?:fastq|fq)(?:\.gz)?$",
    re.IGNORECASE,
)


def extract_sample_stem(filepath):
    """
    Return (sample_stem, mate) for a FASTQ file.
    mate is "1", "2", or None for single-end files.

    Examples:
        SRR6915093_1.fastq.gz       -> ("SRR6915093", "1")
        SRR6915093_2.fastq.gz       -> ("SRR6915093", "2")
        SampleA_R1_001.fastq.gz     -> ("SampleA", "1")
        SampleA_L001_R2_001.fastq   -> ("SampleA", "2")
        SampleB.fastq               -> ("SampleB", None)
    """
    name = filepath.name
    m = PAIR_SUFFIX_RE.search(name)
    if m:
        stem = name[:m.start()]
        return stem, m.group(1)
    # Single-end: strip the extension only.
    stem = re.sub(r"\.(fastq|fq)(\.gz)?$", "", name, flags=re.IGNORECASE)
    return stem, None


def group_files_by_sample(filepaths):
    """Group FASTQ files by sample stem. Returns {stem: {"r1": [...], "r2": [...], "single": [...]}}.

    Multiple R1 / R2 files per stem (Illumina lane splits) are kept in sorted order.
    """
    groups = {}
    for fp in filepaths:
        stem, mate = extract_sample_stem(fp)
        if stem not in groups:
            groups[stem] = {"r1": [], "r2": [], "single": []}
        if mate == "1":
            groups[stem]["r1"].append(fp)
        elif mate == "2":
            groups[stem]["r2"].append(fp)
        else:
            groups[stem]["single"].append(fp)
    for g in groups.values():
        for key in g:
            g[key].sort()
    return groups


# ---------------------------------------------------------------------------
# Reverse complement
# ---------------------------------------------------------------------------

_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq):
    """Reverse-complement a DNA sequence. Preserves N and case."""
    return seq.translate(_RC_TABLE)[::-1]


# ---------------------------------------------------------------------------
# Train/val split — over sample stems, not file paths
# ---------------------------------------------------------------------------

def split_samples_by_stem(groups, val_fraction=0.1, seed=42):
    """Return (train_stems, val_stems). All files belonging to a stem stay together."""
    stems = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(stems)
    n_val = max(1, int(len(stems) * val_fraction))
    val_stems = stems[:n_val]
    train_stems = stems[n_val:]
    return train_stems, val_stems


# ---------------------------------------------------------------------------
# Sample → text (paired-aware)
# ---------------------------------------------------------------------------

def sample_to_text(stem, group):
    """
    Build the text block for one sample. Returns (text, num_reads_emitted).

    Paired-end: emit one <READ_START> ... <READ_END> per molecule, where the
    body is "<r1_seq> <PAIRED_END> <revcomp(r2_seq)>".

    Single-end: emit one <READ_START> ... <READ_END> per read.

    Lane-split samples (multiple R1/R2 files): iterate them in sorted order,
    aligning by position within each lane file.
    """
    parts = [SAMPLE_START]
    num_reads = 0

    if group["r1"] and group["r2"]:
        if len(group["r1"]) != len(group["r2"]):
            raise ValueError(
                f"Sample {stem}: R1 has {len(group['r1'])} files but R2 has {len(group['r2'])}. "
                f"Cannot align lane-split paired-end."
            )
        for r1_path, r2_path in zip(group["r1"], group["r2"]):
            r1_iter = read_fastq_sequences(r1_path)
            r2_iter = read_fastq_sequences(r2_path)
            r1_extras = 0
            r2_extras = 0
            sentinel = object()
            for r1_seq, r2_seq in zip_longest(r1_iter, r2_iter, fillvalue=sentinel):
                if r1_seq is sentinel:
                    r2_extras += 1
                    continue
                if r2_seq is sentinel:
                    r1_extras += 1
                    continue
                molecule = f"{r1_seq} {PAIRED_END} {reverse_complement(r2_seq)}"
                parts.append(f"{READ_START} {molecule} {READ_END}")
                num_reads += 1
            if r1_extras or r2_extras:
                print(
                    f"  WARNING: {stem} {r1_path.name}/{r2_path.name} mismatched: "
                    f"{r1_extras} extra R1, {r2_extras} extra R2 (dropped)",
                    file=sys.stderr,
                )

    elif group["r1"] or group["single"]:
        files = group["r1"] or group["single"]
        for fp in files:
            for seq in read_fastq_sequences(fp):
                parts.append(f"{READ_START} {seq} {READ_END}")
                num_reads += 1

    else:
        # R2 without R1 — orphaned, skip with warning.
        print(f"  WARNING: {stem} has only R2 files, no R1. Skipping.", file=sys.stderr)
        return "", 0

    if num_reads == 0:
        print(f"  WARNING: no reads emitted for sample {stem}", file=sys.stderr)
        return "", 0

    parts.append(SAMPLE_END)
    return " ".join(parts), num_reads


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
# Write text stream
# ---------------------------------------------------------------------------

def write_stream_by_stem(stems, groups, output_path):
    """Process a list of sample stems and write the concatenated text stream."""
    total_reads = 0
    total_samples = 0

    with open(output_path, "w") as out:
        for i, stem in enumerate(stems):
            text, read_count = sample_to_text(stem, groups[stem])
            if not text:
                continue
            if total_samples > 0:
                out.write("\n")
            out.write(text)
            total_reads += read_count
            total_samples += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(stems):
                print(f"  Processed {i + 1}/{len(stems)} samples "
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
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing FASTQ files (paired-end or single-end).")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for train.txt and val.txt.")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of *samples* (not files) for validation (default: 0.1).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split (default: 42).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover files and group by sample stem
    fastq_files = find_fastq_files(input_dir)
    if not fastq_files:
        print(f"ERROR: no FASTQ files found in {input_dir}", file=sys.stderr)
        print("  Looked for: .fastq, .fq, .fastq.gz, .fq.gz", file=sys.stderr)
        sys.exit(1)
    groups = group_files_by_sample(fastq_files)
    print(f"Found {len(fastq_files)} FASTQ files in {len(groups)} samples")

    # Diagnostic: sample-type breakdown
    n_paired = sum(1 for g in groups.values() if g["r1"] and g["r2"])
    n_single = sum(1 for g in groups.values() if (g["r1"] or g["single"]) and not g["r2"])
    n_orphan = sum(1 for g in groups.values() if g["r2"] and not g["r1"])
    print(f"  Paired-end samples: {n_paired}")
    print(f"  Single-end samples: {n_single}")
    print(f"  Orphaned R2 (skipped): {n_orphan}")

    # Split by stem, not by file
    train_stems, val_stems = split_samples_by_stem(groups, args.val_fraction, args.seed)
    print(f"Split: {len(train_stems)} train / {len(val_stems)} val samples")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Writing train.txt ...")
    train_stats = write_stream_by_stem(train_stems, groups, output_dir / "train.txt")
    print(f"  → {train_stats['num_samples']} samples, "
          f"{train_stats['num_reads']:,} reads, "
          f"{train_stats['file_size_mb']:.1f} MB")
    print()

    print("Writing val.txt ...")
    val_stats = write_stream_by_stem(val_stems, groups, output_dir / "val.txt")
    print(f"  → {val_stats['num_samples']} samples, "
          f"{val_stats['num_reads']:,} reads, "
          f"{val_stats['file_size_mb']:.1f} MB")
    print()

    # Manifest — sample stems, not filenames; PE/SE annotation
    manifest_path = output_dir / "manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("# Train/val split manifest\n")
        f.write(f"# seed={args.seed} val_fraction={args.val_fraction}\n")
        f.write(f"# Special tokens: {' '.join(SPECIAL_TOKENS)}\n")
        f.write("# Pair handling: R1 + <PAIRED_END> + reverse_complement(R2)\n")
        f.write("# Split unit: SAMPLE STEM (not file path)\n\n")
        f.write("## TRAIN\n")
        for stem in sorted(train_stems):
            kind = "PE" if (groups[stem]["r1"] and groups[stem]["r2"]) else "SE"
            f.write(f"{stem}\t{kind}\n")
        f.write("\n## VAL\n")
        for stem in sorted(val_stems):
            kind = "PE" if (groups[stem]["r1"] and groups[stem]["r2"]) else "SE"
            f.write(f"{stem}\t{kind}\n")
    print(f"Manifest written to {manifest_path}")

    print()
    print("=" * 60)
    print("DONE")
    print(f"  Train: {train_stats['num_reads']:,} reads from {train_stats['num_samples']} samples ({train_stats['file_size_mb']:.1f} MB)")
    print(f"  Val:   {val_stats['num_reads']:,} reads from {val_stats['num_samples']} samples ({val_stats['file_size_mb']:.1f} MB)")
    print(f"  Output: {output_dir}")
    print()
    print("Output format examples:")
    print(f"  PE: {SAMPLE_START} {READ_START} ACGT... {PAIRED_END} TGCA... {READ_END} ... {SAMPLE_END}")
    print(f"  SE: {SAMPLE_START} {READ_START} ACGTACGT... {READ_END} ... {SAMPLE_END}")
    print()
    print("NOTE: BPE tokenizer must be retrained on this stream so <PAIRED_END> is in vocab.")


if __name__ == "__main__":
    main()
