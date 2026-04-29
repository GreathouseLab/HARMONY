"""Tests for prepare_fastq.py — sample-stem grouping and paired-end concatenation."""

import sys
from pathlib import Path

# Make the repo root importable regardless of where pytest is run from
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from prepare_fastq import (
    extract_sample_stem,
    reverse_complement,
    group_files_by_sample,
    split_samples_by_stem,
    sample_to_text,
    PAIRED_END,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_fastq(path: Path, sequences: list[str]) -> None:
    """Write a minimal FASTQ file with dummy headers and quality strings."""
    with open(path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f"@read_{i}\n{seq}\n+\n{'I' * len(seq)}\n")


# ---------------------------------------------------------------------------
# extract_sample_stem
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected", [
    ("SRR6915093_1.fastq.gz",        ("SRR6915093", "1")),
    ("SRR6915093_2.fastq.gz",        ("SRR6915093", "2")),
    ("SampleA_R1.fastq",             ("SampleA",    "1")),
    ("SampleA_R2.fastq",             ("SampleA",    "2")),
    ("SampleA_R1_001.fastq.gz",      ("SampleA",    "1")),
    ("SampleA_L001_R1_001.fastq.gz", ("SampleA",    "1")),
    ("SampleB.fastq",                ("SampleB",    None)),
])
def test_extract_sample_stem_handles_all_conventions(filename, expected):
    assert extract_sample_stem(Path(filename)) == expected


# ---------------------------------------------------------------------------
# reverse_complement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq, expected", [
    ("ACGT", "ACGT"),
    ("AAAA", "TTTT"),
    ("ACGTN", "NACGT"),
    ("acgt", "acgt"),
])
def test_reverse_complement(seq, expected):
    assert reverse_complement(seq) == expected


# ---------------------------------------------------------------------------
# group_files_by_sample
# ---------------------------------------------------------------------------

def test_grouping_keeps_pair_together(tmp_path):
    write_fastq(tmp_path / "SampleA_1.fastq", ["ACGT"])
    write_fastq(tmp_path / "SampleA_2.fastq", ["TTTT"])
    write_fastq(tmp_path / "SampleB_R1.fastq", ["GGGG"])
    write_fastq(tmp_path / "SampleB_R2.fastq", ["CCCC"])
    files = sorted(tmp_path.glob("*.fastq"))
    groups = group_files_by_sample(files)
    assert set(groups.keys()) == {"SampleA", "SampleB"}
    assert len(groups["SampleA"]["r1"]) == 1
    assert len(groups["SampleA"]["r2"]) == 1
    assert len(groups["SampleB"]["r1"]) == 1
    assert len(groups["SampleB"]["r2"]) == 1


# ---------------------------------------------------------------------------
# split_samples_by_stem
# ---------------------------------------------------------------------------

def test_split_keeps_all_files_of_one_sample_on_one_side(tmp_path):
    for i in range(10):
        write_fastq(tmp_path / f"S{i:02d}_R1.fastq", ["ACGT"])
        write_fastq(tmp_path / f"S{i:02d}_R2.fastq", ["TTTT"])
    files = sorted(tmp_path.glob("*.fastq"))
    groups = group_files_by_sample(files)
    train_stems, val_stems = split_samples_by_stem(groups, val_fraction=0.2, seed=42)
    assert set(train_stems).isdisjoint(set(val_stems))
    assert set(train_stems) | set(val_stems) == set(groups.keys())
    assert len(val_stems) == 2


# ---------------------------------------------------------------------------
# sample_to_text — paired-end
# ---------------------------------------------------------------------------

def test_paired_end_emits_one_read_block_per_molecule(tmp_path):
    write_fastq(tmp_path / "X_1.fastq", ["AAAA", "AGGT", "CGTA"])
    write_fastq(tmp_path / "X_2.fastq", ["GGGG", "TGCA", "AAAT"])
    files = sorted(tmp_path.glob("*.fastq"))
    groups = group_files_by_sample(files)
    text, n = sample_to_text("X", groups["X"])
    assert n == 3
    assert text.count("<READ_START>") == 3
    assert text.count("<READ_END>") == 3
    assert text.count(PAIRED_END) == 3
    assert "AAAA <PAIRED_END> CCCC" in text  # rc(GGGG)
    assert "AGGT <PAIRED_END> TGCA" in text  # rc(TGCA)
    assert "CGTA <PAIRED_END> ATTT" in text  # rc(AAAT)


def test_paired_concatenation_uses_revcomp(tmp_path):
    write_fastq(tmp_path / "X_1.fastq", ["AAAA"])
    write_fastq(tmp_path / "X_2.fastq", ["GGGG"])
    files = sorted(tmp_path.glob("*.fastq"))
    groups = group_files_by_sample(files)
    text, n = sample_to_text("X", groups["X"])
    # revcomp("GGGG") = "CCCC"
    assert "AAAA <PAIRED_END> CCCC" in text
    assert n == 1


def test_paired_emits_warning_on_truncation(tmp_path, capsys):
    """Read-count mismatch within a lane is recoverable: warn and proceed."""
    write_fastq(tmp_path / "X_1.fastq", ["AAAA", "GGGG", "CCCC"])
    write_fastq(tmp_path / "X_2.fastq", ["TTTT"])  # only 1 of 3
    files = sorted(tmp_path.glob("*.fastq"))
    groups = group_files_by_sample(files)
    _, n = sample_to_text("X", groups["X"])
    captured = capsys.readouterr()
    assert n == 1
    assert "mismatched" in captured.err


def test_paired_end_lane_split_must_match_count(tmp_path):
    """Lane-count mismatch is structural: fail loudly rather than misalign."""
    write_fastq(tmp_path / "X_L001_R1_001.fastq", ["AAAA"])
    write_fastq(tmp_path / "X_L002_R1_001.fastq", ["GGGG"])
    write_fastq(tmp_path / "X_L001_R2_001.fastq", ["TTTT"])
    files = sorted(tmp_path.glob("*.fastq"))
    groups = group_files_by_sample(files)
    with pytest.raises(ValueError, match="lane-split"):
        sample_to_text("X", groups["X"])


# ---------------------------------------------------------------------------
# sample_to_text — single-end
# ---------------------------------------------------------------------------

def test_single_end_unchanged(tmp_path):
    write_fastq(tmp_path / "Y.fastq", ["ACGT"])
    files = sorted(tmp_path.glob("*.fastq"))
    groups = group_files_by_sample(files)
    text, n = sample_to_text("Y", groups["Y"])
    assert PAIRED_END not in text
    assert "<READ_START> ACGT <READ_END>" in text
    assert n == 1


# ---------------------------------------------------------------------------
# sample_to_text — orphans
# ---------------------------------------------------------------------------

def test_orphan_r2_is_skipped(tmp_path, capsys):
    write_fastq(tmp_path / "Z_2.fastq", ["GGGG"])
    files = sorted(tmp_path.glob("*.fastq"))
    groups = group_files_by_sample(files)
    text, n = sample_to_text("Z", groups["Z"])
    captured = capsys.readouterr()
    assert text == ""
    assert n == 0
    assert "only R2" in captured.err
