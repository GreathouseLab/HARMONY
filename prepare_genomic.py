"""
Train a BPE tokenizer on genomic text and prepare data for train.py.

Reads the plain-text output from prepare_fastq.py (output/train.txt, output/val.txt),
trains a BPE tokenizer with genomic special tokens, and saves everything in the
format that train.py expects.

Usage:
    python prepare_genomic.py                         # defaults
    python prepare_genomic.py --data-dir output       # custom data dir
    python prepare_genomic.py --vocab-size 4096       # smaller vocabulary

Prerequisites:
    Run prepare_fastq.py first to generate output/train.txt and output/val.txt.
"""

import os
import sys
import time
import math
import pickle
import argparse

# rustbpe is only needed to TRAIN a tokenizer (train_tokenizer); loading/using an existing one
# (Tokenizer.from_directory) does not. Import lazily so environments without rustbpe — e.g. Aurora's
# frameworks module — can still load the tokenizer and train the model. See Aurora port notes.
try:
    import rustbpe
except ImportError:
    rustbpe = None
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length (must match train.py)
TIME_BUDGET = 300        # training time budget in seconds
EVAL_TOKENS = 40 * 524288

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

# Genomic special tokens — must include the 5 tokens from prepare_fastq.py
# plus a BOS token for sequence packing.
SPECIAL_TOKENS = [
    "<|bos|>",
    "<SAMPLE_START>",
    "<SAMPLE_END>",
    "<READ_START>",
    "<READ_END>",
    "<PAIRED_END>",
]
BOS_TOKEN = "<|bos|>"

# Default vocab size — smaller than NLP because DNA alphabet is tiny (A/C/G/T/N).
# Most merges will capture k-mer patterns.
DEFAULT_VOCAB_SIZE = 4096

# BPE split pattern: for genomic data we split on whitespace and special-token
# boundaries. Each contiguous block of bases becomes one chunk for BPE.
SPLIT_PATTERN = r"""[ACGTN]+|[^ACGTN]+"""

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def text_iterator(train_path, max_chars=1_000_000_000, chunk_size=1024 * 1024):
    """Yield text chunks from the training file."""
    nchars = 0
    with open(train_path, "r") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                return
            nchars += len(chunk)
            yield chunk
            if nchars >= max_chars:
                return


def train_tokenizer(train_path, vocab_size):
    """Train BPE tokenizer on genomic text, save as tiktoken pickle."""
    if rustbpe is None:
        raise ImportError(
            "rustbpe is required to TRAIN a tokenizer but is not installed. "
            "If you only need to load an existing tokenizer, use Tokenizer.from_directory(). "
            "To train, install rustbpe (see upstream autoresearch), then re-run.")
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    # --- Train with rustbpe ---
    print("Tokenizer: training BPE tokenizer on genomic data...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(
        text_iterator(train_path), vocab_size_no_special, pattern=SPLIT_PATTERN
    )

    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="genomic_bpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tokenizer
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, vocab_size={enc.n_vocab}")
    print(f"Tokenizer: saved to {tokenizer_pkl}")

    # --- Build token_bytes lookup for BPB evaluation ---
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check with genomic-style text
    test = "<SAMPLE_START> <READ_START> ACGTACGTNNACGT <READ_END> <SAMPLE_END>"
    encoded = enc.encode(test, allowed_special="all")
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed")

    # Show some example encodings
    print("\nExample encodings:")
    examples = ["ACGTACGT", "NNNNNN", "ATATATATATAT", "<READ_START>", "<SAMPLE_END>"]
    for ex in examples:
        ids = enc.encode(ex, allowed_special="all")
        print(f"  {ex:30s} -> {ids}")

    return enc


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py via prepare.py interface)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper compatible with train.py expectations."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode(text, allowed_special="all")
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = [self.enc.encode(t, allowed_special="all") for t in text]
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


# ---------------------------------------------------------------------------
# Dataloader for genomic text files
# ---------------------------------------------------------------------------

def _document_batches(split, data_dir, tokenizer_batch_size=128):
    """Infinite iterator over text chunks from train.txt or val.txt."""
    filename = "train.txt" if split == "train" else "val.txt"
    filepath = os.path.join(data_dir, filename)
    assert os.path.exists(filepath), f"Missing {filepath}. Run prepare_fastq.py first."

    chunk_size = 64 * 1024  # 64KB chunks
    epoch = 1
    while True:
        with open(filepath, "r") as f:
            batch = []
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                batch.append(chunk)
                if len(batch) >= tokenizer_batch_size:
                    yield batch, epoch
                    batch = []
            if batch:
                yield batch, epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, data_dir="output", buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Compatible with train.py interface.
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split, data_dir)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=(device == "cuda"))
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size, data_dir="output"):
    """Bits per byte evaluation on val split."""
    device = next(model.parameters()).device
    token_bytes = get_token_bytes(device=device)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val", data_dir=data_dir)
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train genomic BPE tokenizer")
    parser.add_argument("--data-dir", type=str, default="output",
                        help="Directory containing train.txt and val.txt from prepare_fastq.py")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE,
                        help="BPE vocabulary size (default: 4096)")
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, "train.txt")
    val_path = os.path.join(args.data_dir, "val.txt")

    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found. Run prepare_fastq.py first.")
        sys.exit(1)
    if not os.path.exists(val_path):
        print(f"ERROR: {val_path} not found. Run prepare_fastq.py first.")
        sys.exit(1)

    train_size = os.path.getsize(train_path)
    val_size = os.path.getsize(val_path)
    print(f"Data: train.txt = {train_size / 1e6:.1f} MB, val.txt = {val_size / 1e6:.1f} MB")
    print(f"Tokenizer cache: {TOKENIZER_DIR}")
    print(f"Vocab size: {args.vocab_size}")
    print()

    train_tokenizer(train_path, args.vocab_size)

    print()
    print("Done! Ready to train.")
    print("Next: modify train.py to import from prepare_genomic instead of prepare")
