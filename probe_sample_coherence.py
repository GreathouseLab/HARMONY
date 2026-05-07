"""
Probe 1 — Sample-coherence.

Tests whether trained model embeds sample-level structure: are reads from the
same sample stem more similar in embedding space than reads from different stems?

Metric: ROC-AUC distinguishing within-sample vs between-sample cosine similarities.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC via Mann-Whitney U / (n_pos * n_neg). Equivalent to sklearn's
    roc_auc_score for binary {0,1} y_true; handles ties via average ranks.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos_mask = y_true == 1
    n_pos = int(pos_mask.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Average ranks (1-indexed) handle ties correctly.
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    # Tie correction: average ranks within groups of equal scores.
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + j) + 1.0  # average of (i+1)..(j+1)
            ranks[order[i:j + 1]] = avg
        i = j + 1
    rank_sum_pos = ranks[pos_mask].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ============================================================
# Sampling reads grouped by sample stem from val.txt
# ============================================================

SPECIAL_TOKEN_NAMES = (
    "<|bos|>",
    "<SAMPLE_START>",
    "<SAMPLE_END>",
    "<READ_START>",
    "<READ_END>",
    "<PAIRED_END>",
)


def parse_val_txt_grouped(val_path: Path, n_per_sample: int, seed: int = 42):
    """Parse val.txt and return {sample_idx: [list of read body strings]}.

    val.txt format (post-2026-04-29 leak fix):
        <SAMPLE_START> <READ_START> ACGT... <PAIRED_END> ACGT... <READ_END>
        <READ_START> ... <READ_END> ... <SAMPLE_END>
        <SAMPLE_START> ...

    Read bodies are returned with bases space-separated as they appear in the
    file (and including <PAIRED_END> for paired molecules).
    """
    rng = random.Random(seed)

    samples_reads: dict[int, list[str]] = {}
    current_sample_idx = -1
    current_reads: list[str] = []
    buffer: list[str] = []
    in_read = False

    with open(val_path, "r") as f:
        for chunk in f:
            for tok in chunk.split():
                if tok == "<SAMPLE_START>":
                    current_sample_idx += 1
                    current_reads = []
                elif tok == "<SAMPLE_END>":
                    if current_reads:
                        if len(current_reads) > n_per_sample:
                            current_reads = rng.sample(current_reads, n_per_sample)
                        samples_reads[current_sample_idx] = current_reads
                elif tok == "<READ_START>":
                    in_read = True
                    buffer = []
                elif tok == "<READ_END>":
                    if in_read and buffer:
                        current_reads.append(" ".join(buffer))
                    in_read = False
                    buffer = []
                elif in_read:
                    buffer.append(tok)
                # silently ignore unknown tokens

    return samples_reads


# ============================================================
# Embedding extraction (matches Probe 4 convention)
# ============================================================

def _collect_special_ids(tokenizer) -> set[int]:
    ids: set[int] = set()
    for name in SPECIAL_TOKEN_NAMES:
        try:
            ids.add(tokenizer.enc.encode_single_token(name))
        except Exception:
            pass
    return ids


def embed_reads(model, tokenizer, reads: list[str], device, special_ids: set[int]) -> np.ndarray:
    """Encode each read via mean-pool of final hidden state across DNA token positions.

    Skips special-token positions (READ_START/READ_END/PAIRED_END/etc.) in the
    mean-pool. Returns (n_reads, n_embd) float32 numpy array.

    Embeds one read at a time — same as evaluate_probes.embed_read — to avoid
    introducing padding artifacts that the model wasn't trained on.
    """
    model.eval()
    out: list[np.ndarray] = []
    for body in reads:
        text = f"<READ_START> {body} <READ_END>"
        ids = tokenizer.encode(text)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            h = model(x, return_hidden=True)  # (1, T, n_embd)
        h = h[0]  # (T, n_embd)
        mask = torch.tensor(
            [tid not in special_ids for tid in ids],
            dtype=torch.bool, device=device,
        )
        if not bool(mask.any()):
            # No DNA tokens — should not happen, but guard against div-by-zero
            out.append(np.zeros(h.shape[-1], dtype=np.float32))
            continue
        pooled = h[mask].float().mean(dim=0)
        out.append(pooled.cpu().numpy())
    return np.stack(out, axis=0)


# ============================================================
# Sample-coherence metric
# ============================================================

def compute_sample_coherence(embeddings: np.ndarray, sample_labels: np.ndarray,
                             max_pairs_per_class: int = 50_000, seed: int = 42):
    """Compute ROC-AUC distinguishing within-sample from between-sample cosine sims.

    embeddings: (N, n_embd)
    sample_labels: (N,) int, sample index for each read.

    Returns dict with: auc, n_within_pairs, n_between_pairs, mean/std cosines,
    delta_mean. Returns NaN auc with an `error` field if there aren't enough pairs.
    """
    rng = np.random.RandomState(seed)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
    emb_n = embeddings / norms

    n = len(embeddings)
    sample_labels = np.asarray(sample_labels)

    # Within-sample pairs: for each sample with >=2 reads, all (i,j) i<j pairs.
    within_cos: list[float] = []
    unique_samples = np.unique(sample_labels)
    sample_to_idx = {int(s): np.where(sample_labels == s)[0] for s in unique_samples}
    for s, idxs in sample_to_idx.items():
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                within_cos.append(float(np.dot(emb_n[idxs[i]], emb_n[idxs[j]])))

    if len(within_cos) > max_pairs_per_class:
        sel = rng.choice(len(within_cos), max_pairs_per_class, replace=False)
        within_cos = [within_cos[k] for k in sel]

    # Between-sample pairs: random cross-sample pairs, dedup'd.
    n_between_target = min(len(within_cos), max_pairs_per_class)
    between_cos: list[float] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(n_between_target * 5, 1000)
    while len(between_cos) < n_between_target and attempts < max_attempts:
        i, j = rng.choice(n, 2, replace=False)
        key = (int(min(i, j)), int(max(i, j)))
        if sample_labels[i] != sample_labels[j] and key not in seen:
            seen.add(key)
            between_cos.append(float(np.dot(emb_n[i], emb_n[j])))
        attempts += 1

    within_arr = np.array(within_cos)
    between_arr = np.array(between_cos)

    if len(within_arr) == 0 or len(between_arr) == 0:
        return {
            "auc": float("nan"),
            "error": "insufficient pairs",
            "n_within_pairs": int(len(within_arr)),
            "n_between_pairs": int(len(between_arr)),
        }

    y_true = np.concatenate([np.ones(len(within_arr)), np.zeros(len(between_arr))])
    y_score = np.concatenate([within_arr, between_arr])
    auc = _roc_auc_score(y_true, y_score)

    return {
        "auc": float(auc),
        "n_within_pairs": int(len(within_arr)),
        "n_between_pairs": int(len(between_arr)),
        "mean_within_cos": float(within_arr.mean()),
        "mean_between_cos": float(between_arr.mean()),
        "std_within_cos": float(within_arr.std()),
        "std_between_cos": float(between_arr.std()),
        "delta_mean": float(within_arr.mean() - between_arr.mean()),
    }


# ============================================================
# Main probe driver
# ============================================================

def run_probe1(checkpoint_path: Path, val_path: Path, output_path: Path,
               n_reads_per_sample: int = 100, seed: int = 42,
               device: str | torch.device = "mps",
               model=None, tokenizer=None, ckpt_meta: dict | None = None):
    """Run Probe 1 on a single checkpoint, write JSON results.

    If `model`/`tokenizer` are passed in, the loader is skipped (used by the
    integration in evaluate_probes when running multiple probes back-to-back).
    """
    if isinstance(device, str):
        device_t = torch.device(device)
    else:
        device_t = device

    print("Probe 1 — Sample-coherence")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Val data:   {val_path}")
    print(f"  Device:     {device_t}")
    print(f"  N reads per sample: {n_reads_per_sample}")
    print()

    if model is None or tokenizer is None:
        # Reuse evaluate_probes' loader and the prepare_genomic Tokenizer
        from evaluate_probes import load_checkpoint
        from prepare_genomic import Tokenizer
        t0 = time.time()
        model, cfg, ckpt = load_checkpoint(checkpoint_path, device_t)
        tokenizer = Tokenizer.from_directory()
        ckpt_meta = {
            "n_embd": cfg.n_embd,
            "n_layer": cfg.n_layer,
            "vocab_size": cfg.vocab_size,
            "val_bpb": ckpt.get("val_bpb"),
            "git_sha": ckpt.get("git_sha"),
            "seed": ckpt.get("seed"),
            "hyperparams": ckpt.get("hyperparams"),
        }
        print(f"  Model loaded in {time.time()-t0:.1f}s; n_embd={ckpt_meta['n_embd']}, depth={ckpt_meta['n_layer']}")
    else:
        if ckpt_meta is None:
            ckpt_meta = {}

    special_ids = _collect_special_ids(tokenizer)

    # Sample reads grouped by sample
    t0 = time.time()
    samples_reads = parse_val_txt_grouped(val_path, n_per_sample=n_reads_per_sample, seed=seed)
    n_samples = len(samples_reads)
    n_total_reads = sum(len(rs) for rs in samples_reads.values())
    print(f"  Sampled {n_total_reads} reads across {n_samples} samples in {time.time()-t0:.1f}s")
    if n_samples > 0:
        print(f"    (mean {n_total_reads / n_samples:.1f} reads per sample)")

    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for between-sample comparison; got {n_samples}")

    reads: list[str] = []
    labels: list[int] = []
    for sample_idx, sample_reads in samples_reads.items():
        for r in sample_reads:
            reads.append(r)
            labels.append(sample_idx)
    labels_arr = np.array(labels)

    t0 = time.time()
    embeddings = embed_reads(model, tokenizer, reads, device_t, special_ids)
    print(f"  Embedded {len(reads)} reads in {time.time()-t0:.1f}s; shape={embeddings.shape}")

    t0 = time.time()
    result = compute_sample_coherence(embeddings, labels_arr, seed=seed)
    print(f"  Coherence computed in {time.time()-t0:.1f}s")
    if not np.isnan(result["auc"]):
        print(f"    AUC = {result['auc']:.4f}")
        print(f"    delta_mean = {result['delta_mean']:+.4f} "
              f"(within {result['mean_within_cos']:.4f} vs between {result['mean_between_cos']:.4f})")
        print(f"    n_within={result['n_within_pairs']}, n_between={result['n_between_pairs']}")
    else:
        print(f"    AUC = NaN ({result.get('error')})")

    full_result = {
        "probe": "sample_coherence",
        "checkpoint": str(checkpoint_path),
        "checkpoint_meta": ckpt_meta,
        "n_samples": int(n_samples),
        "n_reads_per_sample": int(n_reads_per_sample),
        "n_reads_total": int(n_total_reads),
        "seed": seed,
        "device": str(device_t),
        **result,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(full_result, f, indent=2)
    print(f"  Saved -> {output_path}")

    return full_result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--val-path", type=Path,
                    default=Path(__file__).parent / "output" / "val.txt")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-reads-per-sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    run_probe1(
        checkpoint_path=args.checkpoint,
        val_path=args.val_path,
        output_path=args.output,
        n_reads_per_sample=args.n_reads_per_sample,
        seed=args.seed,
        device=args.device,
    )
