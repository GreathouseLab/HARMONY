"""
k-mer / Markov floor baseline for within-read MLM (Tier-0 diagnostic).

Question it answers: how much of the depth-2/depth-4 MLM's MSK-DNA top-1 (~0.109) is just
LOCAL token co-occurrence that a non-neural count model captures for free? If this baseline
~= 0.109, the neural net learned nothing beyond local n-gram statistics -> the within-read
ceiling is intrinsic (go to Layer 2 / add information). If neural >> this, there is non-local
within-read signal the net is exploiting and worth pushing on.

Model: bidirectional one-neighbor interpolation, predicting the masked center token from its
TWO immediate neighbors with backoff to single-neighbor and unigram:
    P(c | left, right) = λ_tri·P(c|l,r) + λ_l·P(c|l) + λ_r·P(c|r) + λ_uni·P(c)
Counts estimated on output/train.txt; scored on output/val.txt with the SAME MSK-DNA exclusion
and scoring as floor_diag_big.py (MSK branch only, exclude special ids {4090..4096}).

DESIGN NOTE — this baseline uses GOLD neighbors (from the un-corrupted read), not the masked
input the MLM sees. That is deliberate: it measures the *intrinsic* local predictability of the
data (an upper-ish bound on what immediate local context can give), so it's a generous, clean
ceiling. The mixture weights are fixed/untuned (this is a floor, not a tuned LM).

Sanity check baked in: the pure-unigram arm should reproduce the established top-1 ~= 0.0923,
which validates that this pipeline's tokenization/masking/scoring matches the prior methodology.

Measurement only: no training, no model, no GPU. Writes experiments/kmer_markov_baseline.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_mlm import mlm_mask, EXTENDED_VOCAB, MASK_ID, SPECIAL_IDS_NEVER_MASK
from prepare_genomic import Tokenizer
from probe_sample_coherence import parse_val_txt_grouped

TRAIN_TXT = "output/train.txt"
VAL_TXT = "output/val.txt"
OUT_JSON = Path("experiments/kmer_markov_baseline.json")
EXCL_SET = set(SPECIAL_IDS_NEVER_MASK)            # {4090..4095, 4096} — never a predicted center
VOCAB = EXTENDED_VOCAB                            # 4097
MASK_SEED = 7                                     # same nominal seed as floor_diag_big.py
N_PER_SAMPLE_VAL = 50
# train.txt is ~6 GB; we don't need every read to estimate n-gram stats. Cap reads/sample —
# 1500 × ~125 samples ≈ 190k reads is plenty for bigram/trigram counts over a ~4090 DNA vocab.
N_PER_SAMPLE_TRAIN = 1500
PARSE_SEED = 42
# Fixed, untuned interpolation weights (sum to 1).
LAM_TRI, LAM_L, LAM_R, LAM_UNI = 0.50, 0.20, 0.20, 0.10


def is_dna(tid: int) -> bool:
    return tid not in EXCL_SET


def read_ids(tk, body: str):
    return tk.encode(f"<READ_START> {body} <READ_END>")


def build_counts(tk):
    """Count over ALL train reads. Centers are DNA tokens only; neighbors may be any token
    (incl. special tokens like <PAIRED_END>, which are informative context)."""
    samples = parse_val_txt_grouped(TRAIN_TXT, n_per_sample=N_PER_SAMPLE_TRAIN, seed=PARSE_SEED)
    uni = defaultdict(int)
    left_cond = defaultdict(lambda: defaultdict(int))     # left -> center -> count
    right_cond = defaultdict(lambda: defaultdict(int))    # right -> center -> count
    tri_cond = defaultdict(lambda: defaultdict(int))      # (left,right) -> center -> count
    n_reads = 0
    for s in samples:
        for body in samples[s]:
            ids = read_ids(tk, body)
            n_reads += 1
            L = len(ids)
            for p in range(L):
                c = ids[p]
                if not is_dna(c):
                    continue
                uni[c] += 1
                if p - 1 >= 0:
                    left_cond[ids[p - 1]][c] += 1
                if p + 1 < L:
                    right_cond[ids[p + 1]][c] += 1
                if p - 1 >= 0 and p + 1 < L:
                    tri_cond[(ids[p - 1], ids[p + 1])][c] += 1
    return uni, left_cond, right_cond, tri_cond, n_reads


def normed_array(counter):
    """dict center->count -> dense prob array over VOCAB (sums to 1)."""
    arr = np.zeros(VOCAB, dtype=np.float64)
    tot = 0
    for c, n in counter.items():
        arr[c] = n
        tot += n
    if tot > 0:
        arr /= tot
    return arr


def normed_sparse(counter):
    tot = sum(counter.values())
    if tot == 0:
        return {}
    return {c: n / tot for c, n in counter.items()}


def main():
    tk = Tokenizer.from_directory()
    print("[kmer] building train counts ...")
    uni, left_cond, right_cond, tri_cond, n_train_reads = build_counts(tk)
    puni = normed_array(uni)                                   # dense unigram over DNA centers
    uni_argmax = int(np.argmax(puni))
    left_p = {l: normed_sparse(d) for l, d in left_cond.items()}
    right_p = {r: normed_sparse(d) for r, d in right_cond.items()}
    tri_p = {lr: normed_sparse(d) for lr, d in tri_cond.items()}
    print(f"[kmer] train reads={n_train_reads}  distinct DNA centers={int((puni>0).sum())}  "
          f"left-contexts={len(left_p)} right-contexts={len(right_p)} tri-contexts={len(tri_p)}")

    # ---- eval on val, same MSK-DNA position set as floor_diag ----
    samples = parse_val_txt_grouped(VAL_TXT, n_per_sample=N_PER_SAMPLE_VAL, seed=PARSE_SEED)
    flat = [(s, b) for s in sorted(samples.keys()) for b in samples[s]]
    torch.manual_seed(MASK_SEED)

    n_pos = 0
    # interpolated bidirectional model
    bi_top1 = bi_top5 = 0
    bi_logp = 0.0
    # pure-unigram sanity arm (always predicts uni_argmax)
    uni_top1 = 0
    uni_logp = 0.0
    # left-only causal gold bigram (reference point)
    lo_top1 = lo_top5 = 0
    lo_logp = 0.0

    for _, body in flat:
        ids = read_ids(tk, body)
        L = len(ids)
        tok = torch.tensor([ids], dtype=torch.long)
        # We mask only to SELECT which positions count (MSK-DNA), then predict from GOLD neighbors.
        _, targets = mlm_mask(tok, vocab_size=EXTENDED_VOCAB, mask_prob=0.15)
        for p in range(L):
            if int(targets[0, p].item()) == -1:
                continue
            true_id = ids[p]
            if not is_dna(true_id):
                continue
            n_pos += 1
            l = ids[p - 1] if p - 1 >= 0 else None
            r = ids[p + 1] if p + 1 < L else None

            # ---- bidirectional interpolation ----
            arr = LAM_UNI * puni.copy()
            wsum = LAM_UNI
            if (l, r) in tri_p:
                for c, pr in tri_p[(l, r)].items():
                    arr[c] += LAM_TRI * pr
                wsum += LAM_TRI
            if l is not None and l in left_p:
                for c, pr in left_p[l].items():
                    arr[c] += LAM_L * pr
                wsum += LAM_L
            if r is not None and r in right_p:
                for c, pr in right_p[r].items():
                    arr[c] += LAM_R * pr
                wsum += LAM_R
            arr /= max(wsum, 1e-12)                     # renormalize to a proper dist
            top5 = np.argpartition(-arr, 5)[:5]
            top5 = top5[np.argsort(-arr[top5])]
            if int(top5[0]) == true_id:
                bi_top1 += 1
            if true_id in set(int(x) for x in top5):
                bi_top5 += 1
            bi_logp += float(np.log(max(arr[true_id], 1e-12)))

            # ---- left-only causal gold bigram ----
            larr = LAM_UNI * puni.copy()
            lw = LAM_UNI
            if l is not None and l in left_p:
                for c, pr in left_p[l].items():
                    larr[c] += (LAM_L + LAM_R + LAM_TRI) * pr
                lw += (LAM_L + LAM_R + LAM_TRI)
            larr /= max(lw, 1e-12)
            ltop5 = np.argpartition(-larr, 5)[:5]
            ltop5 = ltop5[np.argsort(-larr[ltop5])]
            if int(ltop5[0]) == true_id:
                lo_top1 += 1
            if true_id in set(int(x) for x in ltop5):
                lo_top5 += 1
            lo_logp += float(np.log(max(larr[true_id], 1e-12)))

            # ---- pure unigram sanity ----
            if uni_argmax == true_id:
                uni_top1 += 1
            uni_logp += float(np.log(max(puni[true_id], 1e-12)))

    def pack(top1, top5, logp):
        return {"top1": top1 / n_pos, "top5": top5 / n_pos, "ce_nats": -logp / n_pos}

    bi = pack(bi_top1, bi_top5, bi_logp)
    lo = pack(lo_top1, lo_top5, lo_logp)
    un = {"top1": uni_top1 / n_pos, "top5": None, "ce_nats": -uni_logp / n_pos}

    summary = {
        "method": "bidirectional one-neighbor interpolation with gold neighbors (count-based, untuned)",
        "interpolation_weights": {"tri": LAM_TRI, "left": LAM_L, "right": LAM_R, "uni": LAM_UNI},
        "train_path": TRAIN_TXT, "val_path": VAL_TXT,
        "n_train_reads": n_train_reads, "n_per_sample_train_cap": N_PER_SAMPLE_TRAIN,
        "eval": {
            "n_msk_dna_positions": n_pos,
            "mask_seed": MASK_SEED, "n_per_sample_val": N_PER_SAMPLE_VAL,
            "excluded_token_ids": sorted(EXCL_SET),
        },
        "results": {
            "unigram_sanity":          un,   # should reproduce ~0.0923 top-1 if pipeline matches
            "left_only_gold_bigram":   lo,
            "bidirectional_gold_trigram": bi,
        },
        "reference_points": {
            "established_unigram_top1": 0.0923, "established_unigram_ce": 6.7304,
            "depth2_lam0_mlm_top1": 0.1090, "depth2_lam0_mlm_ce": 6.2701,
            "depth4_big_mlm_top1": 0.1119, "depth4_big_mlm_ce": 6.8506,
        },
    }
    # interpretation
    bi_vs_mlm = bi["top1"] - 0.1090
    if bi["top1"] >= 0.1090 - 0.005:
        verdict = ("Local co-occurrence ALONE (gold neighbors, count model) reaches the neural MLM's "
                   "MSK-DNA top-1 -> the within-read signal the MLM captures is essentially local "
                   "n-gram statistics; the ceiling looks intrinsic to local context. Push INFORMATION "
                   "(paired mate / longer context) or move to Layer 2, not capacity.")
    elif bi["top1"] >= 0.1090 - 0.02:
        verdict = ("Local co-occurrence explains MOST but not all of the MLM top-1; a little non-local "
                   "within-read signal exists. Marginal room — regularized training may recover it, but "
                   "the bigger lever is still added information.")
    else:
        verdict = ("Local co-occurrence is well BELOW the MLM top-1 -> the MLM exploits non-local "
                   "within-read structure beyond immediate neighbors; within-read modeling is NOT "
                   "saturated and is worth pushing (better optimization/regularization).")
    summary["verdict"] = {"bidirectional_top1_minus_depth2_mlm": bi_vs_mlm, "interpretation": verdict}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[kmer] eval positions (MSK-DNA) = {n_pos}")
    print(f"{'arm':<32}{'top1':>10}{'top5':>10}{'CE(nats)':>12}")
    print(f"{'unigram (sanity ~0.0923)':<32}{un['top1']:>10.4f}{'--':>10}{un['ce_nats']:>12.4f}")
    print(f"{'left-only gold bigram':<32}{lo['top1']:>10.4f}{lo['top5']:>10.4f}{lo['ce_nats']:>12.4f}")
    print(f"{'bidirectional gold trigram':<32}{bi['top1']:>10.4f}{bi['top5']:>10.4f}{bi['ce_nats']:>12.4f}")
    print(f"\n{'depth-2 λ=0 MLM (ref)':<32}{0.1090:>10.4f}{0.1521:>10.4f}{6.2701:>12.4f}")
    print(f"{'depth-4 big MLM (ref)':<32}{0.1119:>10.4f}{0.1436:>10.4f}{6.8506:>12.4f}")
    print(f"\nverdict: {verdict}")
    print(f"[kmer] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
