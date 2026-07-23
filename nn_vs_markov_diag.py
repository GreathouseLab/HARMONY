"""
Step-1 diagnostic: WHY does the neural MLM lose to a trigram count model?

A bidirectional gold-neighbor count model gets MSK-DNA top-1 = 0.1272; the 41M neural MLM
gets 0.1119. Either the BERT 15% masking corrupts the very neighbors the model needs
(-> lever = reduce masking corruption, a trivial training change) or the model underfits
even with clean neighbors (-> lever = architecture/optimization).

This script feeds the NN and the count model the SAME masked sequences and the SAME MSK-DNA
positions, then partitions positions by neighbor cleanliness:
  - both_clean : both immediate neighbors are in-bounds AND the model sees their TRUE token
                 (masked_input[nb] == gold[nb])
  - corrupted  : at least one neighbor is [MASK]/random-replaced or out of bounds (read edge)

For each partition it reports:
  - NN top-1            : neural model fed the masked sequence (what it actually sees)
  - Markov-gold top-1   : count model using GOLD neighbors (the 0.127 ceiling; on both_clean
                          positions gold==what-NN-sees, so this is the fair head-to-head)
  - Markov-corrupt top-1: count model using the SAME corrupted neighbors the NN sees

Verdict:
  NN >= Markov-gold on both_clean  -> deficit is masking corruption -> reduce mask rate / don't
                                      mask adjacent tokens (measurable now via the val-MLM hook).
  NN <  Markov-gold on both_clean  -> architecture/optimization underfit -> arch/training sweep.

Measurement only. Writes experiments/nn_vs_markov_diag.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train_mlm import build_model, mlm_mask, EXTENDED_VOCAB, MASK_ID, SPECIAL_IDS_NEVER_MASK, PAD_ID
from prepare_genomic import Tokenizer
from probe_sample_coherence import parse_val_txt_grouped
from kmer_markov_baseline import (
    build_counts, normed_array, normed_sparse,
    LAM_TRI, LAM_L, LAM_R, LAM_UNI, MASK_SEED, N_PER_SAMPLE_VAL, PARSE_SEED,
)

EXCL_SET = set(SPECIAL_IDS_NEVER_MASK)
OUT_JSON = Path("experiments/nn_vs_markov_diag.json")
T_MAX = 64
CHECKPOINTS = [
    ("depth2_lam0", "experiments/mlm_lam0/checkpoint.pt", 2, 128),
    ("depth4_big",  "experiments/mlm_big/checkpoint.pt",  4, 192),
]


def is_dna(tid: int) -> bool:
    return tid not in EXCL_SET


def markov_argmax(l, r, puni, left_p, right_p, tri_p):
    """Top-1 center prediction from (left,right) via the same interpolation as the baseline.
    l/r may be None (edge) or any token id (incl. [MASK], which simply backs off)."""
    arr = LAM_UNI * puni.copy()
    if l is not None and r is not None and (l, r) in tri_p:
        for c, pr in tri_p[(l, r)].items():
            arr[c] += LAM_TRI * pr
    if l is not None and l in left_p:
        for c, pr in left_p[l].items():
            arr[c] += LAM_L * pr
    if r is not None and r in right_p:
        for c, pr in right_p[r].items():
            arr[c] += LAM_R * pr
    return int(np.argmax(arr))


def load_nn(name, ckpt_path, depth, ar, device):
    model, cfg = build_model(device, depth=depth, aspect_ratio=ar, seq_len=T_MAX)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


def main():
    from device_utils import get_device
    device = get_device()
    tk = Tokenizer.from_directory()

    print("[diag] building train counts (Markov) ...")
    uni, left_cond, right_cond, tri_cond, n_train_reads = build_counts(tk)
    puni = normed_array(uni)
    left_p = {l: normed_sparse(d) for l, d in left_cond.items()}
    right_p = {r: normed_sparse(d) for r, d in right_cond.items()}
    tri_p = {lr: normed_sparse(d) for lr, d in tri_cond.items()}
    print(f"[diag] train reads={n_train_reads}")

    samples = parse_val_txt_grouped("output/val.txt", n_per_sample=N_PER_SAMPLE_VAL, seed=PARSE_SEED)
    flat = [(s, b) for s in sorted(samples.keys()) for b in samples[s]]

    # Precompute the SAME masking for every read once, reused across all NN checkpoints.
    torch.manual_seed(MASK_SEED)
    per_read = []  # (gold_ids[list], masked_ids[list], L, [positions])
    for _, body in flat:
        ids = tk.encode(f"<READ_START> {body} <READ_END>")
        L = len(ids)
        tok = torch.tensor([ids], dtype=torch.long)
        masked, targets = mlm_mask(tok, vocab_size=EXTENDED_VOCAB, mask_prob=0.15)
        gold = ids
        masked_ids = masked[0].tolist()
        positions = []
        for p in range(L):
            if int(targets[0, p].item()) == -1:
                continue
            if int(masked[0, p].item()) != MASK_ID:   # MSK branch only (exclude 10% keep / 10% random)
                continue
            if not is_dna(gold[p]):
                continue
            positions.append(p)
        per_read.append((gold, masked_ids, L, positions))

    # ---- partition + Markov scores (model-independent) ----
    # bucket -> dict of counters
    def fresh():
        return {"n": 0, "nn": {}, "mk_gold": 0, "mk_corrupt": 0}
    buckets = {"both_clean": fresh(), "corrupted": fresh()}
    total_pos = 0
    for gold, masked_ids, L, positions in per_read:
        for p in positions:
            total_pos += 1
            true_id = gold[p]
            has_l, has_r = p - 1 >= 0, p + 1 < L
            l_clean = has_l and masked_ids[p - 1] == gold[p - 1]
            r_clean = has_r and masked_ids[p + 1] == gold[p + 1]
            bkt = "both_clean" if (l_clean and r_clean) else "corrupted"
            b = buckets[bkt]
            b["n"] += 1
            # gold neighbors
            gl = gold[p - 1] if has_l else None
            gr = gold[p + 1] if has_r else None
            if markov_argmax(gl, gr, puni, left_p, right_p, tri_p) == true_id:
                b["mk_gold"] += 1
            # corrupted (what NN sees) neighbors
            cl = masked_ids[p - 1] if has_l else None
            cr = masked_ids[p + 1] if has_r else None
            if markov_argmax(cl, cr, puni, left_p, right_p, tri_p) == true_id:
                b["mk_corrupt"] += 1

    # ---- NN scores per checkpoint ----
    nn_overall = {}
    for name, ckpt_path, depth, ar in CHECKPOINTS:
        print(f"[diag] scoring NN {name} ...")
        model = load_nn(name, ckpt_path, depth, ar, device)
        n_correct = {"both_clean": 0, "corrupted": 0}
        n_total = 0
        with torch.no_grad():
            for gold, masked_ids, L, positions in per_read:
                if not positions:
                    continue
                padded = masked_ids + [PAD_ID] * (T_MAX - L) if L < T_MAX else masked_ids[:T_MAX]
                Lc = min(L, T_MAX)
                inp = torch.tensor([padded], dtype=torch.long, device=device)
                attn = torch.tensor([Lc], dtype=torch.long, device=device)
                logits, _ = model.forward_mlm(inp, attn_lens=attn)
                pred = logits[0].argmax(dim=-1).tolist()
                for p in positions:
                    if p >= T_MAX:
                        continue
                    true_id = gold[p]
                    has_l, has_r = p - 1 >= 0, p + 1 < L
                    l_clean = has_l and masked_ids[p - 1] == gold[p - 1]
                    r_clean = has_r and masked_ids[p + 1] == gold[p + 1]
                    bkt = "both_clean" if (l_clean and r_clean) else "corrupted"
                    n_total += 1
                    if pred[p] == true_id:
                        n_correct[bkt] += 1
        for bkt in buckets:
            buckets[bkt]["nn"][name] = n_correct[bkt]
        nn_overall[name] = (n_correct["both_clean"] + n_correct["corrupted"]) / max(n_total, 1)

    # ---- assemble ----
    def rate(num, den):
        return num / den if den else float("nan")

    report = {"total_msk_dna_positions": total_pos, "n_train_reads": n_train_reads,
              "mask_prob": 0.15, "mask_seed": MASK_SEED, "buckets": {}}
    for bkt, b in buckets.items():
        n = b["n"]
        report["buckets"][bkt] = {
            "n_positions": n,
            "frac_of_total": rate(n, total_pos),
            "markov_gold_top1": rate(b["mk_gold"], n),
            "markov_corrupt_top1": rate(b["mk_corrupt"], n),
            "nn_top1": {name: rate(b["nn"][name], n) for name, *_ in CHECKPOINTS},
        }
    report["nn_overall_top1"] = nn_overall

    # ---- verdict (use strongest NN = depth4_big on both_clean vs markov_gold) ----
    bc = report["buckets"]["both_clean"]
    nn_clean = bc["nn_top1"]["depth4_big"]
    mk_clean = bc["markov_gold_top1"]
    gap_clean = nn_clean - mk_clean
    if gap_clean >= -0.005:
        verdict = ("On CLEAN-neighbor positions the NN matches/beats the gold count model -> the NN's "
                   "overall deficit is concentrated on CORRUPTED-neighbor positions. LEVER: reduce "
                   "masking corruption (lower mask rate and/or never mask two adjacent tokens). "
                   "Trivial training change; measurable immediately via best_val.pt / the val-MLM hook.")
        label = "masking-corruption"
    else:
        verdict = ("The NN trails the gold count model EVEN on clean-neighbor positions -> it is "
                   "underfitting local structure it can fully see. LEVER: architecture/optimization "
                   "(the CLM-derived rotary+softcap+Muon stack may be mistuned for bidirectional MLM), "
                   "not masking and not information.")
        label = "architecture-underfit"
    report["verdict"] = {"label": label, "nn_clean_minus_markovgold_clean": gap_clean,
                         "interpretation": verdict}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    # ---- print ----
    print(f"\n[diag] total MSK-DNA positions = {total_pos}")
    print(f"{'bucket':<14}{'n':>7}{'frac':>8}{'mk_gold':>10}{'mk_corrupt':>12}"
          f"{'NN_d2':>9}{'NN_d4':>9}")
    for bkt in ("both_clean", "corrupted"):
        b = report["buckets"][bkt]
        print(f"{bkt:<14}{b['n_positions']:>7}{b['frac_of_total']:>8.2f}"
              f"{b['markov_gold_top1']:>10.4f}{b['markov_corrupt_top1']:>12.4f}"
              f"{b['nn_top1']['depth2_lam0']:>9.4f}{b['nn_top1']['depth4_big']:>9.4f}")
    print(f"\nNN overall top1: " + ", ".join(f"{k}={v:.4f}" for k, v in nn_overall.items()))
    print(f"verdict [{label}]: {verdict}")
    print(f"[diag] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
