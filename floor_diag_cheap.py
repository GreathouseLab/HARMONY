"""
Cheap floor diagnostics for the MVT checkpoint.
Measurement only: no training, no model edits.

Writes experiments/mlm_mvt/floor_diagnostics_cheap.json.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prepare_genomic import Tokenizer
from probe_sample_coherence import parse_val_txt_grouped


CHECKPOINT = "experiments/mlm_mvt/checkpoint.pt"
TSV_PATH = Path("experiments/mlm_mvt/decoded_predictions.tsv")
OUT_JSON = Path("experiments/mlm_mvt/floor_diagnostics_cheap.json")

# Exclusion list = SPECIAL_IDS_NEVER_MASK in train_mlm.py
# = <|bos|>=4090, <SAMPLE_START>=4091, <SAMPLE_END>=4092, <READ_START>=4093,
#   <READ_END>=4094, <PAIRED_END>=4095, [MASK]=4096.
EXCLUDED_IDS = [4090, 4091, 4092, 4093, 4094, 4095, 4096]
EXCLUDED_NAMES = ["<|bos|>", "<SAMPLE_START>", "<SAMPLE_END>",
                  "<READ_START>", "<READ_END>", "<PAIRED_END>", "[MASK]"]
EXCL_SET = set(EXCLUDED_IDS)

# Contrastive setup (read from code; quoted in D1 above)
SAMPLES_PER_BATCH = 4
READS_PER_SAMPLE  = 8
BATCH_SIZE        = SAMPLES_PER_BATCH * READS_PER_SAMPLE   # 32
N_CANDIDATES      = BATCH_SIZE - 1                          # 31
TEMPERATURE       = 0.07
LAMBDA            = 1.0
OBSERVED_CON_LOSS = 2.8031


def main():
    tk = Tokenizer.from_directory()

    # ===== D2.a — estimate train unigram (DNA tokens only) =====
    # Use the same parsing the loader uses; n_per_sample=200 to match what
    # the MVT trained on (5.6 GB train.txt; 126 samples × 200 reads = 25,200 reads,
    # ~1.2M token observations — plenty for unigram statistics).
    print("[D2] tokenizing train (n_per_sample=200) to estimate unigram...")
    train_groups = parse_val_txt_grouped("output/train.txt", n_per_sample=200, seed=42)
    n_train_reads = sum(len(v) for v in train_groups.values())
    counts = Counter()
    for s_idx, bodies in train_groups.items():
        for body in bodies:
            ids = tk.encode(f"<READ_START> {body} <READ_END>")
            for tid in ids:
                if tid not in EXCL_SET:
                    counts[tid] += 1
    total = sum(counts.values())
    # Distribution restricted to DNA tokens
    sorted_items = sorted(counts.items(), key=lambda kv: -kv[1])
    top_id, top_n = sorted_items[0]
    top_p = top_n / total
    top_text = tk.enc.decode([top_id])
    print(f"  train: {n_train_reads} reads parsed, {total:,} DNA tokens counted")
    print(f"  argmax DNA token: id={top_id}  text={top_text!r}  p={top_p:.4f}  count={top_n}")

    # Build full p vector and entropy
    p_train = np.zeros(4097, dtype=np.float64)
    for tid, c in counts.items():
        p_train[tid] = c / total
    # Eligible token coverage
    n_eligible_seen = int((p_train > 0).sum())
    entropy_train = -np.sum(p_train[p_train > 0] * np.log(p_train[p_train > 0]))

    # ===== D2.b — evaluate unigram on val masked positions =====
    # Read TSV produced by the visualizer/decoder pass. Filter to MSK branch + DNA-only.
    print("[D2.b] reading val masked positions from TSV...")
    rows = []
    with open(TSV_PATH) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    print(f"  total TSV rows (all branches, all true ids): {len(rows)}")

    # Per-branch counts (sanity)
    branch_counts = Counter(r["branch"] for r in rows)
    print(f"  branch composition: {dict(branch_counts)}")

    # Filter: MSK only + DNA only (true_id not in EXCL_SET).
    # By construction true_id should never be a special (mlm_mask excludes them), but filter defensively.
    msk_dna = [r for r in rows if r["branch"] == "MSK" and int(r["true_id"]) not in EXCL_SET]
    n_excluded_msk_true_special = sum(1 for r in rows if r["branch"] == "MSK" and int(r["true_id"]) in EXCL_SET)
    print(f"  MSK rows: {branch_counts['MSK']} ; after DNA filter: {len(msk_dna)} "
          f"(specials excluded from MSK true_id: {n_excluded_msk_true_special})")

    # Unigram top-1 on this set
    true_ids = np.array([int(r["true_id"]) for r in msk_dna], dtype=np.int64)
    uni_top1_hits = int((true_ids == top_id).sum())
    uni_top1_acc = uni_top1_hits / len(true_ids)
    # Unigram CE: -log p_train(true_id) with eps smoothing for never-seen ids.
    EPS = 1e-12
    log_p = np.log(np.where(p_train[true_ids] > 0, p_train[true_ids], EPS))
    uni_ce_nats = float(-log_p.mean())
    n_unseen = int((p_train[true_ids] == 0).sum())
    print(f"  unigram top-1 (always predict argmax DNA id={top_id}): {uni_top1_hits}/{len(true_ids)} = {uni_top1_acc:.4f}")
    print(f"  unigram CE (nats, train-estimated, eval on val MSK DNA): {uni_ce_nats:.4f}  "
          f"(unseen-id positions: {n_unseen})")

    # ===== D3 — model MSK/DNA-only top-1 + top-5 =====
    print("[D3] model accuracy on same filtered subset...")
    mt1 = sum(1 for r in msk_dna if r["top1_correct"] == "1")
    mt5 = sum(1 for r in msk_dna if r["top5_correct"] == "1")
    model_top1 = mt1 / len(msk_dna)
    model_top5 = mt5 / len(msk_dna)
    gap = model_top1 - uni_top1_acc
    print(f"  model MSK DNA top-1: {mt1}/{len(msk_dna)} = {model_top1:.4f}")
    print(f"  model MSK DNA top-5: {mt5}/{len(msk_dna)} = {model_top5:.4f}")
    print(f"  GAP (model top-1 - unigram top-1): {gap:+.4f}")

    # ===== JSON writeout =====
    ln_N = math.log(N_CANDIDATES)
    con_below_floor = ln_N - OBSERVED_CON_LOSS
    con_state = ("below floor" if con_below_floor > 0.1
                 else "near floor" if abs(con_below_floor) <= 0.1
                 else "above floor")

    summary = {
        "checkpoint": CHECKPOINT,
        "checkpoint_step": 19999,
        "reported_losses": {"mlm": 5.7094, "contrastive": OBSERVED_CON_LOSS},
        "contrastive": {
            "batch_size": BATCH_SIZE,
            "samples_per_batch": SAMPLES_PER_BATCH,
            "reads_per_sample": READS_PER_SAMPLE,
            "candidate_set_per_anchor": "all B-1 = 31 non-self rows of (B,B) sim matrix",
            "N_candidates_per_anchor": N_CANDIDATES,
            "ln_N_candidates": ln_N,
            "temperature": TEMPERATURE,
            "lambda": LAMBDA,
            "observed_loss": OBSERVED_CON_LOSS,
            "delta_below_floor_nats": con_below_floor,
            "effective_n_implied_by_loss": math.exp(OBSERVED_CON_LOSS),
            "state": con_state,
            "quote_loss_call": "infonce_with_sample_ids(z, sample_ids, temperature=0.07)  # train_mlm.py:372",
            "quote_lambda": "ap.add_argument('--lam', type=float, default=1.0)  # train_mlm.py:480",
        },
        "unigram_baseline": {
            "train_source": "output/train.txt",
            "train_n_per_sample": 200,
            "train_reads_tokenized": n_train_reads,
            "train_tokens_counted_after_exclusion": total,
            "excluded_token_ids": EXCLUDED_IDS,
            "excluded_token_names": EXCLUDED_NAMES,
            "argmax_token_id": int(top_id),
            "argmax_token_text": top_text,
            "argmax_token_count": int(top_n),
            "argmax_token_prob": float(top_p),
            "unigram_entropy_nats": float(entropy_train),
            "unique_token_ids_seen": n_eligible_seen,
            "eval_set": "val.txt masked positions, MSK branch only, DNA tokens only",
            "eval_n_positions": int(len(msk_dna)),
            "eval_unseen_true_ids_in_train_unigram": int(n_unseen),
            "unigram_top1": float(uni_top1_acc),
            "unigram_top1_hits": int(uni_top1_hits),
            "unigram_ce_nats": float(uni_ce_nats),
        },
        "model_msk_dna_only": {
            "eval_n_positions": int(len(msk_dna)),
            "top1": float(model_top1),
            "top1_hits": int(mt1),
            "top5": float(model_top5),
            "top5_hits": int(mt5),
        },
        "gap": {
            "model_top1_minus_unigram_top1": float(gap),
            "interpretation": (
                "above floor — model beats unigram baseline" if gap > 0.005
                else "at floor — model is essentially unigram-equivalent" if abs(gap) <= 0.005
                else "below floor — model is worse than unigram"
            ),
        },
        "tsv_source": str(TSV_PATH),
        "assumptions": [
            "Unigram is estimated from train.txt at n_per_sample=200 (matches what the MVT loader cached). "
            "This is a uniform-random subsample of train; full train would give a near-identical estimate "
            "(25,200 reads × ~50 tokens = ~1.2M token observations is already statistically saturated).",
            "Val masked positions come from the existing decoded_predictions.tsv (700 reads × ~15% masking = 4745 rows). "
            "Branch filter selects only the 80% MSK branch so KEEP-passthrough and RAND-input don't inflate the accuracy.",
            "DNA-only filter applies to true_id; by construction true_id is never special (mlm_mask excludes them) so this filter is defensive.",
        ],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[json] wrote {OUT_JSON}  ({OUT_JSON.stat().st_size} bytes)")

    # ===== Verification block =====
    print()
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"contrastive: N={N_CANDIDATES}  ln(N)={ln_N:.4f}  loss={OBSERVED_CON_LOSS:.4f}  -> {con_state}")
    print(f"             (effective N implied by loss = exp({OBSERVED_CON_LOSS:.4f}) = {math.exp(OBSERVED_CON_LOSS):.2f})")
    print(f"unigram (DNA, val): top1={uni_top1_acc:.4f}  CE={uni_ce_nats:.4f} nats")
    print(f"model   (MSK,DNA,val): top1={model_top1:.4f}  top5={model_top5:.4f}")
    print(f"gap (model - unigram top1): {gap:+.4f}  -> {summary['gap']['interpretation']}")


if __name__ == "__main__":
    main()
