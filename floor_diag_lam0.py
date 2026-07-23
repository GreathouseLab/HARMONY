"""
λ=0 (pure MLM) isolation diagnostics — same methodology as floor_diag_cheap.py (Prompt A),
but inference-driven (no pre-existing TSV).

For each of 700 val reads (14 samples × 50, seed=7 — same as the prior decode pass that
produced experiments/mlm_mvt/decoded_predictions.tsv), apply the same MLM masking and
score:
  - MSK-branch only (exclude KEEP and RAND)
  - DNA tokens only (exclude special ids {4090..4096})
  - top-1 accuracy, top-5 accuracy, cross-entropy in nats

Also dumps a TSV of decoded predictions for parity with the MVT artifacts.
Writes experiments/mlm_lam0/lam0_isolation.json.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train_mlm import GPT_MLM, build_model, mlm_mask, EXTENDED_VOCAB, MASK_ID
from prepare_genomic import Tokenizer
from probe_sample_coherence import parse_val_txt_grouped


CHECKPOINT = "experiments/mlm_lam0/checkpoint.pt"
OUT_DIR = Path("experiments/mlm_lam0")
OUT_TSV = OUT_DIR / "decoded_predictions.tsv"
OUT_JSON = OUT_DIR / "lam0_isolation.json"

SEED = 7
PAD = 4090
T_MAX = 64

EXCLUDED_IDS = [4090, 4091, 4092, 4093, 4094, 4095, 4096]
EXCLUDED_NAMES = ["<|bos|>", "<SAMPLE_START>", "<SAMPLE_END>",
                  "<READ_START>", "<READ_END>", "<PAIRED_END>", "[MASK]"]
EXCL_SET = set(EXCLUDED_IDS)


def decode_token(tk, tid):
    try:
        return tk.enc.decode([int(tid)])
    except Exception:
        return f"<id={int(tid)}>"


def safe_tsv(s):
    s = s.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
    if s == "" or s.isspace():
        return f"<sp:{len(s)}>"
    return s


def main():
    from device_utils import get_device
    device = get_device()
    tk = Tokenizer.from_directory()
    model, config = build_model(device, depth=2, aspect_ratio=128, seq_len=64)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[lam0] loaded {CHECKPOINT}  step={ckpt.get('step')}  last_losses={ckpt.get('last_losses')}")

    samples = parse_val_txt_grouped("output/val.txt", n_per_sample=50, seed=42)
    flat = []
    for s_idx in sorted(samples.keys()):
        for body in samples[s_idx]:
            flat.append((s_idx, body))
    print(f"[lam0] {len(flat)} val reads ({len(samples)} samples × 50)")

    torch.manual_seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- inference + TSV dump (parity with MVT decoded_predictions.tsv) ----
    n_rows = 0
    branch_counts = Counter()
    msk_dna_rows = []   # list of dicts for stats

    with open(OUT_TSV, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow([
            "read_idx", "sample_idx", "read_len", "pos", "branch",
            "input_id", "input_tok", "true_id", "true_tok",
            "top1_correct", "top5_correct",
            "top1_id", "top1_tok", "top1_prob",
            "top5_ids", "top5_toks", "top5_probs",
        ])

        t0 = time.time()
        for read_idx, (s_idx, body) in enumerate(flat, 1):
            text = f"<READ_START> {body} <READ_END>"
            ids = tk.encode(text)
            L = len(ids)
            padded = ids + [PAD] * (T_MAX - L) if L < T_MAX else ids[:T_MAX]
            tok = torch.tensor([padded], dtype=torch.long, device=device)
            attn_lens = torch.tensor([L], dtype=torch.long, device=device)
            masked, targets = mlm_mask(tok, vocab_size=EXTENDED_VOCAB, mask_prob=0.15)
            with torch.no_grad():
                logits, _ = model.forward_mlm(masked, attn_lens=attn_lens)
            probs = F.softmax(logits[0], dim=-1)

            for p in range(L):
                if int(targets[0, p].item()) == -1:
                    continue
                true_id = int(tok[0, p].item())
                in_id = int(masked[0, p].item())
                top5_p, top5_id_t = probs[p].topk(5)
                top5_ids = [int(x) for x in top5_id_t.tolist()]
                top5_probs = [float(x) for x in top5_p.tolist()]
                top5_texts = [decode_token(tk, t) for t in top5_ids]
                top1_id = top5_ids[0]
                top1_prob = top5_probs[0]
                top1_correct = int(top1_id == true_id)
                top5_correct = int(true_id in top5_ids)
                if in_id == MASK_ID:        branch = "MSK"
                elif in_id == true_id:       branch = "KEEP"
                else:                        branch = "RAND"
                branch_counts[branch] += 1

                # For CE compute: log p(true_id) under softmax
                true_logp = float(torch.log(probs[p, true_id].clamp(min=1e-12)).item())

                if branch == "MSK" and true_id not in EXCL_SET:
                    msk_dna_rows.append({
                        "top1_correct": top1_correct,
                        "top5_correct": top5_correct,
                        "true_logp": true_logp,
                        "true_id": true_id,
                    })

                w.writerow([
                    read_idx, s_idx, L, p, branch,
                    in_id, safe_tsv(decode_token(tk, in_id)),
                    true_id, safe_tsv(decode_token(tk, true_id)),
                    top1_correct, top5_correct,
                    top1_id, safe_tsv(decode_token(tk, top1_id)), f"{top1_prob:.4f}",
                    "|".join(str(t) for t in top5_ids),
                    "|".join(safe_tsv(t) for t in top5_texts),
                    "|".join(f"{p:.4f}" for p in top5_probs),
                ])
                n_rows += 1

        print(f"[lam0] inference done in {time.time()-t0:.1f}s -> {OUT_TSV}  ({OUT_TSV.stat().st_size/1024:.1f} KB)")

    # ---- scoreboard ----
    print(f"[lam0] branch composition: {dict(branch_counts)}")
    n_msk_dna = len(msk_dna_rows)
    n_top1 = sum(r["top1_correct"] for r in msk_dna_rows)
    n_top5 = sum(r["top5_correct"] for r in msk_dna_rows)
    ce_nats = float(-np.mean([r["true_logp"] for r in msk_dna_rows]))
    top1 = n_top1 / n_msk_dna
    top5 = n_top5 / n_msk_dna
    print(f"[lam0] MSK DNA n={n_msk_dna}  top1={top1:.4f}  top5={top5:.4f}  CE={ce_nats:.4f} nats")

    # ---- verdict comparison ----
    unigram_top1 = 0.0923
    unigram_ce = 6.7304
    mvt_top1 = 0.1085
    mvt_top5 = 0.1547
    mvt_train_mlm_ce = 5.7094

    lam0_gap_top1 = top1 - unigram_top1
    mvt_gap_top1 = mvt_top1 - unigram_top1

    if abs(top1 - mvt_top1) < 0.005:
        verdict = "neutral"
    elif top1 > mvt_top1 + 0.005:
        verdict = "hurting"   # MVT had contrastive; removing it improves MLM ⇒ contrastive was hurting
    else:
        verdict = "helping"   # MVT did better; removing contrastive degrades MLM ⇒ contrastive was helping

    summary = {
        "checkpoint": CHECKPOINT,
        "checkpoint_step": ckpt.get("step"),
        "training": {
            "args_changed_from_mvt": {"--lam": "1.0 -> 0"},
            "args_replicated_from_mvt": {
                "--max-steps": 20000, "--eval-every": 2000,
                "--max-runtime-hours": 12, "--log-every": 200,
                "--samples-per-batch": 4, "--reads-per-sample": 8,
                "--reads-cap": 200, "--seq-len": 64,
                "--depth": 2, "--aspect-ratio": 128,
                "--seed": 42,
            },
            "step_0_losses": list(ckpt.get("init_losses", [None, None])),
            "step_final_losses": list(ckpt.get("last_losses", [None, None])),
            "wall_clock_estimate": "~24 minutes (clean, no thermal throttling)",
        },
        "decode_eval": {
            "n_val_reads": len(flat),
            "n_total_masked_positions": n_rows,
            "branch_counts": dict(branch_counts),
            "excluded_token_ids": EXCLUDED_IDS,
            "excluded_token_names": EXCLUDED_NAMES,
            "msk_dna_n_positions": n_msk_dna,
            "msk_dna_top1": top1,
            "msk_dna_top5": top5,
            "msk_dna_ce_nats": ce_nats,
        },
        "comparison": {
            "unigram_baseline": {"top1": unigram_top1, "ce_nats": unigram_ce},
            "mvt_lam_1.0":     {"top1": mvt_top1, "top5": mvt_top5,
                                "training_mlm_ce_at_step_19999": mvt_train_mlm_ce},
            "lam_0":           {"top1": top1, "top5": top5, "ce_nats": ce_nats},
            "lam0_gap_top1_minus_unigram": lam0_gap_top1,
            "mvt_gap_top1_minus_unigram": mvt_gap_top1,
            "delta_lam0_minus_mvt_top1": top1 - mvt_top1,
        },
        "verdict": {
            "label": verdict,
            "interpretation": {
                "hurting": "Pure MLM beats MVT on MSK-DNA top-1 — contrastive was draining capacity away from the MLM task.",
                "neutral": "Pure MLM ≈ MVT on MSK-DNA top-1 — contrastive was neither helping nor hurting MLM within-read prediction.",
                "helping": "MVT beats pure MLM on MSK-DNA top-1 — contrastive was a useful regularizer for MLM.",
            }[verdict],
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[lam0] wrote {OUT_JSON}  ({OUT_JSON.stat().st_size} bytes)")

    print()
    print("=" * 70)
    print("VERDICT TABLE")
    print("=" * 70)
    print(f"{'':<25}{'top1':>10}{'top5':>10}{'CE(nats)':>14}")
    print(f"{'unigram baseline':<25}{unigram_top1:>10.4f}{'--':>10}{unigram_ce:>14.4f}")
    print(f"{'MVT  (λ=1.0)':<25}{mvt_top1:>10.4f}{mvt_top5:>10.4f}{mvt_train_mlm_ce:>14.4f}  (training)")
    print(f"{'THIS (λ=0)':<25}{top1:>10.4f}{top5:>10.4f}{ce_nats:>14.4f}  (val MSK-DNA)")
    print(f"{'':<25}")
    print(f"gap vs unigram (λ=0):     {lam0_gap_top1:+.4f}    -> "
          f"{'WIDER than' if lam0_gap_top1 > mvt_gap_top1 + 0.002 else 'narrower than' if lam0_gap_top1 < mvt_gap_top1 - 0.002 else 'same as'} "
          f"MVT's {mvt_gap_top1:+.4f}")
    print(f"verdict: contrastive was {verdict.upper()} the within-read MLM")


if __name__ == "__main__":
    main()
