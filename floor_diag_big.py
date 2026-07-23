"""
Capacity check: λ=0 with depth=4, aspect_ratio=192 (41M params, 8.5× the 4.82M baseline).
Same decode methodology as floor_diag_lam0.py — only model size differs.
Writes experiments/mlm_big/capacity_check.json.
"""
from __future__ import annotations

import csv, json, math, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train_mlm import GPT_MLM, build_model, mlm_mask, EXTENDED_VOCAB, MASK_ID
from prepare_genomic import Tokenizer
from probe_sample_coherence import parse_val_txt_grouped

CHECKPOINT = "experiments/mlm_big/checkpoint.pt"
OUT_DIR = Path("experiments/mlm_big")
OUT_TSV = OUT_DIR / "decoded_predictions.tsv"
OUT_JSON = OUT_DIR / "capacity_check.json"
DEPTH, ASPECT_RATIO = 4, 192
SEED = 7
PAD = 4090
T_MAX = 64
EXCLUDED_IDS = [4090, 4091, 4092, 4093, 4094, 4095, 4096]
EXCLUDED_NAMES = ["<|bos|>", "<SAMPLE_START>", "<SAMPLE_END>",
                  "<READ_START>", "<READ_END>", "<PAIRED_END>", "[MASK]"]
EXCL_SET = set(EXCLUDED_IDS)


def decode_token(tk, tid):
    try: return tk.enc.decode([int(tid)])
    except Exception: return f"<id={int(tid)}>"


def safe_tsv(s):
    s = s.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
    return f"<sp:{len(s)}>" if (s == "" or s.isspace()) else s


def main():
    from device_utils import get_device
    device = get_device()
    tk = Tokenizer.from_directory()
    model, config = build_model(device, depth=DEPTH, aspect_ratio=ASPECT_RATIO, seq_len=64)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[big] checkpoint: step={ckpt.get('step')} last_losses={ckpt.get('last_losses')}")
    print(f"[big] model: depth={DEPTH} ar={ASPECT_RATIO} n_embd={config.n_embd} params={n_params/1e6:.2f}M")

    samples = parse_val_txt_grouped("output/val.txt", n_per_sample=50, seed=42)
    flat = [(s, b) for s in sorted(samples.keys()) for b in samples[s]]
    print(f"[big] {len(flat)} val reads")

    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    branch_counts = Counter()
    msk_dna_rows = []

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
                if int(targets[0, p].item()) == -1: continue
                true_id = int(tok[0, p].item())
                in_id = int(masked[0, p].item())
                top5_p, top5_id_t = probs[p].topk(5)
                top5_ids = [int(x) for x in top5_id_t.tolist()]
                top5_probs = [float(x) for x in top5_p.tolist()]
                top5_texts = [decode_token(tk, t) for t in top5_ids]
                top1_id = top5_ids[0]; top1_prob = top5_probs[0]
                top1_correct = int(top1_id == true_id)
                top5_correct = int(true_id in top5_ids)
                if in_id == MASK_ID:        branch = "MSK"
                elif in_id == true_id:       branch = "KEEP"
                else:                        branch = "RAND"
                branch_counts[branch] += 1
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
        print(f"[big] inference done in {time.time()-t0:.1f}s -> {OUT_TSV} ({OUT_TSV.stat().st_size/1024:.1f} KB)")

    print(f"[big] branch composition: {dict(branch_counts)}")
    n_msk_dna = len(msk_dna_rows)
    n_top1 = sum(r["top1_correct"] for r in msk_dna_rows)
    n_top5 = sum(r["top5_correct"] for r in msk_dna_rows)
    ce_nats = float(-np.mean([r["true_logp"] for r in msk_dna_rows]))
    top1 = n_top1 / n_msk_dna
    top5 = n_top5 / n_msk_dna
    print(f"[big] MSK DNA n={n_msk_dna} top1={top1:.4f} top5={top5:.4f} CE={ce_nats:.4f} nats")

    # ---- comparison & verdict ----
    unigram_top1, unigram_ce = 0.0923, 6.7304
    lam0_top1, lam0_top5, lam0_ce = 0.1090, 0.1521, 6.2701
    lam0_params_M = 4.82

    # Was the run still climbing at the cap? Check trajectory MLM-loss slope.
    traj = []
    with open(OUT_DIR / "probe_trajectory.csv") as f:
        for r in csv.DictReader(f):
            traj.append({
                "step": int(r["step"]), "wall_s": float(r["wall_clock_s"]),
                "L_MLM": float(r["L_MLM"]), "probe1_auc": float(r["probe1_auc"]),
            })
    # Last 5 evals' MLM means
    if len(traj) >= 5:
        last5 = [t["L_MLM"] for t in traj[-5:]]
        first5 = [t["L_MLM"] for t in traj[:5]]
        still_descending = last5[-1] < min(last5[:-1]) * 0.97 or (sum(first5)/5 - sum(last5)/5) > 0.5
    else:
        still_descending = False

    if top1 - lam0_top1 > 0.01:
        if still_descending:
            verdict = "capacity is the lever (still climbing — extend to confirm asymptote)"
            label = "the lever / undertrained"
        else:
            verdict = "capacity is THE LEVER — bigger model raises MSK-DNA top-1 meaningfully"
            label = "the lever"
    elif abs(top1 - lam0_top1) <= 0.01:
        if still_descending:
            verdict = "inconclusive — bigger model trained better on MLM loss but MSK-DNA top-1 hasn't moved yet; still descending"
            label = "inconclusive-still-climbing"
        else:
            verdict = "capacity is NOT the lever — bigger model trained to lower loss but did not raise MSK-DNA top-1; the ceiling looks intrinsic to read-level prediction at this protocol"
            label = "NOT the lever"
    else:
        verdict = "bigger model UNDERPERFORMS the baseline — implausible without a config bug; report and stop"
        label = "regression"

    msk_dna_trajectory_note = ("MSK-DNA top-1 trajectory per eval is not measured — the periodic eval logs Probe 1 AUC, "
                               "not MSK-DNA top-1. We have one MSK-DNA top-1 number per checkpoint (this final). "
                               "If you want per-step MSK-DNA top-1 you'd need to re-run the decoder on best.pt/latest.pt too.")

    summary = {
        "checkpoint": CHECKPOINT,
        "checkpoint_step": ckpt.get("step"),
        "model": {
            "depth": DEPTH, "aspect_ratio": ASPECT_RATIO,
            "n_embd": config.n_embd, "n_head": config.n_head,
            "total_params_M": n_params / 1e6,
            "fold_vs_lam0_baseline": (n_params / 1e6) / lam0_params_M,
        },
        "args_changed_from_lam0": {"--depth": "2 -> 4", "--aspect-ratio": "128 -> 192",
                                   "--max-steps": "20000 -> 40000"},
        "args_replicated_from_lam0": {
            "--eval-every": 2000, "--max-runtime-hours": 12, "--log-every": 200,
            "--samples-per-batch": 4, "--reads-per-sample": 8, "--reads-cap": 200,
            "--seq-len": 64, "--lam": 0, "--seed": 42,
            "train-val-paths": {"train": "output/train.txt", "val": "output/val.txt"},
        },
        "training": {
            "step_0_losses": list(ckpt.get("init_losses", [None, None])),
            "step_final_losses": list(ckpt.get("last_losses", [None, None])),
            "wall_clock_s": traj[-1]["wall_s"] if traj else None,
            "ms_per_step_cum_final_log": 321,  # from the stdout log
            "probe_trajectory": traj,
            "still_descending_at_cap": still_descending,
            "msk_dna_trajectory_note": msk_dna_trajectory_note,
        },
        "decode_eval": {
            "n_val_reads": len(flat),
            "n_total_masked_positions": n_rows,
            "branch_counts": dict(branch_counts),
            "excluded_token_ids": EXCLUDED_IDS,
            "excluded_token_names": EXCLUDED_NAMES,
            "msk_dna_n_positions": n_msk_dna,
            "msk_dna_top1": top1, "msk_dna_top5": top5, "msk_dna_ce_nats": ce_nats,
        },
        "comparison": {
            "unigram":      {"params_M": 0, "top1": unigram_top1, "ce_nats": unigram_ce},
            "lam0_depth2":  {"params_M": lam0_params_M, "top1": lam0_top1, "top5": lam0_top5, "ce_nats": lam0_ce},
            "big_depth4":   {"params_M": n_params / 1e6, "top1": top1, "top5": top5, "ce_nats": ce_nats},
            "delta_big_minus_lam0_top1": top1 - lam0_top1,
            "delta_big_minus_lam0_ce_nats": ce_nats - lam0_ce,
        },
        "verdict": {"label": label, "interpretation": verdict},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[big] wrote {OUT_JSON} ({OUT_JSON.stat().st_size} bytes)")

    print()
    print("=" * 78)
    print("VERDICT TABLE")
    print("=" * 78)
    print(f"{'':<30}{'params':>10}{'top1':>10}{'top5':>10}{'CE(nats)':>14}")
    print(f"{'unigram baseline':<30}{'--':>10}{unigram_top1:>10.4f}{'--':>10}{unigram_ce:>14.4f}")
    print(f"{'depth-2 (λ=0)':<30}{f'{lam0_params_M:.2f}M':>10}{lam0_top1:>10.4f}{lam0_top5:>10.4f}{lam0_ce:>14.4f}")
    print(f"{'THIS bigger model (λ=0)':<30}{f'{n_params/1e6:.2f}M':>10}{top1:>10.4f}{top5:>10.4f}{ce_nats:>14.4f}")
    print()
    print(f"Δ vs λ=0:  top1 {top1-lam0_top1:+.4f}   top5 {top5-lam0_top5:+.4f}   CE {ce_nats-lam0_ce:+.4f} nats")
    print()
    print("Probe1 AUC trajectory (already noted out of scope for this verdict):")
    for t in traj:
        print(f"  step {t['step']:>5}  wall {t['wall_s']:>7.0f}s  L_MLM={t['L_MLM']:.4f}  probe1_auc={t['probe1_auc']:.4f}")
    print()
    print(f"MLM loss still descending at step-cap? {still_descending}")
    print(f"verdict: {label}")
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
