#!/usr/bin/env python3
"""
rerun_centered.py — AUDIT ONLY. Re-runs the two cosine-based evaluations on the best local
checkpoint, with and without mean-centering, and prints both side by side.

Follows probe_gate.py's pattern: builds GPT_MLM via train_mlm.build_model rather than calling
run_probe4 (whose checkpoint list is hardcoded to CLM-era runs).

Evaluation 1 = sample coherence  (probe_sample_coherence.py)
Evaluation 4 = noise robustness  (evaluate_probes.py primitives)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import os; os.chdir(REPO)

from train_mlm import build_model
from prepare_genomic import Tokenizer
import probe_sample_coherence as P1
from evaluate_probes import (sample_val_reads, add_substitution_noise, embed_read,
                             cosine, center_embeddings, ERROR_RATES, RNG_SEED)

CKPT = Path("experiments/mlm_bigcohort_d6/checkpoint.pt")
VAL = Path("output/val.txt")
N_READS_P4 = 1000
N_PER_SAMPLE_P1 = 50


def load():
    from device_utils import get_device
    device = torch.device(get_device())
    model, cfg = build_model(device, depth=6, aspect_ratio=192, seq_len=64)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    tk = Tokenizer.from_directory()
    return model, tk, device, ck


def special_ids(tk):
    ids = set()
    for n in ("<|bos|>", "<SAMPLE_START>", "<SAMPLE_END>", "<READ_START>", "<READ_END>", "<PAIRED_END>"):
        try: ids.add(tk.enc.encode_single_token(n))
        except Exception: pass
    return ids


def main():
    model, tk, device, ck = load()
    sp = special_ids(tk)
    print(f"checkpoint : {CKPT}  (step {ck.get('step','?')})")
    print(f"device     : {device}\n")

    # ---------- Evaluation 1 ----------
    grouped = P1.parse_val_txt_grouped(VAL, n_per_sample=N_PER_SAMPLE_P1, seed=42)
    reads, labels = [], []
    for s, rs in grouped.items():
        for r in rs:
            reads.append(r); labels.append(s)
    labels = np.array(labels)
    t0 = time.time()
    emb = P1.embed_reads(model, tk, reads, device, sp)
    print(f"Evaluation 1: embedded {len(reads)} reads in {time.time()-t0:.1f}s\n")

    e1 = {}
    for mode, flag in (("uncentered", False), ("centered", True)):
        e1[mode] = P1.compute_sample_coherence(emb, labels, seed=42, center=flag)

    print("EVALUATION 1 — sample coherence")
    print(f"{'':<14}{'AUC':>10}{'within cos':>13}{'between cos':>13}{'delta':>12}")
    for mode in ("uncentered", "centered"):
        r = e1[mode]
        print(f"{mode:<14}{r['auc']:>10.4f}{r['mean_within_cos']:>13.4f}"
              f"{r['mean_between_cos']:>13.4f}{r['delta_mean']:>+12.5f}")

    # ---------- Evaluation 4 ----------
    print("\nEVALUATION 4 — noise robustness")
    raw = sample_val_reads(VAL, N_READS_P4, RNG_SEED)
    t0 = time.time()
    clean = np.stack([embed_read(model, tk, r, device, sp) for r in raw])
    print(f"  embedded {len(raw)} clean reads in {time.time()-t0:.1f}s")
    mean_vec = clean.mean(axis=0, keepdims=True)

    def floor(X):
        """Mean cosine between DIFFERENT reads — the baseline a same-read score must beat."""
        idx = np.random.default_rng(0).permutation(len(X))
        return float(np.mean([cosine(X[i], X[j]) for i, j in zip(range(len(X)), idx) if i != j]))

    f_un, f_ce = floor(clean), floor(clean - mean_vec)
    print(f"  between-read floor : uncentered {f_un:.4f}   centered {f_ce:.4f}")
    print(f"\n{'error rate':<12}{'uncen cos':>12}{'uncen margin':>15}{'cen cos':>12}{'cen margin':>13}")
    for er in ERROR_RATES:
        rng = np.random.default_rng(RNG_SEED + int(er * 1_000_000))
        noisy = np.stack([embed_read(model, tk, add_substitution_noise(r, er, rng), device, sp) for r in raw])
        cu = float(np.mean([cosine(clean[i], noisy[i]) for i in range(len(raw))]))
        cc = float(np.mean([cosine(clean[i] - mean_vec[0], noisy[i] - mean_vec[0]) for i in range(len(raw))]))
        print(f"{er:<12}{cu:>12.4f}{cu-f_un:>+15.4f}{cc:>12.4f}{cc-f_ce:>+13.4f}")

    out = Path("audit/centered_rerun.json")
    out.write_text(json.dumps({"checkpoint": str(CKPT),
                               "evaluation1": {k: {kk: vv for kk, vv in v.items()
                                                   if isinstance(vv, (int, float, str, bool))}
                                               for k, v in e1.items()},
                               "evaluation4_floor": {"uncentered": f_un, "centered": f_ce}}, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
