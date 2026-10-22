#!/usr/bin/env python3
"""ab_noise_centered.py — AUDIT ONLY. Centred noise evaluation on the two D2 checkpoints.
Every cosine reported here is CENTRED. Between-read floor reported alongside."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); import os; os.chdir(REPO)

from train_mlm import build_model
from prepare_genomic import Tokenizer
from evaluate_probes import (sample_val_reads, add_substitution_noise, embed_read,
                             cosine, ERROR_RATES, RNG_SEED)

VAL = Path("output/val.txt"); N = 400   # 400 reads keeps this inside the 10-min budget

def special_ids(tk):
    ids = set()
    for n in ("<|bos|>","<SAMPLE_START>","<SAMPLE_END>","<READ_START>","<READ_END>","<PAIRED_END>"):
        try: ids.add(tk.enc.encode_single_token(n))
        except Exception: pass
    return ids

def run(ckpt_path, tk, sp, device, raw):
    model, _ = build_model(device, depth=2, aspect_ratio=128, seq_len=64)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    clean = np.stack([embed_read(model, tk, r, device, sp) for r in raw])
    mu = clean.mean(axis=0, keepdims=True)
    cc = clean - mu
    perm = np.random.default_rng(0).permutation(len(cc))
    floor = float(np.mean([cosine(cc[i], cc[j]) for i, j in zip(range(len(cc)), perm) if i != j]))
    out = {"checkpoint": str(ckpt_path), "step": int(ck.get("step", -1)),
           "centered_between_read_floor": floor, "centered": {}}
    for er in ERROR_RATES:
        rng = np.random.default_rng(RNG_SEED + int(er*1_000_000))
        noisy = np.stack([embed_read(model, tk, add_substitution_noise(r, er, rng), device, sp) for r in raw])
        out["centered"][str(er)] = float(np.mean([cosine(cc[i], noisy[i]-mu[0]) for i in range(len(raw))]))
    return out

def main():
    from device_utils import get_device
    device = torch.device(get_device())
    tk = Tokenizer.from_directory(); sp = special_ids(tk)
    raw = sample_val_reads(VAL, N, RNG_SEED)
    results = []
    for name, p in (("SMALL-CAP (cap 100)", "experiments/ab_smallcap/checkpoint.pt"),
                    ("LARGE-CAP (cap 1000)", "experiments/ab_largecap/checkpoint.pt")):
        p = Path(p)
        if not p.exists():
            print(f"{name}: checkpoint missing at {p} — skipped"); continue
        r = run(p, tk, sp, device, raw); r["name"] = name; results.append(r)
    print(f"\nCENTRED noise evaluation ({N} reads). All cosines centred; floor = mean cosine between DIFFERENT reads.\n")
    print(f"{'run':<22}{'floor':>9}{'0.5%':>9}{'1%':>9}{'5%':>9}{'margin@5%':>12}")
    for r in results:
        c = r["centered"]
        print(f"{r['name']:<22}{r['centered_between_read_floor']:>9.4f}"
              f"{c['0.005']:>9.4f}{c['0.01']:>9.4f}{c['0.05']:>9.4f}"
              f"{c['0.05']-r['centered_between_read_floor']:>+12.4f}")
    Path("audit/ab_noise_centered.json").write_text(json.dumps(results, indent=2))
    print("\nsaved -> audit/ab_noise_centered.json")

if __name__ == "__main__":
    main()
