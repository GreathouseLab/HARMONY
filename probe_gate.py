"""
Probe 1 + Probe 4 gate on the MVT checkpoint.
Calls the existing probe helpers; does NOT modify probe scripts.

Writes experiments/mlm_mvt/probe_gate.json next to the floor diagnostics.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_mlm import GPT_MLM, build_model
from prepare_genomic import Tokenizer
from probe_sample_coherence import run_probe1
# Probe 4 helpers (do not import or call run_probe4 — its checkpoint list is hardcoded;
# we use the same building blocks with our own model)
from evaluate_probes import (
    sample_val_reads, add_substitution_noise, embed_read, cosine,
    ERROR_RATES, N_READS, RNG_SEED,
)


CHECKPOINT = "experiments/mlm_mvt/checkpoint.pt"
VAL_PATH = Path("output/val.txt")
OUT_DIR = Path("experiments/mlm_mvt")
OUT_JSON = OUT_DIR / "probe_gate.json"


def _load_mvt_model(device: str):
    """Build GPT_MLM with the trained config and load the MVT checkpoint.
    Mirrors how the periodic-eval path in train_mlm.train() did this."""
    model, config = build_model(device, depth=2, aspect_ratio=128, seq_len=64)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config, ckpt


def _collect_special_ids(tk: Tokenizer):
    """Same set used by both probe scripts (READ_START/READ_END/PAIRED_END/etc.)."""
    names = ("<|bos|>", "<SAMPLE_START>", "<SAMPLE_END>",
             "<READ_START>", "<READ_END>", "<PAIRED_END>")
    ids = set()
    for n in names:
        try:
            ids.add(tk.enc.encode_single_token(n))
        except Exception:
            pass
    return ids


def main():
    from device_utils import get_device
    device = get_device()
    print(f"[probe_gate] device={device}")
    tk = Tokenizer.from_directory()
    model, config, ckpt = _load_mvt_model(device)
    print(f"[probe_gate] loaded checkpoint step={ckpt.get('step')} from {CHECKPOINT}")
    print(f"            n_embd={config.n_embd}, depth={config.n_layer}, vocab={config.vocab_size}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================== Probe 1 ===========================
    print("\n=== Probe 1 (sample-coherence ROC-AUC) ===")
    t0 = time.time()
    p1_result = run_probe1(
        checkpoint_path=Path(CHECKPOINT),
        val_path=VAL_PATH,
        output_path=OUT_DIR / "probe1_gate.json",
        n_reads_per_sample=50,
        device=device,
        model=model,
        tokenizer=tk,
        ckpt_meta={
            "n_embd": config.n_embd, "n_layer": config.n_layer,
            "vocab_size": config.vocab_size, "seed": 42,
            "hyperparams": {}, "val_bpb": None, "git_sha": None,
        },
    )
    probe1_auc = float(p1_result.get("auc", float("nan")))
    print(f"[probe_gate] Probe 1 AUC = {probe1_auc:.4f}  (took {time.time()-t0:.1f}s)")

    # =========================== Probe 4 ===========================
    # Use the same building blocks as evaluate_probes.run_probe4 but with OUR loaded model.
    print(f"\n=== Probe 4 (noise-robustness cos@1%) ===")
    special_ids = _collect_special_ids(tk)
    print(f"[probe_gate] sampling {N_READS} val reads (seed={RNG_SEED})...")
    reads = sample_val_reads(VAL_PATH, N_READS, seed=RNG_SEED)
    print(f"  sampled {len(reads)} reads")

    print(f"[probe_gate] generating noisy variants at {ERROR_RATES}...")
    noisy_versions = {}
    for er in ERROR_RATES:
        sub_rng = np.random.default_rng(RNG_SEED + int(er * 1_000_000))
        noisy_versions[er] = [add_substitution_noise(r, er, sub_rng) for r in reads]

    print(f"[probe_gate] embedding {len(reads)} clean reads...")
    t_clean = time.time()
    clean_emb = [embed_read(model, tk, r, device, special_ids) for r in reads]
    print(f"  done in {time.time()-t_clean:.1f}s")

    probe4_rows = []
    cos_at_1pct = None
    for er in ERROR_RATES:
        t_er = time.time()
        sims = []
        for i, r_noisy in enumerate(noisy_versions[er]):
            e_n = embed_read(model, tk, r_noisy, device, special_ids)
            if clean_emb[i] is None or e_n is None:
                continue
            sims.append(cosine(clean_emb[i], e_n))
        sims_arr = np.array(sims, dtype=np.float64)
        row = {
            "error_rate": er,
            "cos_mean": float(sims_arr.mean()),
            "cos_median": float(np.median(sims_arr)),
            "cos_std": float(sims_arr.std()),
            "n_reads": int(len(sims_arr)),
        }
        probe4_rows.append(row)
        print(f"  err={er*100:>4.1f}%  cos_mean={row['cos_mean']:.4f}  median={row['cos_median']:.4f}  std={row['cos_std']:.4f}  n={row['n_reads']}  ({time.time()-t_er:.1f}s)")
        if abs(er - 0.01) < 1e-9:
            cos_at_1pct = row["cos_mean"]

    # =========================== Verdict ===========================
    def verdict(p1, p4):
        # p1 thresholds per brief
        if p1 is None or p1 != p1: return "indeterminate"
        # high probe1 OK?
        p1_above_gate = p1 > 0.55
        p1_collapse   = p1 <= 0.52
        # probe4 ~0.997 collapse signature
        p4_collapse   = (p4 is not None and p4 == p4 and p4 >= 0.99)
        if p1_above_gate and not p4_collapse:
            return "above gate"
        if p1_collapse and p4_collapse:
            return "collapse"
        # mixed includes: low p1 but moderate p4, or borderline p1 with any p4
        return "mixed"

    v = verdict(probe1_auc, cos_at_1pct)

    print()
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    print(f"probe1 AUC={probe1_auc:.4f}  (bar 0.55; <=0.52 collapse)")
    print(f"probe4 cos@1%={cos_at_1pct:.4f}  (~0.997 = collapse)")
    print(f"verdict: {v}")

    out = {
        "checkpoint": CHECKPOINT,
        "checkpoint_step": ckpt.get("step"),
        "device": device,
        "probe1": {
            "auc": probe1_auc,
            "delta_mean": p1_result.get("delta_mean"),
            "mean_within_cos": p1_result.get("mean_within_cos"),
            "mean_between_cos": p1_result.get("mean_between_cos"),
            "n_within_pairs": p1_result.get("n_within_pairs"),
            "n_between_pairs": p1_result.get("n_between_pairs"),
            "n_reads_per_sample": 50,
            "val_path": str(VAL_PATH),
            "script_used": "probe_sample_coherence.run_probe1",
            "json_artifact": str(OUT_DIR / "probe1_gate.json"),
        },
        "probe4": {
            "rows": probe4_rows,
            "cos_at_1pct": cos_at_1pct,
            "n_reads": N_READS,
            "error_rates": list(ERROR_RATES),
            "rng_seed": RNG_SEED,
            "script_used": "evaluate_probes helpers (sample_val_reads, add_substitution_noise, embed_read, cosine)",
            "note": "evaluate_probes.run_probe4 has a hardcoded checkpoint list; we called its helpers directly without modifying the script.",
        },
        "verdict": v,
        "thresholds": {
            "probe1_above_gate": "> 0.55",
            "probe1_collapse": "<= 0.52",
            "probe4_collapse_signature": "cos@1% >= 0.99 (≈0.997 reported as collapse)",
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_JSON} ({OUT_JSON.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
