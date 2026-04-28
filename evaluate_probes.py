"""
HARMONY representation probes on saved checkpoints.

Currently implements Probe 4 (synthetic noise robustness) only. Probes 1, 2,
and 3 are deferred pending the val-split disposition decision (see program.md
Section 5, Concern 2). Probe 4 is split-independent — it tests intra-read
invariance — so it runs cleanly on the existing data.

Usage:
    .venv/bin/python3.12 evaluate_probes.py            # run probe 4 on all 4 checkpoints
    .venv/bin/python3.12 evaluate_probes.py --probe 4  # explicit
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from prepare_genomic import Tokenizer
from model import GPT, GPTConfig

PROJECT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
VAL_PATH = PROJECT_DIR / "output" / "val.txt"

# (display_name, original_val_bpb, checkpoint_path)
CHECKPOINTS = [
    ("baseline_rerun",       1.951289, EXPERIMENTS_DIR / "repro_baseline_rerun"       / "checkpoint.pt"),
    ("lower_matrix_lr",      1.935718, EXPERIMENTS_DIR / "repro_lower_matrix_lr"      / "checkpoint.pt"),
    ("r2_lowlr_warmup_wd01", 1.932465, EXPERIMENTS_DIR / "repro_r2_lowlr_warmup_wd01" / "checkpoint.pt"),
    ("r3_lr_025",            1.950751, EXPERIMENTS_DIR / "repro_r3_lr_025"            / "checkpoint.pt"),
]

ERROR_RATES = (0.005, 0.01, 0.05)
N_READS = 1000
RNG_SEED = 42

BASES = ("A", "C", "G", "T")
READ_RE = re.compile(r"<READ_START>\s+([ACGTN]+)\s+<READ_END>")


# ---------------------------------------------------------------------------
# Read sampling and noise
# ---------------------------------------------------------------------------

def sample_val_reads(path: Path, n: int, seed: int) -> list[str]:
    """Find <READ_START>...<READ_END> bodies in val.txt and sample n of them."""
    rng = random.Random(seed)
    text = path.read_text()
    matches = READ_RE.findall(text)
    if len(matches) < n:
        raise RuntimeError(f"val.txt has only {len(matches)} reads, need {n}")
    indices = rng.sample(range(len(matches)), n)
    return [matches[i] for i in indices]


def add_substitution_noise(seq: str, error_rate: float, rng: np.random.Generator) -> str:
    """Per-base substitution: with prob error_rate, replace with a different base from {A,C,G,T}.
    N bases are left untouched (they're already ambiguous)."""
    arr = np.array(list(seq))
    is_acgt = np.isin(arr, BASES)
    flip_mask = (rng.random(len(arr)) < error_rate) & is_acgt
    out = arr.copy()
    flip_idx = np.flatnonzero(flip_mask)
    for i in flip_idx:
        original = arr[i]
        choices = [b for b in BASES if b != original]
        out[i] = rng.choice(choices)
    return "".join(out)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_read(model: GPT, tokenizer: Tokenizer, dna: str, device: torch.device,
               read_start_id: int, read_end_id: int) -> np.ndarray | None:
    """Tokenize <READ_START> dna <READ_END>, forward, mean-pool over DNA positions."""
    text = f"<READ_START> {dna} <READ_END>"
    ids = tokenizer.encode(text)  # list[int]
    mask = [(t != read_start_id and t != read_end_id) for t in ids]
    if not any(mask):
        return None
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        h = model(x, return_hidden=True)  # (1, T, n_embd)
    h = h[0]  # (T, n_embd)
    mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
    pooled = h[mask_t].float().mean(dim=0)  # cast to fp32 for cosine
    return pooled.cpu().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path, device: torch.device) -> tuple[GPT, GPTConfig, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    with torch.device("meta"):
        m = GPT(cfg)
    m.to_empty(device=device)
    m.init_weights()  # sets dtypes (wte/value_embeds → bf16) and rotary buffers
    missing, unexpected = m.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        # Non-persistent buffers (cos, sin) are excluded from state_dict by design
        non_buffer_missing = [k for k in missing if k not in ("cos", "sin")]
        if non_buffer_missing or unexpected:
            raise RuntimeError(f"state_dict mismatch: missing={non_buffer_missing} unexpected={unexpected}")
    m.eval()
    return m, cfg, ckpt


# ---------------------------------------------------------------------------
# Probe 4
# ---------------------------------------------------------------------------

def run_probe4(checkpoints: list[tuple[str, float, Path]]) -> tuple[list[dict], list[dict]]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    tokenizer = Tokenizer.from_directory()
    rs_id = tokenizer.enc.encode_single_token("<READ_START>")
    re_id = tokenizer.enc.encode_single_token("<READ_END>")

    print(f"Sampling {N_READS} reads from {VAL_PATH}…")
    t0 = time.time()
    reads = sample_val_reads(VAL_PATH, N_READS, RNG_SEED)
    print(f"  done in {time.time()-t0:.1f}s; mean read length = {np.mean([len(r) for r in reads]):.1f} bp")

    # Generate noise once (deterministic seed) and reuse across checkpoints
    rng = np.random.default_rng(RNG_SEED)
    noisy_versions: dict[float, list[str]] = {}
    for er in ERROR_RATES:
        # Reset rng per error rate so noise pattern is independent of error-rate iteration order
        sub_rng = np.random.default_rng(RNG_SEED + int(er * 1_000_000))
        noisy_versions[er] = [add_substitution_noise(r, er, sub_rng) for r in reads]

    rows: list[dict] = []
    headline_rows: list[dict] = []  # for summary table at error_rate=0.01

    for ckpt_name, val_bpb, ckpt_path in checkpoints:
        if not ckpt_path.exists():
            print(f"\n!! SKIP {ckpt_name}: checkpoint missing at {ckpt_path}")
            continue
        print(f"\nLoading {ckpt_name} from {ckpt_path}…")
        t_load = time.time()
        model, cfg, ckpt = load_checkpoint(ckpt_path, device)
        print(f"  loaded in {time.time()-t_load:.1f}s; n_embd={cfg.n_embd}, depth={cfg.n_layer}")
        ckpt_val_bpb = ckpt.get("val_bpb", val_bpb)
        if abs(ckpt_val_bpb - val_bpb) > 0.01:
            print(f"  NOTE: checkpoint reports val_bpb={ckpt_val_bpb:.6f} (vs declared {val_bpb:.6f})")

        # Embed clean reads
        print(f"  embedding {len(reads)} clean reads…")
        t_clean = time.time()
        clean_emb = [embed_read(model, tokenizer, r, device, rs_id, re_id) for r in reads]
        print(f"    done in {time.time()-t_clean:.1f}s")

        for er in ERROR_RATES:
            t_er = time.time()
            sims = []
            for i, r_noisy in enumerate(noisy_versions[er]):
                e_n = embed_read(model, tokenizer, r_noisy, device, rs_id, re_id)
                if clean_emb[i] is None or e_n is None:
                    continue
                sims.append(cosine(clean_emb[i], e_n))
            sims_arr = np.array(sims, dtype=np.float64)
            row = {
                "checkpoint": ckpt_name,
                "error_rate": er,
                "cos_mean": float(np.mean(sims_arr)),
                "cos_median": float(np.median(sims_arr)),
                "cos_std": float(np.std(sims_arr)),
                "n_reads": int(len(sims_arr)),
            }
            rows.append(row)
            print(f"  err={er*100:>4.1f}%  cos_mean={row['cos_mean']:.4f}  median={row['cos_median']:.4f}  std={row['cos_std']:.4f}  n={row['n_reads']}  ({time.time()-t_er:.1f}s)")
            if abs(er - 0.01) < 1e-9:
                headline_rows.append({
                    "checkpoint": ckpt_name,
                    "val_bpb": val_bpb,
                    "probe4_cos_at_1pct": row["cos_mean"],
                })

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return rows, headline_rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path):
    fieldnames = ["checkpoint", "error_rate", "cos_mean", "cos_median", "cos_std", "n_reads"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {path}")


def write_summary_md(rows: list[dict], headline_rows: list[dict], path: Path):
    rows_by_ckpt: dict[str, dict[float, dict]] = {}
    for r in rows:
        rows_by_ckpt.setdefault(r["checkpoint"], {})[r["error_rate"]] = r

    headline_rows_sorted = sorted(headline_rows, key=lambda r: r["val_bpb"], reverse=True)

    lines: list[str] = []
    lines.append("# Probe 4 — Synthetic Noise Robustness")
    lines.append("")
    lines.append(f"_n_reads={N_READS}, seed={RNG_SEED}, error rates={list(ERROR_RATES)}_")
    lines.append("")
    lines.append("## Headline cross-checkpoint table (sorted worst → best val_bpb)")
    lines.append("")
    lines.append("| checkpoint | val_bpb | probe4_cos_at_1pct |")
    lines.append("|---|---|---|")
    for r in headline_rows_sorted:
        lines.append(f"| {r['checkpoint']} | {r['val_bpb']:.6f} | {r['probe4_cos_at_1pct']:.4f} |")
    lines.append("")
    lines.append("## Full results (mean cosine similarity to clean embedding)")
    lines.append("")
    lines.append("| checkpoint | err=0.5% | err=1% | err=5% |")
    lines.append("|---|---|---|---|")
    for ckpt_name, _, _ in CHECKPOINTS:
        if ckpt_name not in rows_by_ckpt:
            continue
        d = rows_by_ckpt[ckpt_name]
        cells = []
        for er in ERROR_RATES:
            if er in d:
                cells.append(f"{d[er]['cos_mean']:.4f} ± {d[er]['cos_std']:.4f}")
            else:
                cells.append("—")
        lines.append(f"| {ckpt_name} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")

    if len(headline_rows_sorted) >= 2:
        # Spearman-style monotonicity check between val_bpb (decreasing → better) and probe4 cos (increasing → more invariant)
        sorted_by_bpb = sorted(headline_rows, key=lambda r: r["val_bpb"])  # best first
        coses = [r["probe4_cos_at_1pct"] for r in sorted_by_bpb]
        monotonic_decreasing = all(coses[i] <= coses[i-1] for i in range(1, len(coses)))
        monotonic_increasing = all(coses[i] >= coses[i-1] for i in range(1, len(coses)))
        delta = max(coses) - min(coses)
        lines.append(f"Across the {len(headline_rows_sorted)} checkpoints, Probe 4 cos@1% spans "
                     f"{min(coses):.4f}–{max(coses):.4f} (Δ = {delta:.4f}). ")
        if delta < 0.005:
            lines.append("This range is essentially flat — Probe 4 does not discriminate among these checkpoints.")
        elif monotonic_decreasing:
            lines.append("Sorted from best (lowest) val_bpb to worst, Probe 4 cos@1% is monotonically *decreasing* — "
                         "i.e. better val_bpb → less noise-invariant. **val_bpb is anti-correlated with HARMONY-aligned "
                         "noise robustness on this corpus.**")
        elif monotonic_increasing:
            lines.append("Sorted from best (lowest) val_bpb to worst, Probe 4 cos@1% is monotonically *increasing* — "
                         "i.e. better val_bpb → more noise-invariant. **val_bpb tracks Probe 4 monotonically across "
                         "these four checkpoints.**")
        else:
            lines.append("The relationship between val_bpb and Probe 4 cos@1% is non-monotonic across the four "
                         "checkpoints (some swap pairs disagree).")
    lines.append("")
    lines.append("## Recommended next step")
    lines.append("")
    lines.append("- **If Probe 4 increases as val_bpb decreases monotonically:** val_bpb is a useful proxy. "
                 "Continue P1 (untouched-knob seeding) and P2 (architectural axes) hyperparameter rounds.")
    lines.append("- **If Probe 4 is flat or anti-correlated:** val_bpb does not track the goal. Pivot to MLM + "
                 "contrastive arm; further val_bpb optimization will not improve representation quality.")
    lines.append("")
    lines.append("Decision pending Nick review of the headline table above and probe 1/2/3 results once unblocked.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Probe 4 alone tests *intra-read invariance to substitution noise*, which is necessary but not "
                 "sufficient for HARMONY's bias-correction deliverable. Probes 1 (community discrimination) and "
                 "2 (sample retrieval, batch-stratified) are needed for full evaluation.")
    lines.append("- Substitution noise at sequence level changes BPE tokenization, so cosine drop reflects both "
                 "token-level and representation-level brittleness.")
    lines.append("- Reads were drawn from val.txt; given the file-pair-level split (program.md §5 Concern 2), "
                 "many sampled reads share an SRR sample with training. This is a benign concern for Probe 4 since "
                 "the comparison is clean-vs-noisy of the *same* read, not held-out generalization.")
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=int, default=4, choices=[4],
                        help="Which probe to run (only 4 implemented).")
    args = parser.parse_args()
    if args.probe != 4:
        raise SystemExit("Only probe 4 is implemented; probes 1/2/3 are deferred per program.md §6.")

    rows, headline_rows = run_probe4(CHECKPOINTS)
    if not rows:
        raise SystemExit("No checkpoints loaded — nothing to write.")
    write_csv(rows, EXPERIMENTS_DIR / "probe_noise_robustness.csv")
    write_summary_md(rows, headline_rows, EXPERIMENTS_DIR / "probe4_summary.md")


if __name__ == "__main__":
    main()
