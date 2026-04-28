# Probe 4 — Synthetic Noise Robustness

_Run 2026-04-28. n_reads=1000 (sampled from val.txt with seed=42), error rates 0.5% / 1% / 5% substitution applied at the DNA-sequence level (pre-tokenization). Embeddings = mean-pool of final transformer hidden state across DNA token positions. Cosine similarity computed between clean and noisy embeddings of each read._

---

## Headline cross-checkpoint table

Sorted by original val_bpb (best → worst).

| checkpoint | val_bpb (original) | val_bpb (reproduced) | probe4_cos_at_1pct |
|---|---|---|---|
| r2_lowlr_warmup_wd01 | **1.932465** | 1.938688 | 0.9931 |
| lower_matrix_lr | 1.935718 | **1.933531** | 0.9895 |
| r3_lr_025 | 1.950751 | 1.957877 | 0.9949 |
| baseline_rerun | 1.951289 | 1.935228 | **0.9969** |

(Bold = best in column.)

## Full results (mean ± std cosine similarity to clean embedding, 1000 reads)

| checkpoint | err=0.5% | err=1% | err=5% |
|---|---|---|---|
| baseline_rerun | 0.9985 ± 0.0029 | 0.9969 ± 0.0045 | 0.9876 ± 0.0081 |
| lower_matrix_lr | 0.9956 ± 0.0160 | 0.9895 ± 0.0335 | 0.9638 ± 0.0575 |
| r2_lowlr_warmup_wd01 | 0.9967 ± 0.0097 | 0.9931 ± 0.0152 | 0.9736 ± 0.0276 |
| r3_lr_025 | 0.9976 ± 0.0054 | 0.9949 ± 0.0094 | 0.9798 ± 0.0163 |

(Raw values in [`probe_noise_robustness.csv`](probe_noise_robustness.csv).)

## Interpretation

**Empirical pattern: lower val_bpb does not predict higher Probe 4 cos@1%. The relationship is weakly anti-correlated.**

Sorted by original val_bpb (best → worst), cos@1% reads: 0.9931, 0.9895, 0.9949, 0.9969. The two checkpoints with the *best* val_bpb (`r2_lowlr_warmup_wd01`, `lower_matrix_lr`) have cos@1% averaging 0.9913. The two *worst*-val_bpb checkpoints (`r3_lr_025`, `baseline_rerun`) average 0.9959. The "worse" cluster on val_bpb is the *more* noise-robust cluster on Probe 4.

A second pattern is visible in the variance: `lower_matrix_lr` has cos@1% std=0.034, ~7× the std of `baseline_rerun` (0.005). The R1-best checkpoint produces representations that are less stable to noise on a per-read basis, consistent with sharper / more confident token-level representations that fragment when BPE re-segments noisy DNA.

The full deltas across the four checkpoints are small in absolute terms (Δ_cos@1% = 0.0074 max, Δ_cos@5% = 0.024). All four checkpoints are >0.96 cos at 5% noise — the model has learned representations that are fairly noise-tolerant in absolute terms. The discriminative signal between checkpoints is small.

## Reproducibility caveat — read this before interpreting

We re-ran each of the four configurations with seed=42 to produce the saved checkpoints used by this probe. **Three of four reproductions drifted outside the ±0.005 tolerance, and the relative ranking changed substantially:**

| checkpoint | original | reproduced | Δ | within ±0.005 |
|---|---|---|---|---|
| baseline_rerun | 1.951289 | 1.935228 | **−0.016** | NO |
| lower_matrix_lr | 1.935718 | 1.933531 | −0.002 | YES |
| r2_lowlr_warmup_wd01 | 1.932465 | 1.938688 | **+0.006** | NO |
| r3_lr_025 | 1.950751 | 1.957877 | **+0.007** | NO |

Original ranking (best→worst): r2_lowlr_warmup_wd01, lower_matrix_lr, r3_lr_025, baseline_rerun.
Reproduced ranking (best→worst): lower_matrix_lr, baseline_rerun, r2_lowlr_warmup_wd01, r3_lr_025.

Mean absolute drift = 0.008 bpb. The full original separation between best (R2 winner) and worst (baseline_rerun) was 0.019 bpb — only ~2.4× the reproducibility noise. **MPS run-to-run variance is on the same order as the val_bpb improvements being claimed.** This is consistent with — and a separate signal of — the same conclusion the probe is testing: small val_bpb deltas in this regime are not load-bearing.

## Recommended next step

**Pivot toward an MLM + contrastive arm; do not invest further compute in P1 / P2 hyperparameter rounds aimed solely at reducing val_bpb.**

The probe gate as defined in [program.md](../program.md) §6 said: *"If Probe 4 is flat or anti-correlated [with val_bpb] → val_bpb does not track the goal, pivot to MLM+contrastive arm."* That is the result we got, with two reinforcing strands:

1. The Probe 4 cos@1% rank order does not align with the val_bpb rank order; the two best-val_bpb checkpoints are below the two worst-val_bpb checkpoints on cos@1%.
2. The val_bpb rank order is itself unstable across re-runs — at the resolution we're operating, "winner" assignments at this scale are noise-driven.

This is a single-probe, single-corpus, single-seed result; it should not be the *only* basis for the pivot decision. Probes 1, 2, 3 are still required for full evaluation, and depend on resolving the val-split paired-end leakage (program.md §5 Concern 2). But the probe-4 evidence cuts in one direction, not in favor of continued val_bpb optimization.

## Caveats and known gaps

- **Probe 4 alone is necessary, not sufficient.** It tests intra-read invariance to substitution noise; it does not test community discrimination (Probe 1) or sample retrieval / batch-artifact (Probe 2) or quality-feature decoding (Probe 3). The full gate is the four-probe table, not just this one row.
- **Substitution noise changes BPE tokenization.** A single-base substitution can re-segment a token boundary. The cosine drop measured here mixes representation-level brittleness with token-level brittleness. Disentangling these would require running the same probe on a model with byte-level tokenization, which is out of scope here.
- **Val_bpb numbers are inflated by paired-end leakage.** Per program.md §5 Concern 2, the train/val split is at file-pair level, not sample level. The val_bpb numbers above (both original and reproduced) are subject to this confound; absolute values are not citable. Probe 4 itself does not depend on the split (it is clean-vs-noisy on the *same* read), so the probe result is unaffected by the leakage.
- **Reads were drawn from val.txt for convenience.** The same probe could be run on train.txt with identical methodology; result would likely be similar since clean/noisy embeddings are computed on the same read.
- **Reproducibility noise is large relative to claimed improvements.** As shown above, MPS run-to-run variance ≈ ±0.008 bpb. Any future claim of a val_bpb improvement <~0.02 bpb should be treated with skepticism unless verified across at least 2-3 seeds.

## Files produced

- [`probe_noise_robustness.csv`](probe_noise_robustness.csv) — the raw 12-row (4 ckpts × 3 error rates) table.
- [`reproduction_summary.json`](reproduction_summary.json) — per-checkpoint reproduction metadata (original vs reproduced val_bpb, wall time, paths).
- [`repro_*/checkpoint.pt`](.) — the four `.pt` checkpoints (gitignored; ~23 MB each).
- [`evaluate_probes.py`](../evaluate_probes.py) — probe driver. Currently implements only Probe 4; Probes 1/2/3 stubs deferred pending val-split disposition.
