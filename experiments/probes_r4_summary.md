# R4 Probe Panel — Sample Coherence + Noise Robustness

_Run 2026-05-06 on the 3 R4 leak-free checkpoints. Probe 1 (sample-coherence): 100 reads × 14 val samples (= 1400 reads), 50k within-pair vs 50k between-pair cosines, ROC-AUC. Probe 4 (substitution-noise robustness): 1000 reads, error rates 0.5%/1%/5%, cos-sim of clean vs noisy embeddings._

## Headline table — R4 (leak-free, sorted best → worst val_bpb)

| checkpoint | val_bpb | probe1_AUC | probe1_Δmean | probe4_cos@1% |
|---|---|---|---|---|
| r4_baseline | **1.939903** | 0.5226 | +0.0066 | **0.9974** |
| r4_baseline_curve | 1.939989 | 0.5224 | +0.0066 | 0.9974 |
| r4_r3_winner | 1.948008 | **0.5242** | **+0.0084** | 0.9969 |

(Bold = best in column.)

## What the columns say in isolation

- **Probe 1 AUC ∈ [0.5224, 0.5242]** — all three checkpoints are essentially at chance (0.50). The trained model has only a faint sample-coherence signal (~2 percentage points above random). For reference: an untrained random-init model on the same val.txt scored AUC = 0.5068.
- **Probe 4 cos@1% ∈ [0.9969, 0.9974]** — all three checkpoints are extremely insensitive to 1% substitution noise (clean and noisy embeddings are ≥99.7% cosine-similar). Spread across checkpoints is 0.0005, well below the within-checkpoint std (~0.003).
- **Probe 1 within-sample mean cos ≈ 0.88, between-sample mean cos ≈ 0.88** — embeddings are clustered into a narrow cone (cosines ~0.88 to *every* other read), with a tiny offset distinguishing same-sample pairs.

## Within-R4 ranking — val_bpb vs probes

| ranking | best → worst |
|---|---|
| val_bpb | r4_baseline ≻ r4_baseline_curve ≻ r4_r3_winner |
| probe1_AUC | r4_r3_winner ≻ r4_baseline ≻ r4_baseline_curve |
| probe4_cos@1% | r4_baseline ≻ r4_baseline_curve ≻ r4_r3_winner |

- val_bpb is **anti-correlated** with Probe 1 (sample-coherence): the val_bpb-worst checkpoint has the strongest sample-coherence signal.
- val_bpb is **correlated** with Probe 4 (noise robustness): the val_bpb-best checkpoint is the most noise-invariant.

Interpreted together, the val_bpb-best models are the most input-invariant — they collapse different reads into nearby embeddings. That collapse helps Probe 4 by definition (clean ≈ noisy when both map to the same point) and hurts Probe 1 (within-sample similarity can't beat between-sample if everything looks the same). Both probes corroborate the same underlying pattern.

## Caveats

- **n=3 with effectively n=2 distinct configs.** `r4_baseline` and `r4_baseline_curve` are duplicates (val_bpb 1.939903 vs 1.939989). The actual contrast is "untouched config @ 1.940" vs "R3 winner config @ 1.948".
- **Magnitudes are tiny.** AUC range 0.0018, cos@1% range 0.0005. Direction of the effect is consistent across probes but a pivot decision should not rest on these magnitudes alone.
- **Pre-leak Probe 4 (Apr 28) showed weak anti-correlation with val_bpb on the R1–R3 repro checkpoints.** The R4 leak-free Probe 4 instead shows weak *correlation* with val_bpb. The flip is small in absolute terms but worth noting — the leak fix changed the sign of one of the probe relationships.

## Recommended next step

The R4 panel is consistent with "val_bpb on this corpus rewards representational collapse, not biology learning." But the n=3 sample size (effectively n=2) is too thin to anchor a pivot decision alone. Suggested before pivoting to MLM+contrastive:

1. Add 2–3 more R4 configs at deliberately *different* val_bpb levels (e.g. early-stop checkpoints at 1.96 / 1.95 / 1.94) and re-run the panel — check whether the direction holds across a wider val_bpb range.
2. Run Probe 1 on `repro_baseline_rerun` (already done — AUC=0.5045 on pre-leak R3 baseline, vs random-init 0.5068 — both at chance) and the other 3 pre-leak repro checkpoints, to populate the full pre-leak ↔ post-leak comparison the spec's step 1 anticipated.

## Files

- Per-checkpoint combined JSONs: [probes_r4_baseline/](probes_r4_baseline/), [probes_r4_baseline_curve/](probes_r4_baseline_curve/), [probes_r4_r3_winner/](probes_r4_r3_winner/)
- Per-probe JSONs in same dirs: `<ckpt>_probe1.json`, `<ckpt>_combined.json` (combined contains probe4 rows as a nested list)
- Pre-leak Probe 4 (Apr 28): [probe4_summary.md](probe4_summary.md), [probe_noise_robustness.csv](probe_noise_robustness.csv)
- Pre-leak Probe 1 (today, baseline only): [probe1_baseline.json](probe1_baseline.json) — AUC = 0.5045
- Random-init sanity: [probe1_random_init.json](probe1_random_init.json) — AUC = 0.5068
