# HARMONY autoresearch program

**Status as of 2026-04-27:** 38 experiments completed across 3 rounds, R2 winner at val_bpb=1.932465. P0 infrastructure committed at `1cb7510`. Hyperparameter rounds paused pending probe-gate evaluation of whether val_bpb tracks HARMONY's actual deliverable.

---

## 1. What HARMONY is

HARMONY is a perception layer for microbiome data. It takes raw sequencing reads from any protocol — 16S amplicon, WGS shotgun, long-read PacBio HiFi or ONT — and produces bias-corrected representations that downstream tools (taxonomic classifiers, abundance estimators, association studies) consume.

**HARMONY's deliverable is bias-corrected reads, not taxonomic assignment.** Taxonomy is the user's job downstream. A later HARMONY version may add taxonomic outputs as an optional head, but the core value proposition is debiasing across platforms.

This positioning is the differentiator from BiomeGPT (operates on processed feature tables) and Read2Pheno (end-to-end phenotype prediction from reads). HARMONY operates upstream of bias propagation, on raw reads, and emits per-read or per-k-mer outputs — never per-sample collapsed embeddings.

## 2. What this repo is

A fork of Karpathy's nanoGPT autoresearch loop (originally `miolini/autoresearch-macos`), adapted for macOS/MPS and used as the **Phase -1 feasibility study** for HARMONY.

Phase -1 answers two questions:
- Can language models learn meaningful patterns from raw microbiome reads at all?
- What is the empirical scaling law (loss vs compute slope, à la Chinchilla)?

This is the smallest experiment that tells us whether the broader HARMONY/Genesis Mission architecture is viable before committing to Argonne A100 compute via the DDF application. The deliverable from this repo is a scaling-paper-grade result plus a compute estimate for Phase 0.

## 3. How the loop actually works

**Note: this section supersedes the prior description of a manual git-branch loop, which is no longer the implementation.**

The loop is LLM-driven and CSV-driven, not branch-based:

1. Proposer (Claude Sonnet 4.6 via API, in `autoresearch_llm.py`) reads prior experiment history from `experiments/results.csv` plus the `HYPERPARAMS` dict (14 knobs).
2. Proposer generates a named experiment configuration.
3. Loop swaps `train.py.baseline` → `train.py` with the proposed config, runs `train.py` with a 5-minute time budget.
4. As of commit `1cb7510`: subprocess streams stdout line-by-line, parses step `dt:` values, and fast-fails any run whose first 3 steps average > 8s (saves ~14.5 min per doomed run; ~7 such runs typical per sweep).
5. Output parser extracts `val_bpb`, `peak_vram_mb`, `mfu_percent`, appends row to `experiments/results.csv` (15-column canonical schema).
6. Repeat.

`train.py.baseline` is the canonical baseline; `train.py` is overwritten per experiment then restored. Working tree should be clean between experiments after commit `1cb7510`.

## 4. Training corpus

**140-sample WGS subset, 200K reads/sample, ~28M reads total. Single-protocol.**

This is a deliberate Phase -1 simplification — the goal is to test whether the model learns *anything* from raw microbiome reads before testing whether it learns *across platforms*. Multi-protocol training is deferred until HCHS/SOL paired data (1,772 paired 16S V4 + WGS samples, now confirmed open-access) is integrated.

Implication: the current val_bpb=1.932 is single-modality feasibility evidence. It says nothing about HARMONY's central cross-protocol invariance claim. Frame R1–R3 results in writeups accordingly.

## 5. Critical caveats — val_bpb numbers are a proxy AND may be confounded

The autoresearch loop optimizes val_bpb on a **causal next-token prediction objective** (autoregressive language model, predict next BPE token). Per the 2026-02-03 design decision, this is "next-read prediction" as Nick recommended: the model predicts subsequent tokens including across read boundaries, learning sample-level community structure implicitly without supervised labels.

Two separate concerns about val_bpb as a measure of HARMONY-aligned representation quality:

**Concern 1 — proxy quality unknown.** The April 24 ML/DL microbiome literature review found language-modeling-only objectives are typically insufficient for biological sequence representation quality on downstream tasks; contrastive arms are usually required. Whether causal LM val_bpb tracks HARMONY's actual deliverable (bias-corrected, protocol-invariant, community-discriminative representations) is empirically untested. The probe gate (Section 6) tests this directly.

**Concern 2 — possible data leakage.** Discovered 2026-04-27: the train/val split is at file-pair level, not sample level. Paired-end reads from the same molecule can have Read 1 in train and Read 2 in val (e.g., SRR6915093_1 in train, SRR6915093_2 in val). All 38 historical val_bpb numbers may be systematically inflated by this leakage. The R2 winner's apparent 0.02 bpb improvement over baseline may be partially or fully attributable to data leakage rather than learning. Three remediation options under discussion: (a) refit split at sample level and rerun, (b) compute leak-aware val_bpb from existing data, (c) accept inflation and treat val_bpb as relative ordering only.

Until these are resolved, val_bpb numbers in this repo should be reported with both caveats. They remain useful as relative ordering across configurations sharing the same data pipeline, but absolute values should not be cited as feasibility evidence.

## 6. The probe gate

Before any further val_bpb optimization rounds (R4, R5, ...), four probes run on existing checkpoints. None require retraining.

**Probe 1 — within-sample vs between-sample consistency.** Tests community-discriminative property. Run on four checkpoints (R1 baseline, R1 best, R2 winner, R3 best) for cross-checkpoint comparison.

**Probe 2 — sample retrieval, stratified by metadata.** Tests community-discrimination, detects batch artifact via stratification on collection date / center / body site.

**Probe 3 — read-quality discrimination.** Tests artifact-awareness. Conditional on FASTQ Phred score availability post-tokenization; may be skipped if BPE pipeline strips quality.

**Probe 4 — synthetic noise robustness.** The most HARMONY-aligned probe. Embed clean reads, apply synthetic substitution noise at 0.5%/1%/5%, measure cosine similarity between clean and noisy embeddings of the same underlying read. Run on four checkpoints alongside Probe 1.

The gating output is a 4-row table: `checkpoint | val_bpb | probe1_ratio | probe4_cos_at_1pct`.

- If Probe 4 score increases as val_bpb decreases monotonically across the four checkpoints → val_bpb is a useful proxy, continue P1/P2 hyperparameter rounds.
- If Probe 4 is flat or anti-correlated → val_bpb does not track the goal, pivot to MLM+contrastive arm.
- If Probe 2 shows >50% retrieval accuracy or clusters by date/center → diagnose batch artifact before any further training.

## 7. Connection to broader HARMONY

| Phase | Role | Scale | Compute |
|---|---|---|---|
| Phase -1 (this repo) | Feasibility + scaling law | ~5–15M params | M3 Pro / Mac mini, MPS |
| Phase 0 | Genesis Simulator + scaled training | ~100M–1B params | Argonne DDF (pending) |
| Phase 1 | Production training + publication | Up to ~1B params | Argonne DDF (pending) |

The 1B parameter run is the bridge between this repo (M3 Pro local) and Genesis Mission scale (Argonne A100). Evo2 at 40B params cost $1–2M as Nick's reference point — 1B is well within reach of DDF allocation.

## 8. Open questions (as of 2026-04-27)

Items to resolve with Nick on 2026-04-28 or shortly after:

- **Val split disposition (NEW 2026-04-27).** Refit at sample level and rerun, compute leak-aware estimate from existing data, or accept inflation and treat val_bpb as relative ordering only? Affects how all 38 historical numbers are reported.
- **Phase -1 success criterion.** val_bpb only, probe-based, or hybrid (scaling-law slope + probe scores + protocol-invariance R²)?
- **val_bpb as proxy.** Is it tracking HARMONY's goal? Probe 4 cross-checkpoint table provides empirical answer.
- **Hierarchy.** Sense A (within-read encoder + across-read set aggregator), Sense B (DeepSeek hybrid attention), or Sense C (phylogeny-as-positional-bias)?
- **HCHS/SOL integration.** Replace iHMP IBD as primary biological validation, supplement it, or train stratified BPE on it?
- **BPE 4,096 vocab corpus.** What was it trained on? Stratification needed for multi-protocol.
- **16S primer-strip preprocessing.** In current pipeline (cutadapt or equivalent)?
- **Output representation.** Per-read embeddings vs per-k-mer logits? Per-sample is ruled out (collapses to BiomeGPT failure mode).

## 9. What this repo deliberately does NOT do

- Taxonomic assignment (downstream user job)
- Multi-protocol training (deferred to HCHS/SOL integration)
- Bias-correction evaluation (deferred to Genesis Simulator paired-protocol pairs)
- Hierarchical architecture (deferred to probe gate decision)
- Contrastive learning arm (deferred to probe gate decision)
- DANN gradient-reversal protocol-discriminator (deferred to Phase 0)
- Long-read context lengths (deferred — 30 kb HiFi reads = ~15K tokens, beyond Phase -1 budget)

These are not gaps in the research program; they are deliberate sequencing decisions to keep Phase -1 narrow and answerable on local compute.

## 10. References

- Karpathy autoresearch: `github.com/karpathy/autoresearch`
- Fork basis: `miolini/autoresearch-macos`
- Chinchilla scaling: Hoffmann et al. 2022, DeepMind
- Evo2 reference: Brixi et al. 2025, Arc Institute / Stanford / NVIDIA
- HARMONY V2 architecture: see `ARCHITECTURE_DOCUMENTATION.md` in main project
- ML/DL microbiome review (2026-04-24): Hernández Medina et al. 2022 + Przymus et al. 2025
- HCHS/SOL paired data: Usyk et al. (open-access, 1,772 paired samples)

---

*Last updated: 2026-04-27 — replaces prior version that described a manual git-branch loop superseded by the LLM-driven CSV loop in `autoresearch_llm.py`. Section 5 corrected: training objective is causal next-token prediction (not MLM as earlier conversation drafts incorrectly stated), matching 2026-02-03 design decision. Section 8 adds val split paired-end leakage as new open question.*
