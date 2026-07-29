# HARMONY — Status & Aurora Handoff
_Status as of 2026-07-08. Prepared for the move to Aurora (ANL)._

---

## 1. What HARMONY actually is (goal, restated)

**Microbiome Harmonizer.** Correct **technical bias** across microbiome studies so datasets can be
combined for cross-study analysis. Biases to remove: sequencing platform (Illumina vs PacBio),
primer / 16S region, DNA extraction protocol, lab/study, and eventually 16S vs shotgun.

The within-read masked-language-model (MLM) work below was **foundation** — proving we can model
reads at all. It is **not** harmonization by itself; harmonization needs an objective that explicitly
separates biology from technical bias (Section 6).

**Design decisions locked 2026-07-08:**
| Decision | Choice |
|---|---|
| Level of operation | **Sequence / read level** (only level that can bridge different primers/regions and 16S↔shotgun; abundance-table methods like ComBat/MMUPHin/ConQuR need matching features) |
| Initial scope | **Within-16S**, across platforms / protocols / labs |
| Success metric | **Recover known biology** consistently across studies |
| Biological ground truth | **Mock communities** (defined composition sequenced across platforms) |
| Supervision available | **Technical metadata only** (no biological labels, no technical replicates) |

---

## 2. Build status — what exists and runs

**Core pipeline (pre-existing):** `prepare_fastq.py` → `output/train.txt`/`val.txt` (tokenized reads,
BPE vocab 4096 + `[MASK]`=4096), `paired_data_loader.py` (sample-aware batching),
`train_mlm.py` (bidirectional MLM + optional InfoNCE contrastive), `probe_sample_coherence.py` (Probe-1).

**Added during this investigation:**
| File | Purpose |
|---|---|
| `train_mlm.py` (extended) | **val-MLM eval hook** (MSK-DNA top-1/top-5/CE on val, deterministic RNG-isolated masking, does not perturb training); **`best_val.pt`** selection on val CE; **`eval_train_mlm`** (train-accuracy = memorization check); configurable `--mlm-softcap`, `--mask-prob`; trajectory CSV now logs train *and* val |
| `kmer_markov_baseline.py` | Non-neural count-model floor (bidirectional gold-neighbor interpolated trigram) |
| `nn_vs_markov_diag.py` | Neighbor-corruption diagnostic (clean vs corrupted neighbor split) |
| `vanilla_encoder_mlm.py` | Off-the-shelf `nn.TransformerEncoder` control (architecture test) |
| `build_dashboard.py` → `experiments/dashboard.html` | Whole-investigation Tufte dashboard |
| `dashboard_run.py` → `experiments/dashboard_run_depth6.html` | Single-run dashboard (last run) |
| `decoded_visualizer.py` (extended) | Per-read colored decoded predictions; now size-configurable via `VIZ_DEPTH`/`VIZ_ASPECT` |
| `dnabert2_setup.py` | **Verified** DNABERT-2 load/embed on Apple Silicon (patches the triton import) |
| `run_mlm_sweep.sh` | Hyperparameter sweep driver |

**Completed runs:** `mlm_lam0`, `mlm_big`, `mlm_memcheck`, `mlm_moredata`, `mlm_bigcohort`,
`mlm_bigcohort_d4`, `mlm_bigcohort_d6`, `mlm_vanilla` (all with checkpoints + trajectories).

---

## 3. Training results — Phase −1 (within-read MLM)

**Metric:** MSK-DNA top-1 on held-out validation (fill in a masked DNA chunk, is the single best guess right).
**Baselines:** chance/unigram **0.0923**; non-learning lookup table (count model) **0.127** (0.135 on clean-neighbor positions).

### The scaling story
| Run | Data | Model | Val top-1 (peak) | train−val gap | Read |
|---|---|---|---|---|---|
| memcheck | 6.3k reads | depth-2, 5M | 0.108 | **+0.19** | memorizes, doesn't generalize |
| more-data | 252k reads | depth-2, 5M | 0.119 | −0.00 | first real data gain, replicates |
| big-cohort | 630k reads | depth-2, 5M | 0.118 | −0.00 | **data saturated** at this model size |
| big-cohort | 630k reads | depth-4, 41M | 0.165 | −0.02 | **capacity unlock** |
| big-cohort | 630k reads | **depth-6, 120M** | **0.175** | −0.02 | still rising; stopped by 10h wall |

### Conclusions (validated)
1. **Capacity × data are multiplicative.** Neither works alone. On tiny data, *nothing* responded
   (objective, capacity, softcap, mask-rate, even a vanilla BERT control at 0.107) — which produced a
   **false "ceiling" conclusion** that more data overturned. Nick's data-regime diagnosis was correct.
2. **No fitting bug.** Memorization sanity check passed: train top-1 climbed 0.10→0.29 on small data.
3. **Generalizes, not memorizes.** At 41M and 120M params, **val ≥ train** throughout — no overfitting.
4. **Well past the non-learning baseline** (0.175 vs 0.127), so real within-read signal exists.
5. **Not converged — compute-bound.** depth-6 hit a 10-hour wall at step 32k/40k (~1.1 s/step, 9.6h)
   while still improving. depth-8 (453M) is impractical on Apple Silicon (days/run).

---

## 4. DNABERT-2 evaluation (pretrained backbone test)

- **Verified working** on Apple Silicon (`dnabert2_setup.py`); 117M params; ~52 ms/read on CPU.
  Needed one patch: its remote code hard-imports `triton` (no Apple build) — the model already falls
  back to PyTorch attention, so we neuter that import in a local copy.
- **Head-to-head on sample coherence (identical metric):** DNABERT-2 **0.516** vs our depth-2 **0.525**,
  depth-6 **0.521**, chance **0.500**. All near chance.
- **Interpretation:** not a DNABERT-2 failure — **read-level sample coherence is intrinsically
  near-chance for metagenomes** (a sample is a mixture of hundreds of species, so two reads from the
  same specimen need not resemble each other). **The metric was wrong; sample identity requires
  aggregating the read population, not comparing individual reads.**

---

## 5. Data status — the critical gap

**Current data:** `output/train.txt` (6.0 GB), **140 samples** (126 train / 14 val), all from a single
consecutive SRA series `SRR6915091`–`SRR6915230` → **one study, one platform, paired-end**, no
technical metadata, **no mock communities**.

Each sample holds ~271k reads; our largest run used 5,000/sample (630k total) = **<2% of available reads**.

**Gap:** the Harmonizer needs **multi-platform data + mock communities with known composition**.
Current data has neither, so **Phase 0 (quantify the bias) is blocked on data acquisition.**
Candidate sources: ZymoBIOMICS / ATCC mock standards across platforms, MBQC, mockrobiota.

---

## 6. Proposed Harmonizer architecture (most recent design)

```
16S reads (multi-platform)
      │
 [read encoder: DNABERT-2 (frozen) or our depth-6]
      │  per-read embeddings
 [attention / set pooling]              ← sample = a SET of reads (permutation-invariant)
      │
      ├──► z_bio  (harmonized, batch-invariant)   ── OUTPUT
      └──► z_tech (the captured bias)
```
**Losses:** (a) **adversarial batch-invariance** — gradient reversal so platform/primer/protocol are
*unpredictable* from `z_bio`; (b) **technical head** — `z_tech` *predicts* the metadata (sinks the bias);
(c) **reconstruction** — `(z_bio, z_tech)` rebuilds the sample embedding (stops z_bio collapsing);
(d) **mock anchors** — same mock across platforms → same `z_bio`, and `z_bio` → known composition.

**Why mocks are the linchpin:** they are the *same known biology across different technology*, so they
simultaneously (1) supervise alignment, (2) anchor biology preservation, (3) provide evaluation ground
truth, and (4) act as the **canary against over-correction** (the central risk: if biology correlates
with batch, an adversary can erase biology — mocks are unconfounded by construction and would show it).

---

## 7. What Aurora unlocks + porting notes

**The binding constraint is compute.** Priority runs:
1. **Finish depth-6 to convergence** (it was still rising at 32k/40k steps).
2. **Extend the capacity ladder** — depth-8 (453M) and beyond: does 0.175 keep climbing?
3. **Use far more data** — we're using <2% of available reads; scale reads/sample and steps together.
4. **Train the Harmonizer** (adversarial disentangler) once mock/multi-platform data is acquired.

Aurora uses **Intel Data Center GPU Max (Ponte Vecchio) via oneAPI**, where PyTorch runs on the
**`xpu`** device (Intel Extension for PyTorch). torch 2.6 already exposes `torch.xpu`.

**DONE (2026-07-08): device-selection abstraction.** Added `device_utils.py` (selection order
`xpu → cuda → mps → cpu`, override via `$HARMONY_DEVICE`, plus device-agnostic `empty_cache` /
`synchronize` / `manual_seed_all`). Rewired all active scripts (`train_mlm.py`, `model.py`,
`probe_sample_coherence.py`, `vanilla_encoder_mlm.py`, `nn_vs_markov_diag.py`, `decoded_visualizer.py`,
`floor_diag_*.py`, `probe_gate.py`, `smoke_run.py`, `wiring_verify.py`). **MPS numerics verified
unchanged** — step-0 loss 8.3176 matches the historical baseline; `model._device_type` still resolves
to the same value so the optimizer compile path is untouched. XPU deliberately falls through to the
non-`torch.compile` optimizer path (like mps) until validated.

**RESOLVED (2026-07-23) — bf16 dtype dependency (`model.py`).** The model cast token embeddings,
value-embeds, and rotary cos/sin to **bfloat16** while linear weights stayed fp32. MPS/CUDA auto-promote
that mixed matmul; **XPU and CPU reject it** (`RuntimeError: ... BFloat16 != float`) — confirmed on
Aurora hardware. Fixed via `model.py:_EMB_DTYPE`: **fp32 on xpu/cpu, bf16 on cuda/mps**, overridable
with `HARMONY_FP32=1`. Validated on an Aurora compute node: `device=xpu`, step-0 loss ≈ **8.318**, loss
descends 8.32 → 5.6 over 200 steps, ~30 ms/step (depth-2). The Muon optimizer's internal bf16 block is
self-consistent (bf16×bf16) and runs fine. Benign warning remaining: "IPEX doesn't support xetla" —
SDPA falls back to a working non-optimized attention kernel (a throughput note, not a correctness bug).

**Remaining porting tasks:**
- ✅ ~~Validate bf16/Muon/attention on XPU~~ — done; xpu path is fp32 (above).
- Add **DDP / multi-GPU + multi-node** scaling — **the next big build** (runs currently use 1 of a
  node's 12 tiles).
- Transfer the full `train.txt` (5.6 GB) to Aurora via **Globus** (only `val.txt` is up so far).
- Revisit the **xetla/SDPA** attention kernel for throughput on long runs.
- Confirm exact module/toolchain versions against current ANL Aurora documentation.

---

## 8. Immediate next steps

| Priority | Action | Blocker |
|---|---|---|
| 1 | Port `train_mlm.py` to Aurora XPU + DDP; reproduce depth-6 baseline | none |
| 2 | Finish depth-6 to convergence; run depth-8 capacity ladder | Aurora access |
| 3 | Acquire **mock-community, multi-platform 16S** dataset | data acquisition |
| 4 | Run **Phase 0** — variance decomposition (platform vs true composition) in embedding space | needs #3 |
| 5 | Build + train the adversarial Harmonizer (Section 6) | needs #3, #4 |

**Dashboards for reference:** `experiments/dashboard.html` (whole investigation),
`experiments/dashboard_run_depth6.html` (last run), `experiments/mlm_bigcohort_d6/decoded_predictions.html`
(per-read decoded predictions, MSK top-1 17.5% / top-5 23.6%).
