# HARMONY — Microbiome Harmonizer

Turning raw microbiome sequencing reads into **bias-corrected, cross-study-comparable representations.**

---

## The problem

Microbiome datasets from different studies are often incompatible, so researchers can't combine
them, run cross-study analyses, or validate findings across cohorts. The incompatibility comes from
**technical bias**:

- sample collection & DNA extraction protocols
- primer / 16S-region choice (V4 vs V3–V4 vs full-length)
- sequencing platform (Illumina vs PacBio)
- assay type (16S amplicon vs shotgun metagenomics)

**HARMONY's goal:** learn a representation that **removes technical bias while preserving biology**,
so data from different studies/platforms can be merged. Because different primers/regions literally
*observe different parts of the genome*, HARMONY works at the **sequence/read level** — the only level
that can align across assays that don't share features (unlike abundance-table methods such as ComBat,
MMUPHin, ConQuR).

> **Status:** research prototype. Phase −1 (can we model reads at all?) is complete and positive; the
> harmonizer architecture is designed and the project is scaling up (Aurora) + acquiring the
> multi-platform / mock-community data it needs. See
> [HARMONY_STATUS_AURORA_HANDOFF.md](HARMONY_STATUS_AURORA_HANDOFF.md) for the full status.

---

## Phase −1 results — within-read masked language modeling

**Metric:** MSK-DNA top-1 on held-out validation — mask a DNA chunk, is the model's single best guess
right? Baselines: chance/unigram **0.092**; a non-learning count model (lookup table) **0.127**.

The headline finding is a clean **capacity × data interaction** (all runs on a 630k-read cohort):

| Model | Params | Val top-1 | Notes |
|---|---|---|---|
| depth-2 | 5M | 0.118 | data saturated at this size |
| depth-4 | 41M | 0.165 | capacity unlock |
| **depth-6** | **120M** | **0.175** | still rising when a 10 h wall stopped it |

Conclusions: (1) **capacity and data are multiplicative** — neither moves the needle alone, which is
why early single-variable sweeps looked flat and produced a *false* "ceiling"; (2) the model
**generalizes, not memorizes** (validation ≥ train even at 120M params); (3) it's **well past** the
non-learning baseline, so real within-read signal exists; (4) further scaling is **compute-bound** —
hence the move to Aurora. A DNABERT-2 backbone was evaluated and found no better than our model on
sample-level coherence (both near chance), which is a property of metagenomic reads, not of DNABERT-2.

---

## Harmonizer architecture (design)

```
16S reads (multi-platform)
      │
 [read encoder: DNABERT-2 (frozen) or our depth-6 MLM]
      │  per-read embeddings
 [attention / set pooling]              ← a sample = a SET of reads (permutation-invariant)
      │
      ├──► z_bio   (harmonized, batch-invariant)   ── OUTPUT
      └──► z_tech  (the captured technical bias)
```

**Losses:** adversarial batch-invariance (gradient reversal — platform/primer/protocol unpredictable
from `z_bio`); a technical head that *predicts* the metadata (sinks the bias into `z_tech`);
reconstruction (prevents `z_bio` collapse); and **mock-community anchors** — the same known biology
sequenced across platforms, which simultaneously supervise alignment, preserve biology, provide the
evaluation ground truth, and act as the canary against over-correction (the central risk: erasing
biology that correlates with batch).

---

## Installation

Apple Silicon / MPS, a single NVIDIA GPU, or Intel Max GPU (Aurora). Python 3.12; [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # runtime deps
uv sync --group dev          # + pytest
```

Optional (DNABERT-2 comparison, isolated env recommended):
```bash
pip install torch "transformers>=4.38,<5" einops
python dnabert2_setup.py     # downloads ~450MB, auto-patches the triton import, smoke-tests
```

---

## Usage

**Train the within-read MLM** (the current workhorse; writes checkpoints + a train-vs-val trajectory):
```bash
python train_mlm.py --out-dir experiments/myrun \
  --depth 6 --aspect-ratio 192 --lam 0 \
  --reads-cap 5000 --samples-per-batch 4 --reads-per-sample 8 --seq-len 64 \
  --max-steps 40000 --eval-every 2000 --seed 42 \
  --mlm-softcap 15 --mask-prob 0.15 --max-runtime-hours 10
```
Key flags: `--depth`/`--aspect-ratio` (model size), `--reads-cap` (reads kept per sample = data volume),
`--lam` (contrastive weight; 0 = pure MLM), `--mlm-softcap`/`--mask-prob` (sweepable). Every
`--eval-every` steps it logs **val and train** MSK-DNA top-1/top-5/CE and saves `best_val.pt`.

**Diagnostics & baselines:**
```bash
python kmer_markov_baseline.py     # non-neural count-model floor
python nn_vs_markov_diag.py        # neighbor-corruption diagnostic
python vanilla_encoder_mlm.py      # off-the-shelf nn.TransformerEncoder control
```

**Dashboards & decoded predictions** (self-contained HTML):
```bash
python build_dashboard.py          # experiments/dashboard.html (whole investigation)
python dashboard_run.py            # experiments/dashboard_run_depth6.html (single run)
VIZ_CHECKPOINT=experiments/myrun/checkpoint.pt VIZ_OUT_HTML=experiments/myrun/decoded.html \
  VIZ_DEPTH=6 VIZ_ASPECT=192 python decoded_visualizer.py   # per-read colored top-1/top-5 view
```

---

## Running on Aurora (Intel Max GPU / oneAPI)

Aurora (ANL) uses **Intel Data Center GPU Max ("Ponte Vecchio")** via oneAPI, exposed in PyTorch as the
**`xpu`** device. HARMONY is device-abstracted for it.

**Device selection — [device_utils.py](device_utils.py).** Autodetects in order **`xpu → cuda → mps →
cpu`**; override anywhere with the environment variable:
```bash
HARMONY_DEVICE=xpu python train_mlm.py ...      # force a backend (xpu|cuda|mps|cpu)
python device_utils.py                          # prints the detected device banner
```
It also provides device-agnostic `empty_cache`, `synchronize`, and `manual_seed_all`. All training,
eval, and diagnostic scripts use it — no hardcoded `mps`/`cuda` anywhere in the active path. On
torch ≥ 2.5, `torch.xpu` is native; older oneAPI stacks are handled by an opportunistic
`import intel_extension_for_pytorch`.

**Validate after allocation** (numerics regression — expect step-0 loss ≈ **8.317**):
```bash
HARMONY_DEVICE=xpu python train_mlm.py --out-dir experiments/_xpu_check \
  --depth 2 --aspect-ratio 128 --lam 0 --max-steps 200 --eval-every 0 \
  --reads-cap 50 --seed 42 --max-runtime-hours 1
```

**Known port issue — bfloat16.** `model.py` runs token/value embeddings and rotary in **bf16** while
linear weights stay fp32. MPS and CUDA auto-promote that mixed matmul; **CPU rejects it**
(`BFloat16 != float`). Intel Max GPUs support bf16, so it should work on `xpu` as on MPS/CUDA — but
**validate on hardware.** CPU-only debugging (e.g., login nodes) would require an explicit fp32 mode
(a small, deliberate `model.py` change, not yet made).

**Remaining port tasks:** validate bf16 + the Muon optimizer + the custom bidirectional-attention path
on XPU; add DDP multi-GPU / multi-node scaling (currently single-device); confirm toolchain versions
against current ANL Aurora docs. Details in
[HARMONY_STATUS_AURORA_HANDOFF.md](HARMONY_STATUS_AURORA_HANDOFF.md).

---

## Data pipeline

```
FASTQ (.fastq.gz)
   │  prepare_fastq.py     paired-end aware; sample-stem split; R1 + <PAIRED_END> + revcomp(R2)
output/{train,val}.txt
   │  prepare_genomic.py   BPE tokenizer (vocab 4096 + [MASK]=4096), dataloader
   │  paired_data_loader.py  sample-aware batching (K samples × M reads)
train_mlm.py               bidirectional MLM (+ optional InfoNCE contrastive) trainer
```

**Current data & the gap.** `output/train.txt` is one study (`SRR6915091`–`SRR6915230`, 140 samples,
single platform, paired-end). The harmonizer needs **multi-platform data + mock communities with known
composition** — acquiring these is the top non-compute priority (candidates: ZymoBIOMICS/ATCC mock
standards across platforms, MBQC, mockrobiota).

---

## Repository layout

```
train_mlm.py                 bidirectional MLM trainer (val+train eval hook, best_val.pt, sweepable)
model.py                     GPT, MuonAdamW, bidirectional attention (importable)
paired_data_loader.py        sample-aware paired-read batching
prepare_fastq.py             FASTQ → text stream (paired-end aware, sample-stem split)
prepare_genomic.py           BPE tokenizer, dataloader
device_utils.py              device abstraction: xpu / cuda / mps / cpu   ← Aurora port
dnabert2_setup.py            verified DNABERT-2 loader for Apple Silicon (triton patch)

kmer_markov_baseline.py      non-neural count-model floor
nn_vs_markov_diag.py         neighbor-corruption diagnostic
vanilla_encoder_mlm.py       off-the-shelf nn.TransformerEncoder control
probe_sample_coherence.py    Probe-1 sample-coherence (ROC-AUC)

build_dashboard.py           whole-investigation Tufte dashboard
dashboard_run.py             single-run dashboard
decoded_visualizer.py        per-read colored decoded-prediction view (size-configurable)

experiments/                 per-run checkpoints, trajectories, dashboards, JSON results
tests/                       pytest suite
HARMONY_STATUS_AURORA_HANDOFF.md   full status + Aurora port notes
ARCHITECTURE_V2_REDESIGN.md / ARCHITECTURE_V3_UNIFIED.md   architecture vision
```

---

## Lineage & License

Forked from [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos), a macOS/MPS
adaptation of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). HARMONY-specific work
is the data pipeline, the MLM/contrastive trainer, the diagnostics, and the harmonizer design.

**MIT**, inherited from the upstream projects.
