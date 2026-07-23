# HARMONY — Microbiome Harmonizer

Turns raw microbiome sequencing reads into representations that are bias-corrected and comparable across studies.

---

## The problem

Microbiome datasets from different studies usually can't be combined. You can't run a cross-study
analysis or check whether a finding holds up in another cohort, because the numbers aren't measuring
the same thing. Most of that mismatch is technical, not biological. It comes from things like:

- how the sample was collected and how DNA was extracted
- which primer or 16S region was targeted (V4, V3–V4, full-length)
- the sequencing platform (Illumina vs PacBio)
- the assay itself (16S amplicon vs shotgun metagenomics)

HARMONY tries to learn a representation that strips out the technical bias but keeps the biology, so
studies and platforms can actually be merged. It works on the sequences themselves, at the read level,
because different primers and regions literally look at different parts of the genome. That's the only
level that can line up assays that don't even share features, which is where abundance-table methods
like ComBat, MMUPHin, and ConQuR run out of room.

> **Status:** research prototype. Phase −1 (can we model reads at all?) is done, and the answer was
> yes. The harmonizer architecture is designed, and we're now scaling up on Aurora and tracking down
> the multi-platform and mock-community data the next phase needs. Full status is in
> [HARMONY_STATUS_AURORA_HANDOFF.md](HARMONY_STATUS_AURORA_HANDOFF.md).

---

## Phase −1 results — within-read masked language modeling

The metric is MSK-DNA top-1 on held-out validation: mask a chunk of DNA, and ask whether the model's
single best guess is right. Two baselines to beat. Chance (unigram) sits at 0.092. A non-learning
count model, just a lookup table, gets 0.127.

Every run below used the same 630k-read cohort. What jumped out was that capacity and data only pay
off together:

| Model | Params | Val top-1 | Notes |
|---|---|---|---|
| depth-2 | 5M | 0.118 | saturated the data at this size |
| depth-4 | 41M | 0.165 | capacity starts to matter |
| **depth-6** | **120M** | **0.175** | still climbing when a 10 h wall stopped it |

A few things follow from this. Capacity and data are multiplicative, so moving either one alone does
almost nothing. That's why the early single-variable sweeps looked flat and convinced us there was a
ceiling that wasn't real. The model generalizes rather than memorizes: validation stays at or above
train even at 120M params. And it's well clear of the non-learning baseline, so there is genuine signal
inside a read to learn. The remaining limit is compute, which is what pushed us onto Aurora. We also
tried a DNABERT-2 backbone and it did no better than our own model on sample-level coherence, both near
chance. That's a property of metagenomic reads, not a knock on DNABERT-2.

---

## Harmonizer architecture (design)

```
16S reads (multi-platform)
      │
 [read encoder: DNABERT-2 (frozen) or our depth-6 MLM]
      │  per-read embeddings
 [attention / set pooling]              ← a sample = a SET of reads (permutation-invariant)
      │
      ├──►  z_bio   (harmonized, batch-invariant)   ── OUTPUT
      └──►  z_tech  (the captured technical bias)
```

There are four losses working against each other. An adversarial batch-invariance loss (via gradient
reversal) makes platform, primer, and protocol unpredictable from `z_bio`. A technical head does the
opposite job on purpose, predicting the metadata so the bias drains into `z_tech`. A reconstruction
loss keeps `z_bio` from collapsing. And mock-community anchors carry most of the weight: the same known
biology sequenced on different platforms. Those anchors do four things at once. They supervise the
alignment, they hold the biology in place, they give us the evaluation ground truth, and they're the
canary for over-correction, which is the whole risk here. The failure mode we're guarding against is
erasing real biology just because it happens to correlate with batch.

---

## Installation

Runs on Apple Silicon / MPS, a single NVIDIA GPU, or the Intel Max GPU on Aurora. Python 3.12, with
[uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # runtime deps
uv sync --group dev          # + pytest
```

For the DNABERT-2 comparison, use an isolated env:

```bash
pip install torch "transformers>=4.38,<5" einops
python dnabert2_setup.py     # downloads ~450MB, auto-patches the triton import, smoke-tests
```

---

## Usage

Train the within-read MLM. This is the current workhorse, and it writes checkpoints plus a
train-vs-val trajectory as it goes:

```bash
python train_mlm.py --out-dir experiments/myrun \
  --depth 6 --aspect-ratio 192 --lam 0 \
  --reads-cap 5000 --samples-per-batch 4 --reads-per-sample 8 --seq-len 64 \
  --max-steps 40000 --eval-every 2000 --seed 42 \
  --mlm-softcap 15 --mask-prob 0.15 --max-runtime-hours 10
```

The flags that matter: `--depth`/`--aspect-ratio` set model size, `--reads-cap` sets how many reads you
keep per sample (i.e. data volume), `--lam` is the contrastive weight (0 means pure MLM), and
`--mlm-softcap`/`--mask-prob` are the two you'll sweep. Every `--eval-every` steps it logs both val and
train MSK-DNA (top-1, top-5, CE) and saves `best_val.pt`.

Diagnostics and baselines:

```bash
python kmer_markov_baseline.py     # non-neural count-model floor
python nn_vs_markov_diag.py        # neighbor-corruption diagnostic
python vanilla_encoder_mlm.py      # off-the-shelf nn.TransformerEncoder control
```

Dashboards and decoded predictions, all self-contained HTML:

```bash
python build_dashboard.py          # experiments/dashboard.html (whole investigation)
python dashboard_run.py            # experiments/dashboard_run_depth6.html (single run)
VIZ_CHECKPOINT=experiments/myrun/checkpoint.pt VIZ_OUT_HTML=experiments/myrun/decoded.html \
  VIZ_DEPTH=6 VIZ_ASPECT=192 python decoded_visualizer.py   # per-read colored top-1/top-5 view
```

### The pipeline, file by file — data to results

```
 FASTQ files (raw sequencing reads)
      │
   ① prepare_fastq.py      → output/train.txt, val.txt      (reads as text)
   ② prepare_genomic.py    → BPE tokenizer                   (DNA → tokens)
      │
   ③ paired_data_loader.py → batches (K samples × M reads)
      │
   ④ model.py  +  ⑤ train_mlm.py   ── trains the model ──►  checkpoints + trajectories
      │            (⑥ contrastive_loss.py, ⑦ device_utils.py support these)
      │
   ⑧ evaluation / diagnostics  ──►  ⑨ dashboards & decoded views
```

**Stage 1 — Data preparation (raw reads → tokenized text)**

- ① `prepare_fastq.py` — takes raw FASTQ files and writes them out as plain text. It's paired-end
  aware, so it joins each molecule as `R1 + <PAIRED_END> + reverse-complement(R2)`, and it splits
  train/val by sample rather than by file, which is what closed the leak we had earlier. Writes
  `output/train.txt` and `val.txt`.
- ② `prepare_genomic.py` — trains the BPE tokenizer on that text, a vocabulary of about 4,096 DNA
  chunks, so a read can be turned into the numbered tokens the model expects.

**Stage 2 — Feeding the model (text → batches)**

- ③ `paired_data_loader.py` — builds the batches, and it's sample-aware. Each step it pulls
  K samples × M reads (say 4 samples × 8 reads), tokenizes and pads them, and hands the model one
  batch. It also guarantees at least 2 samples per batch, which the contrastive loss needs. The
  `--reads-cap` knob, your data-volume dial, lives here.

**Stage 3 — The model itself**

- ④ `model.py` — the core transformer. GPT architecture plus the custom Muon+AdamW optimizer.
  Everything else imports it.
- ⑤ `train_mlm.py` — the current workhorse. It subclasses `model.py` into a bidirectional
  fill-in-the-blank (MLM) model and runs the training loop. The part we added is the evaluation: it
  scores validation and train accuracy at intervals, saves `best_val.pt`, and logs the train-vs-val
  trajectory. Every result above came out of this file.
- ⑥ `contrastive_loss.py` — the optional InfoNCE contrastive loss, the sample-coherence arm. It only
  kicks in when `--lam > 0`; our recent runs set λ=0, pure MLM.
- ⑦ `device_utils.py` — cuts across everything else and picks the chip for you: `xpu → cuda → mps →
  cpu`. This is the piece the Aurora port hangs on.

**Stage 4 — Evaluation & diagnostics (is it actually learning?)**

These load a trained checkpoint and score it. None of them train.

- `probe_sample_coherence.py` — Probe 1, the sample-level readout. Do reads from the same sample
  cluster together? Reported as AUC.
- `kmer_markov_baseline.py` — the non-learning count-model floor, 0.127, which is what we hold the
  neural model up against.
- `nn_vs_markov_diag.py` — splits masked positions by whether their neighbors were clean or corrupted.
- `vanilla_encoder_mlm.py` — an off-the-shelf BERT control, there to check whether our own architecture
  was the problem.
- `floor_diag_big.py` / `floor_diag_lam0.py` / `floor_diag_cheap.py` — capacity and floor checks on
  specific checkpoints: the depth-4 capacity check, the pure-MLM isolation, and a cheaper floor check.
- `evaluate_probes.py` / `probe_gate.py` — run representation probes and probe gates on saved
  checkpoints, left over from the earlier probe framework.

**Stage 5 — Visualization (making results human-readable)**

- `build_dashboard.py` → `experiments/dashboard.html` — the whole-investigation Tufte dashboard.
- `dashboard_run.py` → the single-run dashboard, focused on depth-6.
- `decoded_visualizer.py` — a per-read colored view of the model's actual predictions, green for
  correct and red for a miss.

**Stage 6 — The pretrained-backbone alternative**

- `dnabert2_setup.py` — a verified loader for DNABERT-2, the 117M pretrained DNA model, on Apple
  Silicon. It lets us embed reads with a pretrained model instead of training one from scratch. An
  option for the encoder that we tested but didn't adopt.

**Supporting cast (testing / reproducibility)**

- `wiring_verify.py` — sanity-checks the model wiring on real batches. Doesn't train.
- `smoke_run.py` — a fast smoke test, under 200 steps, to confirm nothing's broken before you commit to
  a real run.
- `reproduce_checkpoints.py` — re-creates specific checkpoints for probing.

**Legacy — the earlier "autoresearch" phase (not in the current flow)**

Leftovers from the CLM (next-token) era, before the MLM pivot. Nick advised getting off the auto-loop,
so we did.

- `train.py` — the old single-file CLM trainer, superseded by `train_mlm.py`.
- `prepare.py` — the old one-time data prep, now split into `prepare_fastq.py` and `prepare_genomic.py`.
- `autoresearch.py` / `autoresearch_llm.py` — the LLM-driven hyperparameter loops, retired in favor of
  exploring by hand.

---

## Running on Aurora (Intel Max GPU / oneAPI)

Aurora (ANL) runs Intel Data Center GPU Max ("Ponte Vecchio") through oneAPI, which PyTorch exposes as
the `xpu` device. HARMONY is written to be device-abstract, so this mostly just works, but there are a
couple of things to check on real hardware.

Device selection lives in [device_utils.py](device_utils.py). It autodetects in the order
`xpu → cuda → mps → cpu`, and you can override it anywhere with an environment variable:

```bash
HARMONY_DEVICE=xpu python train_mlm.py ...      # force a backend (xpu|cuda|mps|cpu)
python device_utils.py                          # prints the detected device banner
```

It also wraps `empty_cache`, `synchronize`, and `manual_seed_all` so they work regardless of backend.
Every training, eval, and diagnostic script goes through it, so there's no hardcoded `mps` or `cuda`
left in the active path. On torch ≥ 2.5, `torch.xpu` is native; older oneAPI stacks are handled by an
opportunistic `import intel_extension_for_pytorch`.

Once you've been allocated a node, run the numerics check. Step-0 loss should come out to about
**8.317**:

```bash
HARMONY_DEVICE=xpu python train_mlm.py --out-dir experiments/_xpu_check \
  --depth 2 --aspect-ratio 128 --lam 0 --max-steps 200 --eval-every 0 \
  --reads-cap 50 --seed 42 --max-runtime-hours 1
```

One known port issue: bfloat16. In `model.py`, the token/value embeddings and rotary run in bf16 while
the linear weights stay fp32. MPS and CUDA auto-promote that mixed matmul, but CPU refuses it
(`BFloat16 != float`). Intel Max GPUs support bf16, so it should behave the same as MPS and CUDA on
`xpu`, but that's exactly the kind of thing you confirm on the hardware rather than assume. If you need
to debug on CPU-only nodes (a login node, say), you'd have to add an explicit fp32 mode, which is a
small deliberate change to `model.py` that hasn't been made yet.

Still on the port list: validate bf16, the Muon optimizer, and the custom bidirectional-attention path
on XPU; add DDP for multi-GPU and multi-node (it's single-device right now); and check the toolchain
versions against the current ANL Aurora docs. The details are in
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

Where the data currently stands, and where it falls short: `output/train.txt` is a single study
(`SRR6915091`–`SRR6915230`, 140 samples, one platform, paired-end). The harmonizer needs more than
that. It needs multi-platform data and mock communities whose composition is known ahead of time.
Getting hold of those is the top priority that isn't about compute. The candidates we're chasing are
the ZymoBIOMICS/ATCC mock standards run across platforms, MBQC, and mockrobiota.

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

Forked from [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos), which is a
macOS/MPS adaptation of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The
HARMONY-specific work is the data pipeline, the MLM/contrastive trainer, the diagnostics, and the
harmonizer design. Licensed MIT, following the upstream projects.
