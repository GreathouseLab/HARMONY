# HARMONY

Phase −1 feasibility study for HARMONY — a perception layer for microbiome data that turns raw sequencing reads into bias-corrected representations.

## What this repo is

A causal-LM autoresearch loop, adapted for genomic FASTQ on macOS / MPS, used as the smallest experiment that can answer two questions about HARMONY before committing to Argonne A100 compute:

1. Can language models learn meaningful patterns from raw microbiome reads at all?
2. What is the empirical scaling slope (loss vs compute, à la Chinchilla)?

The full architectural vision (dual-head read-error-correction + community-level bias-correction, k-mer tokenization, etc.) lives in [ARCHITECTURE_V2_REDESIGN.md](ARCHITECTURE_V2_REDESIGN.md). The current Phase −1 status, open questions, and probe gate spec live in [program.md](program.md). The unmodified upstream `program.md` is preserved as [program_OG.md](program_OG.md) for reference.

## Status (2026-04-29)

- **38 R1–R3 hyperparameter experiments** completed via the LLM-driven loop; best val_bpb = 1.932465 (R2 winner).
- **Probe 4 (synthetic noise robustness)** run on four representative checkpoints, reported in [experiments/probe4_summary.md](experiments/probe4_summary.md). Across-checkpoint cosine similarity at 1% noise is *anti-correlated* with val_bpb. Per the gate in program.md §6, this points toward an MLM + contrastive arm rather than further val_bpb optimization — but Probes 1/2/3 are still needed for full evaluation, and they are blocked on the val-split disposition decision (see below).
- **Reproducibility caveat.** Re-running each of the four chosen R1–R3 configs under seed=42 produced val_bpb drift averaging 0.008 bpb, with the relative ranking changing substantially. MPS run-to-run variance is on the same order as the val_bpb improvements being claimed; absolute val_bpb deltas <0.02 between runs should not be treated as load-bearing.
- **Paired-end leakage fix** committed 2026-04-29 (`prepare_fastq.py`). The previous file-level train/val split could place R1 in train and R2 in val for the same molecule. New split is at sample-stem level, and paired reads now emit as one molecule per `<READ_START>` block via `<r1_seq> <PAIRED_END> <reverse_complement(r2_seq)>`. **All 38 historical val_bpb numbers are inflated by the prior leak; R4-and-onward is the new baseline.** Historical numbers are preserved in `experiments/results.csv` for provenance.

## Data pipeline

```
  FASTQ (.fastq.gz)
       │
       ▼  prepare_fastq.py     paired-end aware; sample-stem split; <PAIRED_END> joining
  output/{train,val}.txt
       │
       ▼  prepare_genomic.py   BPE tokenizer, dataloader, BPB evaluation
  ~/.cache/autoresearch/
       │
       ▼  train.py             single-file GPT trainer (5-minute wall budget)
  val_bpb + experiments/<run>/checkpoint.pt
       │
       ▼  evaluate_probes.py   frozen-checkpoint representation probes (Probe 4 only for now)
  experiments/probe_*.csv, probe_*_summary.md
```

## Quick start

Apple Silicon / MPS or a single NVIDIA GPU; Python 3.10+; [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                                          # install runtime deps
uv sync --group dev                                              # add pytest

uv run prepare_fastq.py --input-dir <fastq_dir> --output-dir output
uv run prepare_genomic.py                                        # train BPE tokenizer
uv run train.py                                                  # 5-minute run; saves checkpoint if HARMONY_CHECKPOINT_PATH set

uv run pytest tests/ -v                                          # unit tests
```

## Autoresearch loop

`autoresearch_llm.py` proposes hyperparameter configurations via Claude Sonnet, runs `train.py` with a 5-minute time budget, fast-fails any run whose first 3 post-warmup steps average >8 s (saves ~14 min per doomed config on MPS), and appends results to `experiments/results.csv`. Each subprocess gets `HARMONY_CHECKPOINT_PATH` set so checkpoints are written automatically for downstream probing.

```bash
ANTHROPIC_API_KEY=...  uv run autoresearch_llm.py --max-experiments 12
```

## Probe gate

`evaluate_probes.py` consumes saved checkpoints and runs frozen-weight probes:

- **Probe 4 — noise robustness** (implemented). Sample 1000 val reads, generate noisy versions at 0.5/1/5% substitution rate at the DNA-sequence level (pre-tokenization), embed clean and noisy via mean-pool over DNA token positions, report cosine similarity per checkpoint × error rate.
- **Probes 1, 2, 3** (deferred). Require resolving the val-split disposition (program.md §5 Concern 2) and producing a read→sample provenance index for the val stream. See [program.md §6](program.md) for the full gate spec.

```bash
uv run evaluate_probes.py            # probe 4 across checkpoints listed in the script
```

## Repo layout

```
prepare_fastq.py             FASTQ → text stream (paired-end aware, sample-stem split)
prepare_genomic.py           BPE tokenizer, dataloader, evaluation
train.py                     single-file GPT trainer (5-min wall budget)
model.py                     GPT, MuonAdamW, helpers (importable)
autoresearch_llm.py          Claude-driven hyperparameter loop
evaluate_probes.py           frozen-checkpoint representation probes (Probe 4)
reproduce_checkpoints.py     one-shot orchestrator for re-running historical winners

program.md                   current Phase −1 status, probe gate, open questions
program_OG.md                unmodified upstream program.md (preserved for reference)
ARCHITECTURE_V2_REDESIGN.md  broader HARMONY architecture vision

experiments/                 per-run logs, results.csv, probe outputs
tests/                       pytest suite (currently: prepare_fastq.py — 33 tests)
```

## Lineage

Forked from [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos), itself a macOS / MPS adaptation of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The original autoresearch single-file architecture is preserved here; HARMONY-specific work is concentrated in the data pipeline (`prepare_fastq.py`, `prepare_genomic.py`), the LLM-driven loop (`autoresearch_llm.py`), the probe driver (`evaluate_probes.py`), and the new `program.md`.

To pull future upstream fixes:

```bash
git fetch upstream
git merge upstream/master   # `upstream` remote already wired to miolini/autoresearch-macos
```

## License

MIT, inherited from the upstream autoresearch-macos and karpathy/autoresearch projects.
