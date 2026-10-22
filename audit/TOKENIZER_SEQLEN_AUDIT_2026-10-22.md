# HARMONY audit — tokenizer compression and seq_len vs read length

**Date:** 2026-10-22
**Scope:** audit only. No file outside `audit/` was created, modified or deleted.

---

## Deliverable 0 — dirty-state preflight

```
$ git status --porcelain
(no output)
$ git log -1 --oneline
246f7cd add audit_tokenizer.py — measure tokens/read + bp/token to settle the two open claims
$ git rev-parse --abbrev-ref HEAD
feat/tokenizer-audit
```

Branch `feat/tokenizer-audit`, HEAD `246f7cd`, **tree clean**. No uncommitted changes, so the
audit proceeded without stopping.

---

## Deliverable 1 — tokenizer and provenance

| Property | Value | Source |
|---|---|---|
| Artifact | `~/.cache/autoresearch/tokenizer/tokenizer.pkl` (49,258 bytes) | `prepare_genomic.py:42-43`, `:168-171` |
| Companion | `~/.cache/autoresearch/tokenizer/token_bytes.pt` (17,584 bytes) | `prepare_genomic.py:199-200` |
| Artifact mtime | 2026-04-29 18:43 | `ls -la` |
| Loaded by | `Tokenizer.from_directory()` | `train_mlm.py:35`, `:414` |
| Type | BPE, trained with `rustbpe` | `prepare_genomic.py:99-102` |
| Vocab size **as stored** | **4096** (measured via `get_vocab_size()`, not assumed) | audit script output |
| Composition | 4,090 learned merges + 6 special tokens | `prepare_genomic.py:100`, `:47-57` |
| Split pattern | `[ACGTN]+\|[^ACGTN]+` | `prepare_genomic.py:63` |
| Special tokens | `<\|bos\|>`, `<SAMPLE_START>`, `<SAMPLE_END>`, `<READ_START>`, `<READ_END>`, `<PAIRED_END>` | `prepare_genomic.py:47-57` |

`train_mlm.py:46` sets `EXTENDED_VOCAB = 4097`, adding `[MASK]` = 4096 on top of the stored 4096.

**Training corpus of the tokenizer — partially determined.** `train_tokenizer(train_path, vocab_size)`
(`prepare_genomic.py:83`) accepts the corpus path as an argument; no config, log or manifest in the
repo records which path was passed. The artifact's mtime (2026-04-29 18:43) is 19 minutes after
`output/train.txt` was written (18:24 the same day), and that date is the paired-end leak fix, which
is circumstantial evidence it was retrained on the post-fix corpus. **Finding: the tokenizer's
training corpus is not recorded anywhere in the repo and rests on mtime inference.** The open
question in `ARCHITECTURE_V3_UNIFIED.md:148` — whether the vocabulary is WGS-only or a stratified
multi-protocol mix — therefore cannot be answered from the repo.

---

## Deliverable 2 — measured bp/token on the training corpus

Data the trainer consumes: `output/train.txt` (5,986,623,438 bytes, 2026-04-29), pre-tokenization
text. Tokenized live by the loader at `paired_data_loader.py:56-57`; no `.bin` is used.

Script: `audit/measure_bp_per_token.py`. It samples complete `<READ_START> … <READ_END>` records
from random byte offsets across the whole 6 GB file (not the head), and tokenizes each with the
exact loader convention `f"<READ_START> {body} <READ_END>"`.

```
$ .venv/bin/python3.12 audit/measure_bp_per_token.py --txt output/train.txt --n 10000 --seed 42
========================================================================
HARMONY tokenizer audit — bp/token and tokens/read
========================================================================
corpus          : output/train.txt
tokenizer       : ~/.cache/autoresearch/tokenizer/tokenizer.pkl
vocab_size      : 4096
records sampled : 10,000   (seed=42)

tokens per read (incl. <READ_START>/<READ_END>)
  mean   : 48.27
  median : 48.0
  p5     : 45
  p95    : 51
  min/max: 35 / 55

read length (DNA bases, both mates, excl. special tokens)
  mean   : 198.53 bp
  median : 200.0 bp

compression
  bp/token (DNA tokens only) : 4.290
  bp/token (incl. wrappers)  : 4.113

truncation exposure
  reads >  64 tokens : 0 / 10,000 = 0.00%
  reads >  96 tokens : 0 / 10,000 = 0.00%
  reads > 128 tokens : 0 / 10,000 = 0.00%
========================================================================
```

**Note on units.** One record is a paired-end mate pair — two 100 bp mates joined by `<PAIRED_END>`,
hence 198.53 bp mean, not 150. Per 150 bp of DNA this tokenizer produces roughly **35 tokens**.

---

## Deliverable 3 — effective seq_len and truncation behaviour

### Where seq_len is set

| Location | Value | Note |
|---|---|---|
| `paired_data_loader.py:22` | `SEQ_LEN_DEFAULT = 64` | loader default |
| `paired_data_loader.py:33` | `seq_len: int = SEQ_LEN_DEFAULT` | constructor default |
| `paired_data_loader.py:43` | `max_read_len = max_read_len or seq_len` | per-read drop threshold |
| `train_mlm.py:697` | `--seq-len` default `64` | CLI default |
| `train_mlm.py:717` | `--seq-len-max` default `0` | 0 = curriculum OFF |
| `train_mlm.py:266` | `build_model(..., seq_len=64)` | model-build default |
| `train_mlm.py:421` | `model_seq_len = seq_max` | model built at the ceiling |
| `run_ddp_aurora.sh:41` | `--seq-len 64` | production Aurora run |
| `run_ddp_depth8.sh:38` | `--seq-len 64` | depth-8 scaling probe |
| `run_ddp_2node.sh:41` | `--seq-len 64` | two-node validation |
| `run_ddp_curriculum.sh:41` | `--seq-len 64 --seq-len-max 512` | **only** curriculum launcher |

### What happens to an over-length read

It is **dropped, not truncated** (`paired_data_loader.py:58-60`):

```python
ids = tokenizer.encode(text)
if len(ids) > self.max_read_len:
    n_dropped_too_long += 1
    continue
```

Truncation exists in exactly one place, and only for a read that alone exceeds the window
(`paired_data_loader.py:104`): `row.extend(r[:T])` — a head-side truncation of the first read in a
row. With `max_read_len` defaulting to `seq_len`, no read reaching `_pack_row` can trigger it.

### Read boundaries and multi-read contexts

Every record is wrapped in `<READ_START>` / `<READ_END>` before tokenization
(`paired_data_loader.py:56`), and both are real tokens in the 4,096 vocabulary, so boundaries are
visible to the model. `_pack_row` (`paired_data_loader.py:91-109`) concatenates whole reads from one
sample until the next would overflow.

At T=64 a 48-token read leaves 16 tokens of headroom — not enough for a second 48-token read — so
**each row holds exactly one read plus padding**, and the model does **not** see two reads in one
context. That changes only under `run_ddp_curriculum.sh`, where T grows to 512 and a row packs
roughly 10 reads.

### Cross-reference with Deliverable 2

**0.00%** of training reads are currently truncated or dropped. Mean tokens per read is 48.27
against a 64-token window; the longest record in 20,000 sampled was 56 tokens. Average unused
headroom is about 15.7 tokens per row, i.e. **~25% of each context is padding**.

---

## Deliverable 4 — what the logs say

Every Mac run that used this loader logged a zero drop count:

```
mlm_bigcohort.log        [paired_loader] parsed 126 samples, 630000 reads (dropped 0 reads > seq_len=64 tokens)
mlm_bigcohort_d4.log     [paired_loader] parsed 126 samples, 630000 reads (dropped 0 reads > seq_len=64 tokens)
mlm_bigcohort_d6.log     [paired_loader] parsed 126 samples, 630000 reads (dropped 0 reads > seq_len=64 tokens)
mlm_memcheck.log         [paired_loader] parsed 126 samples,   6300 reads (dropped 0 reads > seq_len=64 tokens)
mlm_moredata.log         [paired_loader] parsed 126 samples, 252000 reads (dropped 0 reads > seq_len=64 tokens)
mlm_sweep_A_control.log  [paired_loader] parsed 126 samples,  25200 reads (dropped 0 reads > seq_len=64 tokens)
mlm_sweep_B_capoff.log   [paired_loader] parsed 126 samples,  25200 reads (dropped 0 reads > seq_len=64 tokens)
mlm_sweep_C_mask10.log   [paired_loader] parsed 126 samples,  25200 reads (dropped 0 reads > seq_len=64 tokens)
mlm_sweep_D_both.log     [paired_loader] parsed 126 samples,  25200 reads (dropped 0 reads > seq_len=64 tokens)
mlm_vanilla.log          [paired_loader] parsed 126 samples,  25200 reads (dropped 0 reads > seq_len=64 tokens)
```

This agrees with Deliverable 3: seq_len=64, zero reads lost.

**Aurora runs cannot be checked from this repo.** No Aurora stdout, result JSON or trajectory CSV is
present locally — `experiments/` holds no artifact recording `seq_len`, `sequence_len` or
`cur_seq_len`, because loop logs and per-step artifacts are gitignored. The Aurora configuration is
therefore established from the launcher scripts (all `--seq-len 64`) rather than from a run record.

**No run with a longer seq_len or a retrained tokenizer was found.** `run_ddp_curriculum.sh`
(seq-len 64→512) exists on branch `feat/seqlen-curriculum`, merged at `1a0a55b`, but no artifact in
this repo shows it has been executed.

---

## Deliverable 5 — verdict on the three readings

| Reading | Verdict | The one measurement that decides it |
|---|---|---|
| (a) the tokenizer underperforms on our corpus | **REFUTED as stated** | Measured **4.290 bp/token**, not the claimed ~2.0. It sits modestly below the 5–6 bp/token expectation, but there is no 3× inefficiency. |
| (b) the 75-token figure was wrong | **SUPPORTED** | A 198.53 bp record tokenizes to **48.27 tokens**; scaled to 150 bp that is ~35 tokens, not 75 — wrong by more than 2×. |
| (c) reads are ~75 tokens and seq_len is 64, so every read is truncated | **REFUTED** | **0 / 10,000** sampled reads exceed 64 tokens; the longest was 55. Confirmed independently by ten training logs reporting 0 reads dropped. |

**Has the model currently in training ever seen a complete read?**

Yes. Every read fits inside the 64-token window with about 16 tokens to spare, and both read-boundary
markers are present in the context. No read has been truncated or dropped in any logged run. The
model has never seen *two* reads in one context, because two 48-token reads do not fit in 64 — that
is what the 64→512 curriculum is built to change, and it has not yet been run.

---

## Deliverable 6 — verification

**Reproducibility, different seed:**

| Seed | bp/token (DNA) | Mean tokens/read |
|---|---|---|
| 42 | 4.290 | 48.27 |
| 1337 | 4.291 | 48.28 |

Movement: **0.02%**, well inside the 2% tolerance.

**Working tree after the audit:**

```
$ git status --porcelain
?? audit/
```

Only the new `audit/` directory is untracked. Nothing tracked was modified.

---

## Summary

1. The tokenizer packs **4.29 bp per token** — below the 5–6 rule of thumb, but nowhere near the
   feared 2.0; no token-count estimate on the project inherits a 3× error.
2. A read is **48 tokens**, not 75. The 75-token figure was simply wrong.
3. **Nothing is being truncated.** Zero reads exceed the 64-token window, confirmed by measurement
   and by ten independent training logs.
4. The model **has** always seen complete reads with their boundary markers — but only ever **one
   read at a time**, so no cross-read learning has happened yet.
5. The real inefficiency is the opposite of the one suspected: **~25% of every context is padding**,
   and the 64→512 curriculum that would use it has been built but never run.
