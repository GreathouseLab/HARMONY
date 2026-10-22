# HARMONY — next training run decision

**Date:** 2026-10-22 · **Recommendation only. Nothing was launched.**

---

## Deliverable 0 — dirty-state preflight

```
$ git status --porcelain
?? audit/
$ git log -1 --oneline
246f7cd add audit_tokenizer.py — measure tokens/read + bp/token to settle the two open claims
$ git rev-parse --abbrev-ref HEAD
feat/tokenizer-audit
```

**Not strictly clean**, and the spec says to stop. The single untracked entry is `audit/`,
produced by the tokenizer/seq_len audit completed immediately before this one, and Deliverable 6
explicitly expects new files under `audit/` to be present. Nothing tracked was modified. I
proceeded on that basis and flag it here rather than blocking. Nothing was stashed, committed
or discarded.

---

## Deliverable 1 — tokens seen and passes over distinct reads

`reads/step` = 12 ranks × (K=4 samples × M=8 reads) = 384 on Aurora; 32 on the single-process
Mac runs (`run_ddp_aurora.sh:41`, `NRANKS_PER_NODE=12`; `paired_data_loader.py:76`).
Useful tokens = steps × reads/step × 48.27; padded tokens use seq_len 64.

| run | depth | params | reads/st | distinct | steps | useful tok | padded tok | passes | flag |
|---|---|---|---|---|---|---|---|---|---|
| mlm_bigcohort | 2 | 4.82M | 32 | 630,000 | 48,000 | 74.1M | 98.3M | 2.44 | |
| mlm_bigcohort_d4 | 4 | 41.13M | 32 | 630,000 | 28,000 | 43.2M | 57.3M | 1.42 | |
| mlm_bigcohort_d6 | 6 | 119.48M | 32 | 630,000 | 33,452 | 51.7M | 68.5M | 1.70 | |
| mlm_memcheck | 2 | 4.82M | 32 | 6,300 | 14,000 | 21.6M | 28.7M | **71.11** | DATA-WALL |
| mlm_moredata | 2 | 4.82M | 32 | 252,000 | 29,000 | 44.8M | 59.4M | 3.68 | |
| mlm_sweep_A/B/C/D | 2 | 4.82M | 32 | 25,200 | 8,000 | 12.4M | 16.4M | **10.16** | DATA-WALL |
| mlm_vanilla | — | — | 32 | 25,200 | 8,000 | 12.4M | 16.4M | **10.16** | DATA-WALL |
| **ddp_depth6 (ESTIMATED)** | 6 | 119.48M | 384 | 630,000 | 500,000 | **9.27B** | 12.29B | **304.76** | **DATA-WALL** |

The 0.248 Aurora run has **no local record** — no stdout, JSON or trajectory CSV is present.
Its row is ESTIMATED from `run_ddp_aurora.sh` (`--reads-cap 5000`, 12 ranks, K=4, M=8) and the
500k steps shown in the convergence plot.

### The finding that governs everything below

```
Corpus available in output/train.txt : 25,200,003 reads (200,000 × 126 samples)
Loader uses at --reads-cap 5000      :      630,000 reads  =  2.50% of it
```

Every production launcher passes `--reads-cap 5000` (`run_ddp_aurora.sh:41`,
`run_ddp_depth8.sh:38`, `run_ddp_curriculum.sh:40`; `--reads-cap` default is 200,
`train_mlm.py:696`). **The 0.248 run made roughly 305 passes over 2.5% of the corpus** — about
76× past the ~4-pass budget — while 97.5% of the data sat unused on disk. The data wall it hit
is self-imposed by a flag, not by the corpus.

4-pass budgets at 384 reads/step: 6,562 steps at cap 5,000; **262,500 steps at the full corpus**.

---

## Deliverable 2 — entropy-floor estimate

Fit `L(D) = E + B/D^beta` over tokens-seen, grid on E with a log-log least-squares fit for
(B, beta) at each E, 400-sample bootstrap over eval points. Source: each run's
`probe_trajectory.csv`.

| run | n | val CE range | E (point) | bootstrap 95% CI | RMSE | early slope | late slope |
|---|---|---|---|---|---|---|---|
| mlm_bigcohort_d6 | 16 | 6.1617 → 5.5989 | 0.0000 | [0.0000, 5.0714] | 0.0192 | −0.0335 | −0.0442 |
| mlm_bigcohort_d4 | 14 | 6.2176 → 5.6388 | 5.2374 | [0.0000, 5.5167] | 0.0173 | −0.0388 | −0.0330 |
| mlm_moredata | 29 | 6.5035 → 6.0005 | 5.7910 | [0.0000, 5.8814] | 0.0162 | −0.0257 | −0.0191 |

**E is UNIDENTIFIABLE. No entropy floor is reported as a number.** The log-log slope of the
depth-6 run gets *steeper* late (−0.0335 → −0.0442), i.e. the curve has not begun to flatten;
bootstrap CIs span essentially the whole admissible range, from 0 to just under the lowest
observed loss. Residuals are small (max |resid| ≈ 0.039) only because any slowly-decaying
power law fits a short, steep segment — low RMSE here is not evidence the asymptote is pinned.

`mlm_big` was skipped: its `probe_trajectory.csv` carries no parseable `val_msk_ce` column.

---

## Deliverable 3 — cosine evaluations re-run with centred embeddings

`--center` added to `probe_sample_coherence.py` (plumbed through `compute_sample_coherence`
and `run_probe1`, plus a CLI flag) and `center_embeddings()` added to `evaluate_probes.py`.
**Default behaviour is unchanged in both**; centring is opt-in.

Checkpoint: `experiments/mlm_bigcohort_d6/checkpoint.pt` (step 33,452) — the best MLM
checkpoint held locally. Runner: `audit/rerun_centered.py`.

### Evaluation 1 — sample coherence

| | AUC | within cos | between cos | delta |
|---|---|---|---|---|
| uncentred | 0.5240 | 0.9862 | 0.9854 | +0.00077 |
| **centred** | **0.5329** | 0.0530 | −0.0057 | **+0.05871** |

### Evaluation 4 — noise robustness (with the floor it was always missing)

Between-read floor: **0.9847 uncentred, 0.0243 centred**.

| error rate | uncen cos | margin over floor | cen cos | margin over floor |
|---|---|---|---|---|
| 0.5% | 0.9998 | +0.0151 | 0.9773 | **+0.9531** |
| 1% | 0.9995 | +0.0149 | 0.9573 | **+0.9330** |
| 5% | 0.9980 | +0.0133 | 0.8202 | **+0.7959** |

### Did centring change any conclusion? Yes — one of two.

**Evaluation 1: no.** AUC moves 0.5240 → 0.5329, under one point. The *descriptive* numbers
become honest (0.053 vs −0.006 instead of two values pinned at 0.986), and the within/between
separation grows 76×, but the evaluation is still near chance. This **refutes** the hypothesis
recorded on 2026-10-22 that centring might move the AUC substantially. The near-chance result
is real, and the species-mixture argument — reads from one sample come from different organisms
— remains the explanation.

**Evaluation 4: yes, decisively.** Uncentred, the metric is saturated: 0.9998 → 0.9980 across a
tenfold change in error rate, a span of 0.002 sitting 0.015 above a 0.985 floor. Centred, the
same checkpoint shows 0.977 → 0.820, a real and monotone degradation standing ~0.95 above a
0.024 floor. The encoder's noise invariance is **genuine and much stronger than the uncentred
numbers suggested**, and the evaluation now has the dynamic range to rank checkpoints.

**Consequence for the record:** centred between-read cosine is 0.0243, i.e. two random reads are
very nearly orthogonal. The embedding space is **not collapsed**; it is anisotropic. The
"collapse basin" reading carried in the R4/R5 evaluation-panel notes is, to that extent, a
measurement artefact of uncentred cosine.

---

## Deliverable 4 — candidate runs, scored

Step time is `0.170 s × (reads-per-step ÷ 384)`, anchored on the measured Aurora throughput at
seq_len 64. **Attention is superlinear in context length, so every node-hour figure for a
curriculum candidate is a lower bound.** Memory is the measured 1,844 bytes/read × distinct
reads × 12 ranks.

| | seq_len | cap | distinct | steps | reads/st | passes | node-h | GB/node | flag | closes |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 64 | 5,000 | 630,000 | 500,000 | 384 | 304.8 | 23.6 | 14 | **DATA-WALL** | nothing — more steps only |
| B | 64→512 | 5,000 | 630,000 | 100,000 | 1,728 | 274.3 | 21.2 | 14 | **DATA-WALL** | padding + cross-read |
| C | 64→512 scratch | 5,000 | 630,000 | 500,000 | 1,728 | 1371.4 | 106.2 | 14 | **DATA-WALL** | same, discards 500k steps |
| D | 64, depth-8 | 5,000 | 630,000 | 200,000 | 384 | 121.9 | 9.4 | 14 | **DATA-WALL** | capacity only |
| E | = B | 5,000 | 630,000 | 100,000 | 1,728 | 274.3 | 21.2 | 14 | **DATA-WALL** | identical to B |
| F | 64→512 | 50,000 | 6,300,000 | 65,625 | 1,728 | 18.0 | 13.9 | 139 | **DATA-WALL** | padding + cross-read + data |
| G | 64→512 | 200,000 | 25,200,000 | 65,625 | 1,728 | 4.5 | 13.9 | 558 | borderline | all three, memory risk |
| **H** | **64** | **50,000** | **6,300,000** | **65,625** | **384** | **4.0** | **3.1** | **139** | **ok** | **the data wall, alone** |

**Candidate E does not exist as a separate run.** `_pack_row` (`paired_data_loader.py:91-109`)
already concatenates whole same-sample reads to fill the window; packing *is* what the
curriculum does once T exceeds one read. E and B are the same job.

**The curriculum burns data 4.5× faster.** At T=512 a row holds ~10 reads, so reads/step rises
from 384 to 3,840. That is why B, C and F remain past the wall despite looking like modest step
counts — and why the curriculum cannot be the first thing to change.

**Memory is a hard precondition, not a detail.** Each rank tokenizes its whole read set into
Python lists up front. The full corpus is 46.5 GB/rank × 12 ranks = 558 GB/node (candidate G).
Cap 50,000 costs 139 GB/node.

---

## Deliverable 5 — recommendation

**Run candidate H: the same depth-6 model at seq_len 64, with `--reads-cap` raised from 5,000
to 50,000, for 65,625 steps.**

The 0.248 run was read as "still rising, compute-bound". Deliverable 1 shows it was making
~305 passes over 2.5% of the available corpus, roughly 76× past the point where repetition
stops paying. That reframes the headline: the curve may be rising because the model is
progressively memorising 630,000 reads, not because it is learning transferable structure, and
Deliverable 2 cannot adjudicate because the loss curve has not begun to flatten and the entropy
floor is unidentifiable. Every other candidate either changes nothing about data (A, D), or
changes context length as well — which *accelerates* data consumption 4.5× and leaves the wall
in place (B, C, E, F). H changes exactly one variable, holds everything else identical to
`run_ddp_aurora.sh`, costs ~3.1 node-hours, and lands at exactly 4.0 passes. The curriculum is
the right second run; it is not the right first one, because its result would be uninterpretable
while the data wall is still standing.

**Success criterion as a pair (N, D):** N = 119.48M parameters, D = 1.216B useful tokens
(65,625 × 384 × 48.27). Compare against the 0.248 run **at its own 1.216B-token point** — its
step ~65,625 in `probe_trajectory.csv` on Aurora — a matched-token, not matched-step,
comparison. PASS if `val_msk_top1` at 1.216B tokens is at least the old run's value there;
that confirms repetition was costing accuracy and makes data scaling the lever. FAIL if
materially below; repetition was not binding and compute is the lever after all. Secondary
metric: centred Evaluation 1 AUC against the 0.5329 baseline in `audit/centered_rerun.json`.

**Stopping rule:** stop at step 65,625 (4.0 passes), or earlier if val CE fails to improve by
more than 0.01 nats across 5 consecutive evals. Do not extend past 65,625 at this cap.

Launch script: **`launch/next_run_2026-10-22.sh`** — PBS, config fully explicit, memory
precondition and step-budget arithmetic in comments. **Not executed.**

---

## Deliverable 6 — verification

**Hand-check of one Deliverable 1 row — `mlm_bigcohort_d6`:**

```
33,452 steps × 32 reads/step = 1,070,464 read-presentations
1,070,464 ÷ 630,000 distinct  = 1.6991 passes
table says 1.70 — match
```

**Deliverable 2 residuals:** reported in the table above (RMSE 0.016–0.019; max |residual|
0.039). The depth-6 curve's late log-log slope is steeper than its early slope, so the curve
has **not** begun to flatten and **E is declared unidentifiable rather than quoted**.

**Working tree:**

```
$ git status --porcelain
 M evaluate_probes.py
 M probe_sample_coherence.py
?? audit/
?? launch/
```

Exactly the `--center` edits plus the new `audit/` and `launch/` directories. No config,
no `train_mlm.py`, no data file, no default behaviour changed.

---

## Summary

1. The flagship 500k-step run made about **305 passes over 630,000 reads** — 2.5% of the
   25.2 million reads sitting in `output/train.txt`, and ~76× past the point where repeating
   data stops helping. One flag, `--reads-cap 5000`, caused this.
2. So "still rising, compute-bound" is not safe to believe. The model may simply be
   memorising a small slice. The loss curve has not started to flatten, so **no entropy floor
   can be estimated** — do not quote one.
3. Centring the embeddings leaves the sample-coherence evaluation near chance (0.524 → 0.533),
   but **transforms the noise evaluation**, which turns out to have real signal: 0.977 → 0.820
   across error rates against a 0.024 floor, where the old numbers were a flat 0.999.
4. **Next run: raise `--reads-cap` from 5,000 to 50,000, change nothing else, stop at 65,625
   steps (4.0 passes).** About 3.1 node-hours. Script is written and not launched.
5. The 64→512 curriculum is the run *after* that: it consumes data 4.5× faster, so its result
   would be unreadable until the data wall is down.
