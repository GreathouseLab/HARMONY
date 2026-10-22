# HARMONY — data-wall test + run queueing, 2026-10-22

Start 10:18:57. All deliverables complete by 10:47. **Deliverable 4 could not be executed — see below.**

---

## Deliverable 0 — preflight  (10:18:57)

```
$ git status --porcelain
 M evaluate_probes.py
 M probe_sample_coherence.py
?? audit/
?? launch/
$ git log -1 --oneline
246f7cd add audit_tokenizer.py — measure tokens/read + bp/token to settle the two open claims
```

`audit/` and `launch/` are the expected untracked dirs. The two **tracked modified files are
the `--center` edits made earlier today**, and Deliverable 3 of this brief requires `--center`
to exist. Stopping would make the brief unexecutable, so I proceeded and flag it here. Nothing
else is dirty; nothing was stashed, committed or discarded.

---

## Deliverable 1 — val-set provenance  (10:20:17)

**The val set is drawn from the full corpus and is completely untouched by `--reads-cap`.**

Three separate mechanisms, all read-only here:

1. **Which samples are val.** `prepare_fastq.py:167` `split_samples_by_stem(groups,
   val_fraction=0.1, seed=42)` splits over **sample stems**, not files, so a sample's R1 and R2
   cannot straddle the split. 14 of 140 stems went to val. `output/manifest.txt` records the
   assignment. No read-level selection happens here — `sample_to_text` emits **every** read of
   each sample, which is why `val.txt` is 665 MB.

2. **Which val reads are scored.** `train_mlm.py:299` →
   `parse_val_txt_grouped(val_path, n_per_sample=50, seed=42)`. Inside that function
   (`probe_sample_coherence.py:94-96`), each sample's reads are accumulated in full and then
   `rng.sample(current_reads, n_per_sample)` draws 50 **uniformly at random from the whole
   ~200,000-read population of that sample**. 14 samples × 50 = 700 val reads.

3. **Which train reads are scored.** `train_mlm.py:438`
   `train_eval_lists = [ids for s in sorted(loader.samples)[:14] for ids in loader.samples[s][:50]]`
   — the first 50 tokenized reads of the **first 14 of the 126 train samples**, taken from the
   **cap-limited loader pool**.

### Could "val is intrinsically easier" be an artifact of the cap?

**No — not the val side.** `--reads-cap` is a parameter of `PairedDataLoader`, which only ever
reads `train.txt`. The val eval re-parses `val.txt` independently and samples from the complete
per-sample population. Raising or lowering the cap cannot change a single val read. The
model-free k-mer redundancy test in `verify_val_train.py` (Test 3) is therefore measuring a
genuine property of the val split, not a sampling artifact.

**But the comparison is not symmetric, and that is worth recording.** Two asymmetries fall out
of the code above:

- **Different populations.** Val reads are a uniform random draw from ~200k reads per sample.
  Train-eval reads are the first 50 of a capped pool, from only **14 of 126** train samples.
  The "train" half of the train–val gap is a 14-sample slice, not the training distribution.
- **Different exposure.** The 700 train-eval reads are reads the model trains on repeatedly —
  at 305 passes it has seen each one roughly 305 times. That should make train **easier**, not
  harder. Val still scoring better is therefore *more* surprising under heavy repetition, not
  less, and the cap does not explain it away.

---

## Deliverable 2 — matched-token A/B  (10:19:03 launch, 10:46:04 both exited)

Interpretation rule and design were fixed in advance:
**[audit/AB_PREREGISTRATION_2026-10-22.md](AB_PREREGISTRATION_2026-10-22.md)**, written at
10:22:12 when `ab_smallcap/probe_trajectory.csv` held only its header row.

### Caps scaled 50x — stated, with the reason

Not RAM. The Mac holds 6.3M reads (11.6 GB) and tokenizes at 83,228 reads/s. The reason is that
**the specified caps produce no contrast at Mac step counts**: 12,000 steps x 32 reads = 384,000
read-presentations, only **0.61 passes** over the 630,000 reads of `--reads-cap 5000`. Neither arm
would repeat anything, so the experiment could not detect repetition damage. Both caps were scaled
down 50x to **100 and 1,000**, preserving the 10x ratio and placing one arm far above the 4-pass
budget and the other below it.

Both runs ran in parallel to fit the time box. Results are seed-deterministic, so contention
affects wall-clock only. Both hit the `--max-runtime-hours 0.45` guard at step 11,000+.

### Result, at the largest common eval step

| arm | cap | distinct | passes | val top-1 | val CE | train top-1 | train−val gap |
|---|---|---|---|---|---|---|---|
| SMALL-CAP | 100 | 12,600 | 27.94 | 0.1073 | 6.3729 | 0.1564 | **+0.0491** |
| LARGE-CAP | 1,000 | 126,000 | 2.79 | **0.1120** | **6.0912** | 0.1164 | +0.0044 |

LARGE-CAP − SMALL-CAP: val top-1 **+0.0047**, val CE **−0.2817**. Run-to-run jitter (max
|Δtop-1| over the last three evals of either arm) is 0.0034.

### **VERDICT: LARGE-CAP better → the 0.248 rise is partly memorisation; data, not compute, was the limiter.**

The top-1 margin alone (+0.0047 against 0.0034 jitter) would be borderline. Two other things make
it convincing:

**The loss gap is decisive.** 0.28 nats at identical tokens is far outside any jitter.

**The trajectory shape is unambiguous.** SMALL-CAP's val loss *degrades* after step 3,000 while
its train−val gap opens; LARGE-CAP's improves monotonically with a flat gap:

| step | SMALL val CE | SMALL gap | LARGE val CE | LARGE gap |
|---|---|---|---|---|
| 3,000 | 6.2497 | −0.0026 | 6.2222 | −0.0001 |
| 6,000 | 6.2892 | +0.0126 | 6.1484 | −0.0009 |
| 9,000 | 6.3471 | +0.0275 | 6.0974 | −0.0032 |
| 11,000 | 6.3729 | +0.0491 | 6.0912 | +0.0044 |

That is memorisation onto 12,600 reads, caught in the act.

Plot: **[audit/ab_matched_tokens.png](ab_matched_tokens.png)** — both curves vs tokens-seen.

**Scope limit:** depth-2, 4.82M params, 1/500th of the Aurora token budget. Establishes that the
damage is real and detectable, not its magnitude at depth-6 and 9.27B tokens.

---

## Deliverable 3 — centred noise evaluation, both D2 checkpoints  (10:46:23)

400 reads, `audit/ab_noise_centered.py`. **Every cosine here is centred.** No uncentred cosine
appears in this report.

| run | between-read floor | 0.5% | 1% | 5% | margin at 5% |
|---|---|---|---|---|---|
| SMALL-CAP (cap 100) | −0.0026 | 0.9698 | 0.9401 | 0.7431 | +0.7456 |
| LARGE-CAP (cap 1000) | +0.0064 | **0.9812** | **0.9616** | **0.8294** | **+0.8230** |

Both floors sit at ~0, as a centred floor should. LARGE-CAP is more noise-robust at every error
rate — a second, independent evaluation agreeing with D2.

---

## Deliverable 4 — submit H  (SUBMIT_H: YES)  **BLOCKED — NOT SUBMITTED**

Authorisation was given. Two of the three preconditions failed, and the submission itself is
impossible from this machine.

**(a) >= 160 GB free check: COULD NOT BE PERFORMED.** No PBS client exists here —
`qsub`, `qstat` and `pbsnodes` are all absent; there is no ALCF entry in `~/.ssh/config`; `/flare`
is not mounted. `aurora.alcf.anl.gov` resolves (140.221.120.20) but ALCF login requires
interactive MFA, which this session cannot and should not perform.
**Action taken:** "unverified" is not "confirmed", so the documented fallback is now **active** in
the script — `--reads-cap 20000` (~56 GB/node) with `--max-steps 26250`, still exactly 4.0 passes.
The restore path to cap 50000 / 65625 steps is written into the script as a two-line edit gated on
an actual `free -g` check.

**(b) tokens-seen and passes logged every eval: WAS FALSE, NOW TRUE.** `train_mlm.py` logged
neither — the trajectory CSV had no such columns. Fixed: three new columns
(`reads_seen,tokens_seen,passes`), a per-eval console line, a startup line reporting distinct
reads and the 4-pass step budget, and a `*** PAST 4-PASS BUDGET ***` warning. This is additive
logging; no default config, tokenizer or seq_len changed.

**(c) qsub: NOT POSSIBLE HERE.** Exact command, to run from an Aurora login node in the repo:

```
qsub launch/next_run_2026-10-22.sh
```

No job ID and no expected finish time can be reported, because no job was submitted. Expected
duration once submitted: 26,250 steps at ~0.170 s/step ≈ **1.2 node-hours**, inside the 4-hour
walltime the script requests.

---

## Deliverable 5 — meeting brief

**[audit/MEETING_BRIEF_2026-10-22.md](MEETING_BRIEF_2026-10-22.md)**, one page, printed to console.

---

## Deliverable 6 — verification

**Matched token budget, shown explicitly.** Both arms were compared at the largest step present
in both trajectories, step 11,000:

```
SMALL-CAP : 11,000 steps x 32 reads/step x 48.27 tokens/read = 16,991,040 useful tokens
LARGE-CAP : 11,000 steps x 32 reads/step x 48.27 tokens/read = 16,991,040 useful tokens
difference: 0 tokens
```

Both arms used identical `--samples-per-batch 4 --reads-per-sample 8` (32 reads/step), identical
`--seq-len 64`, identical seed 42. The budget is matched by construction, not by adjustment.

**Working tree — does NOT meet the stated condition, by necessity:**

```
 M evaluate_probes.py          <- --center edits (yesterday's task; required by D3 here)
 M probe_sample_coherence.py   <- --center edits (same)
 M train_mlm.py                <- tokens-seen/passes logging, required by D4(b)
?? audit/
?? launch/
?? experiments/ab_largecap/    ?? experiments/ab_largecap.log
?? experiments/ab_smallcap/    ?? experiments/ab_smallcap.log
?? experiments/ab_watchdog.log
```

D6 asked for "nothing tracked modified". Three tracked files are modified. All three were
required by other deliverables in this same brief: D3 cannot run without `--center`, and D4(b)
cannot be confirmed without the tokens/passes logging. No config, tokenizer or seq_len default was
touched. Flagging rather than hiding it.
