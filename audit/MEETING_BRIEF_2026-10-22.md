# HARMONY — 10-minute brief, 2026-10-22

## 1. We have been training on 2.5% of our data, and re-reading it 305 times

`output/train.txt` holds **25,200,003 reads**. Every production run so far set a flag,
`--reads-cap 5000`, that uses only **630,000** of them — 2.5%.

The flagship run (val top-1 0.248, "still rising") therefore made about **305 passes** over that
small slice. A pass is one trip through the data; past roughly **4 passes**, re-reading the same
data stops buying accuracy and starts buying memorisation. We were **76x** past that line.

## 2. We tested it today, and repetition is measurably hurting us

Two training runs on the Mac, **identical in every way except how much distinct data they could
see**, stopped at the **same token budget** (16,991,040 tokens — same compute, same model, same
seed). The only difference: one re-read its data 27.9 times, the other 2.8 times.

| | distinct reads | passes | val top-1 | val loss | train−val gap |
|---|---|---|---|---|---|
| SMALL-CAP | 12,600 | 27.9 | 0.1073 | 6.3729 | **+0.0491** |
| LARGE-CAP | 126,000 | 2.8 | **0.1120** | **6.0912** | +0.0044 |

Under the rule we wrote down **before** running: **LARGE-CAP better → the 0.248 rise is partly
memorisation; data, not compute, was the limiter.**

The clearest signal is not the accuracy gap, it is the shape. SMALL-CAP's validation loss **got
worse after step 3,000** (6.2497 → 6.3729) while its train−val gap opened from −0.003 to +0.049 —
the model learning its 12,600 reads by heart instead of learning biology. LARGE-CAP improved
steadily throughout and its gap stayed at zero. In biomarker terms: the first model found a
signature that fits its discovery cohort beautifully and does not replicate; the second found one
that holds up.

Honest caveat: this is a small model at 1/500th the compute of the real run. It shows the damage
is **real and detectable**, not how large it is at full scale.

## 3. The noise check agrees, and it now works properly

"Noise robustness" asks: scramble a few DNA bases, does the model's reading of the read survive?
All numbers below are **centred** — a correction we added this week, without which the metric was
pinned near 1.0 and could not distinguish anything.

| | floor | 0.5% error | 1% error | 5% error |
|---|---|---|---|---|
| SMALL-CAP | −0.003 | 0.9698 | 0.9401 | 0.7431 |
| LARGE-CAP | +0.006 | **0.9812** | **0.9616** | **0.8294** |

"Floor" is the score two unrelated reads get — effectively zero, as it should be. The data-rich
model is more robust at every noise level. Two independent evaluations point the same way.

## 4. Status of the recommended Aurora run ("H")

**Not submitted. I could not submit it.** This machine is a Mac: it has no PBS client
(`qsub`, `qstat`, `pbsnodes` are all absent), no configured ALCF access, and `/flare` is not
mounted. Submission needs a person on an Aurora login node.

I also could not check free memory on a compute node, for the same reason. Because "unverified"
is not "confirmed", the script now ships with the **safe fallback active**: `--reads-cap 20000`
(~56 GB/node) at 26,250 steps — still exactly 4.0 passes.

Ready to run, from an Aurora login node in the repo directory:

```
qsub launch/next_run_2026-10-22.sh
```

To restore the fuller run, first confirm >= 160 GB free on a compute node (`qsub -I ... ; free -g`),
then change `--reads-cap 20000` -> `50000` **and** `--max-steps 26250` -> `65625` together.

`train_mlm.py` now prints tokens-seen and passes at every evaluation and writes them to the
results file, with a `*** PAST 4-PASS BUDGET ***` warning — so this cannot happen silently again.

## 5. Three decisions needed

1. **Who submits H, and at which size?** Someone with ALCF access must run the `qsub`. Default is
   the safe 20,000-cap version; the 50,000 version needs a one-minute memory check first.
2. **Do we re-baseline the published numbers?** Val top-1 0.248 and the whole "capacity x data"
   scaling ladder were measured on 2.5% of the corpus, and the headline run at ~305 passes. Do we
   re-run the ladder properly, or publish with the caveat attached?
3. **Do we make passes a hard gate?** Proposal: no run launches if its projected passes exceed 4.
   The number is now logged automatically; this makes it a rule rather than a habit.
