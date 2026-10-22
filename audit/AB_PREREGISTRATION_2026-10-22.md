# Pre-registration — matched-token A/B (written before any result was read)

Written at the timestamp above, when `experiments/ab_smallcap/probe_trajectory.csv` held
only its header row. No val number from either run had been seen.

## Design

Both runs: depth-2, aspect-ratio 128, seq_len 64, seed 42, Muon, `--mask-prob 0.15`,
`--mlm-softcap 15`, 12,000 steps, 32 reads/step. **Identical token budget by construction**
(same steps x same reads/step x same 48.27 tokens/read).

| | `--reads-cap` | distinct reads | passes at 12,000 steps |
|---|---|---|---|
| SMALL-CAP | 100 | 12,600 | 30.5 |
| LARGE-CAP | 1,000 | 126,000 | 3.05 |

### Why the caps are 100/1000 and not 5000/50000

Not a RAM limit — the Mac holds 6.3M reads (11.6 GB) and tokenizes them in 1.3 min.
The reason is that **at Mac step counts the specified caps produce no contrast**: 12,000
steps x 32 reads = 384,000 read-presentations, which is only 0.61 passes over the 630,000
reads of `--reads-cap 5000`. Neither arm would repeat anything, so the experiment could not
detect repetition damage. Reproducing the Aurora pass ratio (305 vs ~30) within the time box
requires scaling both caps down by 50x. The ratio between arms is preserved at 10x, and
LARGE-CAP sits under the 4-pass budget while SMALL-CAP sits far above it — the same
qualitative contrast as the real decision.

## Interpretation rule (fixed in advance)

At the matched final token count (18,535,680 useful tokens):

- **LARGE-CAP better** -> the 0.248 rise is partly memorisation; data, not compute, was the
  limiter. Supports running H.
- **No difference** -> the 630k slice is representative; the compute-bound reading stands.
  H becomes lower priority.
- **SMALL-CAP better** -> report it plainly and do not explain it away. Would mean repetition
  is not harming this model at this scale, and the data-wall argument needs rethinking.

"Better" is judged on val MSK-DNA top-1 at the matched token count, with val cross-entropy as
the tiebreak. A gap smaller than the run-to-run jitter visible in the last three evals of
either curve counts as "no difference".

## Scope limit, stated in advance

This is a depth-2 4.82M-parameter model at 1/500th of the Aurora token budget. It can show
whether repetition damage is **detectable** in this regime. It cannot establish the magnitude
of that damage at depth-6 and 9.27B tokens.
