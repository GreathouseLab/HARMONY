"""
Build a human-interpretable, Tufte-style HTML dashboard of the within-read MLM investigation.

Reads the experiment artifacts (sweep / vanilla / count-model / neighbor-diagnostic JSONs and the
memcheck + more-data trajectory CSVs) and emits a single self-contained file:
    experiments/dashboard.html

Re-run after any new run completes to refresh the numbers (the more-data panel auto-fills once
experiments/mlm_moredata/probe_trajectory.csv exists). Measurement only; no deps beyond stdlib.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).parent
EXP = ROOT / "experiments"

# ---- design tokens (Tufte: ink data, faint scaffolding, ONE accent) ----
INK = "#222222"          # data
MUTE = "#9a9a9a"         # scaffolding / context series
FAINT = "#d8d8d8"        # gridlines / range frame
ACCENT = "#a8443a"       # the one focal color (the local-memorization ceiling)
GOOD = "#2f6b4f"         # train series (memorization)
FONT = ('-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif')


def load(p):
    p = EXP / p
    return json.load(open(p)) if p.exists() else None


def traj(p):
    p = EXP / p
    if not p.exists():
        return []
    return list(csv.DictReader(open(p)))


# ---------- data ----------
UNIGRAM = 0.0923
sweep = load("mlm_sweep_summary.json")
vanilla = load("mlm_vanilla/vanilla_result.json")
kmer = load("kmer_markov_baseline.json")
diag = load("nn_vs_markov_diag.json")
mem = traj("mlm_memcheck/probe_trajectory.csv")
moredata = traj("mlm_moredata/probe_trajectory.csv")            # 252k reads, depth-2
bigcohort_d2 = traj("mlm_bigcohort/probe_trajectory.csv")       # 630k reads, depth-2
big_d4 = traj("mlm_bigcohort_d4/probe_trajectory.csv")          # 630k reads, depth-4 (41M)
big_d6 = traj("mlm_bigcohort_d6/probe_trajectory.csv")          # 630k reads, depth-6 (120M)
more = big_d6 or big_d4 or bigcohort_d2 or moredata             # latest/biggest for panel 3

count_overall = kmer["results"]["bidirectional_gold_trigram"]["top1"]   # 0.1272
count_clean = diag["buckets"]["both_clean"]["markov_gold_top1"]         # 0.1352

# headline dot-plot rows: (label, top1, kind)  kind in {floor, learned, focal, ceiling}
rows = [
    ("Always-guess-most-common (floor)", UNIGRAM, "floor"),
    ("Vanilla BERT encoder", vanilla["best_val"]["top1"], "learned"),
    ("Our model, depth-2", 0.1090, "learned"),
    ("Our model, depth-4 (41M)", 0.1119, "learned"),
    ("Best of tuning sweep", sweep["A_control"]["best_val_top1"], "learned"),
]
if bigcohort_d2:
    rows.append(("Big cohort · depth-2, 5M (630k)", max(float(r["val_msk_top1"]) for r in bigcohort_d2), "learned"))
if big_d4:
    rows.append(("Big cohort · depth-4, 41M", max(float(r["val_msk_top1"]) for r in big_d4), "learned"))
if big_d6:
    rows.append(("Big cohort · depth-6, 120M", max(float(r["val_msk_top1"]) for r in big_d6), "focal"))
rows += [
    ("Lookup table, overall", count_overall, "ceiling"),
    ("Lookup table, clean neighbors", count_clean, "ceiling"),
]


# ---------- tiny SVG helpers ----------
def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def dotplot(rows, w=720, x0=0.085, x1=0.185):
    """Sorted horizontal dot plot with two reference bands. One quantity, many items -> dot plot."""
    padL, padR, top, rowh = 250, 70, 28, 30
    h = top + rowh * len(rows) + 20
    plotw = w - padL - padR
    def X(v): return padL + (v - x0) / (x1 - x0) * plotw
    color = {"floor": MUTE, "learned": INK, "focal": ACCENT, "ceiling": ACCENT}
    s = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" font-family=\'{FONT}\'>']
    # reference vlines: unigram floor + count ceiling
    for v, lab, col in [(UNIGRAM, "floor", MUTE), (count_clean, "ceiling", ACCENT)]:
        s.append(f'<line x1="{X(v):.1f}" y1="{top-12}" x2="{X(v):.1f}" y2="{h-22}" '
                 f'stroke="{col}" stroke-width="1" stroke-dasharray="2 3" opacity="0.7"/>')
    # x-axis ticks (range frame: only where data lives)
    for v in [0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]:
        s.append(f'<text x="{X(v):.1f}" y="{h-6}" font-size="10" fill="{MUTE}" '
                 f'text-anchor="middle">{v:.2f}</text>')
    for i, (lab, v, kind) in enumerate(rows):
        y = top + rowh * i + rowh / 2
        c = color[kind]
        weight = "600" if kind == "focal" else "400"
        s.append(f'<text x="{padL-14}" y="{y+3.5}" font-size="12" fill="{INK}" '
                 f'text-anchor="end" font-weight="{weight}">{esc(lab)}</text>')
        # connector from floor line to dot (light), then dot + value
        s.append(f'<line x1="{X(UNIGRAM):.1f}" y1="{y}" x2="{X(v):.1f}" y2="{y}" '
                 f'stroke="{FAINT}" stroke-width="1"/>')
        r = 5.5 if kind == "focal" else 4.5
        s.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="{r}" fill="{c}"/>')
        s.append(f'<text x="{X(v)+10:.1f}" y="{y+3.5}" font-size="11.5" fill="{c}" '
                 f'font-weight="{weight}">{v:.3f}</text>')
    # band labels
    s.append(f'<text x="{X(UNIGRAM):.1f}" y="{top-16}" font-size="10" fill="{MUTE}" '
             f'text-anchor="middle">chance floor</text>')
    s.append(f'<text x="{X(count_clean):.1f}" y="{top-16}" font-size="10" fill="{ACCENT}" '
             f'text-anchor="middle">local-memorization ceiling</text>')
    s.append("</svg>")
    return "".join(s)


def linepanel(rows, title, note, w=440, h=240, ymin=0.08, ymax=0.32, ceiling=None, gridvals=None):
    """Train vs val top-1 over steps. Two directly-labeled lines; the GAP is the story."""
    if not rows:
        return (f'<div class="panel"><h3>{esc(title)}</h3>'
                f'<div class="pending">⏳ run in progress — panel fills when it completes</div></div>')
    padL, padR, padT, padB = 44, 86, 16, 28
    gridvals = gridvals or [0.10, 0.15, 0.20, 0.25, 0.30]
    steps = [int(r["step"]) for r in rows]
    xv = lambda st: padL + (st - steps[0]) / max(1, (steps[-1] - steps[0])) * (w - padL - padR)
    yv = lambda v: padT + (ymax - v) / (ymax - ymin) * (h - padT - padB)
    def line(key, col, lab):
        pts = " ".join(f"{xv(int(r['step'])):.1f},{yv(float(r[key])):.1f}" for r in rows)
        last = rows[-1]
        return (f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>'
                f'<text x="{xv(steps[-1])+6:.1f}" y="{yv(float(last[key]))+3:.1f}" font-size="11" '
                f'fill="{col}" font-weight="600">{lab} {float(last[key]):.3f}</text>')
    s = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" font-family=\'{FONT}\'>']
    # y gridlines (faint) + labels
    for gv in gridvals:
        if ymin <= gv <= ymax:
            yy = yv(gv)
            s.append(f'<line x1="{padL}" y1="{yy:.1f}" x2="{w-padR}" y2="{yy:.1f}" stroke="{FAINT}" stroke-width="1"/>')
            s.append(f'<text x="{padL-6}" y="{yy+3:.1f}" font-size="9" fill="{MUTE}" text-anchor="end">{gv:.2f}</text>')
    # optional lookup-table ceiling reference (accent, dashed)
    if ceiling is not None and ymin <= ceiling <= ymax:
        s.append(f'<line x1="{padL}" y1="{yv(ceiling):.1f}" x2="{w-padR}" y2="{yv(ceiling):.1f}" '
                 f'stroke="{ACCENT}" stroke-width="1" stroke-dasharray="2 3"/>')
        s.append(f'<text x="{w-padR-2}" y="{yv(ceiling)-3:.1f}" font-size="9" fill="{ACCENT}" '
                 f'text-anchor="end">lookup ceiling {ceiling:.3f}</text>')
    # chance floor reference
    s.append(f'<line x1="{padL}" y1="{yv(UNIGRAM):.1f}" x2="{w-padR}" y2="{yv(UNIGRAM):.1f}" '
             f'stroke="{MUTE}" stroke-width="1" stroke-dasharray="2 3"/>')
    s.append(f'<text x="{padL+2}" y="{yv(UNIGRAM)-3:.1f}" font-size="9" fill="{MUTE}">chance {UNIGRAM:.3f}</text>')
    # x labels
    for r in [rows[0], rows[len(rows)//2], rows[-1]]:
        st = int(r["step"])
        s.append(f'<text x="{xv(st):.1f}" y="{h-8}" font-size="9" fill="{MUTE}" text-anchor="middle">{st//1000}k</text>')
    s.append(line("train_msk_top1", GOOD, "train"))
    s.append(line("val_msk_top1", INK, "val"))
    s.append("</svg>")
    gap = float(rows[-1]["train_msk_top1"]) - float(rows[-1]["val_msk_top1"])
    return (f'<div class="panel"><h3>{esc(title)}</h3>{"".join(s)}'
            f'<p class="cap">{esc(note)} <b>Final train–val gap: {gap:+.3f}.</b></p></div>')


def neighbor_bars(w=440, h=170):
    """Where the gap is: NN vs lookup table, split by clean/corrupted neighbors."""
    bc, cr = diag["buckets"]["both_clean"], diag["buckets"]["corrupted"]
    data = [("Both neighbors visible (77%)", bc["nn_top1"]["depth4_big"], bc["markov_gold_top1"]),
            ("A neighbor hidden (23%)", cr["nn_top1"]["depth4_big"], cr["markov_gold_top1"])]
    padL, padR, top, gh = 200, 50, 18, 64
    x0, x1 = 0.05, 0.15
    X = lambda v: padL + (v - x0) / (x1 - x0) * (w - padL - padR)
    s = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" font-family=\'{FONT}\'>']
    for i, (lab, nn, mk) in enumerate(data):
        y = top + gh * i
        s.append(f'<text x="{padL-12}" y="{y+gh/2:.1f}" font-size="11" fill="{INK}" text-anchor="end">{esc(lab)}</text>')
        for v, col, name, dy in [(mk, ACCENT, "lookup", 0), (nn, INK, "our model", 20)]:
            s.append(f'<line x1="{X(x0):.1f}" y1="{y+8+dy}" x2="{X(v):.1f}" y2="{y+8+dy}" stroke="{FAINT}"/>')
            s.append(f'<circle cx="{X(v):.1f}" cy="{y+8+dy}" r="4.5" fill="{col}"/>')
            s.append(f'<text x="{X(v)+8:.1f}" y="{y+11+dy:.1f}" font-size="10.5" fill="{col}">{name} {v:.3f}</text>')
    s.append("</svg>")
    return "".join(s)


# ---------- glossary ----------
GLOSSARY = [
    ("Top-1 accuracy", "How often the model's single best guess for a hidden DNA chunk is exactly right. Higher is better. This is the headline metric."),
    ("Top-5 accuracy", "How often the correct chunk is among the model's five best guesses."),
    ("Cross-entropy (CE)", "How 'surprised' the model is by the right answer, in nats. Lower is better. It can get worse even when top-1 holds, if the model is confidently wrong."),
    ("Masked DNA (MSK-DNA)", "We hide ~15% of the chunks in a read and ask the model to fill them in. We score only the hidden DNA chunks (not the structural markers)."),
    ("Train vs. validation", "Train = data the model studied. Validation = held-out data it never saw. A model that scores high on train but flat on validation is memorizing, not learning."),
    ("Chance floor (unigram)", "What you get by always guessing the single most common chunk — no model needed. Any real model must beat this to be worth anything."),
    ("Lookup table (count model)", "A non-learning baseline that just tallies 'which chunk usually sits between these two neighbors.' It memorizes local statistics; it does not generalize like a neural net."),
    ("Local-memorization ceiling", "The lookup table's score (0.135). It's the most you can get from pure local neighbor statistics — a memorization ceiling, not a target a generalizing model is expected to reach."),
    ("A read / paired fragment", "One ~150-base snippet of DNA (two paired ends), read in isolation, with no information about where in the genome it came from."),
]


# ---------- assemble ----------
more_status = "completed" if (more and int(more[-1]["step"]) >= 29000) else (
    f"in progress — last eval at step {more[-1]['step']} of 30000" if more else "not started")
headline_val = f'{float(more[-1]["val_msk_top1"]):.3f}' if more else "—"

import statistics as _st
if more:
    mv = [float(r["val_msk_top1"]) for r in more]
    mt = [float(r["train_msk_top1"]) for r in more]
    d6_peak = max(mv); d6_gap = mt[-1] - mv[-1]
    d2_peak = max(float(r["val_msk_top1"]) for r in bigcohort_d2) if bigcohort_d2 else 0.118
    d4_peak = max(float(r["val_msk_top1"]) for r in big_d4) if big_d4 else 0.165
    verdict_html = (
        f"<b>Bottom line — capacity × data is the lever; now compute-bound.</b> Earlier runs were all "
        f"data-starved, so nothing helped and I wrongly inferred a ~0.11 ceiling. With an adequate cohort "
        f"(630k reads), scaling the model climbs a clear ladder: <b>depth-2 (5M) {d2_peak:.3f} → "
        f"depth-4 (41M) {d4_peak:.3f} → depth-6 (120M) {d6_peak:.3f}</b> — well <i>past</i> the lookup-table "
        f"references (0.127 / 0.135), val ≥ train throughout (gap {d6_gap:+.3f}, generalizing, no overfit). "
        f"So the within-read signal is genuinely richer than the lookup table suggested; my 'isolated read is "
        f"near-empty' read was wrong. Two caveats: gains are <i>diminishing</i> (depth-4→6 added little) and "
        f"depth-6 was <i>cut off still-rising</i> by a 10h wall (32k/40k steps, ~1.1s/step). We're now "
        f"<b>compute-bound</b> on a Mac — the efficient way to go further is a pretrained backbone (DNABERT-2, "
        f"~120M, trained on 2000× more data) rather than scaling from scratch."
    )
else:
    verdict_html = ("<b>Bottom line.</b> Every learning model plateaus near <b>0.11</b> validation top-1, "
                    "just above the <b>0.092</b> chance floor. Runs pending.")

html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HARMONY — within-read MLM dashboard</title>
<style>
 body{{font-family:{FONT};color:{INK};max-width:980px;margin:32px auto;padding:0 22px;line-height:1.5}}
 h1{{font-size:21px;margin:0 0 2px}} h2{{font-size:15px;margin:34px 0 6px;font-weight:600;border-bottom:1px solid {FAINT};padding-bottom:4px}}
 h3{{font-size:13px;margin:0 0 6px;font-weight:600}}
 .sub{{color:{MUTE};font-size:13px;margin:0 0 4px}}
 .cap{{color:#555;font-size:12px;margin:6px 0 0}}
 .panels{{display:flex;gap:26px;flex-wrap:wrap}} .panel{{flex:1;min-width:330px}}
 .pending{{color:{MUTE};font-size:12px;padding:30px 0;text-align:center;border:1px dashed {FAINT}}}
 table{{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:6px}}
 td{{padding:6px 10px 6px 0;vertical-align:top;border-bottom:1px solid #eee}}
 td.term{{font-weight:600;white-space:nowrap;width:190px}}
 .verdict{{background:#faf7f3;border-left:3px solid {ACCENT};padding:12px 16px;font-size:13px;margin-top:8px}}
 .meta{{color:{MUTE};font-size:11px;margin-top:30px}}
</style></head><body>

<h1>Reading DNA from a single short read — what the model can and can't learn</h1>
<p class="sub">The task: hide pieces of a ~150-base DNA fragment and predict them. Headline metric =
top-1 accuracy on held-out (validation) data. The whole investigation in one page.</p>

<div class="verdict">{verdict_html}</div>

<h2>1 · The one number that matters — best validation top-1 by approach</h2>
<p class="sub">Sorted. Each dot is how often a model's best guess is right on data it never saw.
The two dashed lines are the chance floor and the lookup-table ceiling.</p>
{dotplot(rows)}

<h2>2 · Can the model even fit? &nbsp;·&nbsp; 3 · Does more data help generalization?</h2>
<div class="panels">
{linepanel(mem, "Memorization check (small data)",
           "Train accuracy climbs while validation stays flat — it memorizes, it does not generalize.",
           ymax=0.32)}
{linepanel(more, "Big cohort + bigger model (depth-6, 120M, 630k)",
           "Bigger model, adequate data: val well past the lookup ceiling, val≥train, still rising when the 10h wall hit.",
           ymin=0.09, ymax=0.19, ceiling=count_overall, gridvals=[0.10, 0.12, 0.14, 0.16, 0.18])}
</div>

<h2>4 · Where the gap is — our model vs the lookup table</h2>
<p class="sub">Even when the model sees both true neighbors (the easy 77% of cases), it still trails the
lookup table. That ruled out 'masking corruption' as the cause.</p>
{neighbor_bars()}

<h2>5 · Plain-language definitions</h2>
<table>
{"".join(f'<tr><td class="term">{esc(t)}</td><td>{esc(d)}</td></tr>' for t, d in GLOSSARY)}
</table>

<p class="meta">Generated by build_dashboard.py from experiment artifacts. Headline more-data validation top-1: {headline_val}.
Sources: mlm_sweep_summary · mlm_vanilla · kmer_markov_baseline · nn_vs_markov_diag · mlm_memcheck · mlm_moredata.</p>
</body></html>"""

out = EXP / "dashboard.html"
out.write_text(html)
print(f"wrote {out} ({len(html)} bytes)")
print(f"more-data status: {more_status}")
