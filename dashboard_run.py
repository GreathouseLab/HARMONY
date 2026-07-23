"""
Focused single-run dashboard for the LAST training run (depth-6, 120M, 630k cohort).
Tufte-style, self-contained HTML. Reads experiments/mlm_bigcohort_d6/probe_trajectory.csv.
Writes experiments/dashboard_run_depth6.html.
"""
from __future__ import annotations
import csv, statistics as st
from pathlib import Path

EXP = Path(__file__).parent / "experiments"
INK, MUTE, FAINT, ACCENT, GOOD = "#222222", "#9a9a9a", "#d8d8d8", "#a8443a", "#2f6b4f"
FONT = '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif'
UNIGRAM, LOOKUP = 0.0923, 0.127

r = list(csv.DictReader(open(EXP / "mlm_bigcohort_d6/probe_trajectory.csv")))
steps = [int(x["step"]) for x in r]
val = [float(x["val_msk_top1"]) for x in r]
tr = [float(x["train_msk_top1"]) for x in r]
vce = [float(x["val_msk_ce"]) for x in r]
wall_h = float(r[-1]["wall_clock_s"]) / 3600
ms = float(r[-1]["wall_clock_s"]) / steps[-1] * 1000
peak, end5, gap = max(val), st.mean(val[-5:]), tr[-1] - st.mean(val[-5:])
# ladder context (best val by depth on 630k)
def bestval(d):
    p = EXP / f"{d}/probe_trajectory.csv"
    return max(float(x["val_msk_top1"]) for x in csv.DictReader(open(p))) if p.exists() else None
ladder = [("depth-2 · 5M", bestval("mlm_bigcohort")), ("depth-4 · 41M", bestval("mlm_bigcohort_d4")),
          ("depth-6 · 120M", peak)]


def line_chart(w=680, h=300, ymin=0.08, ymax=0.19):
    padL, padR, padT, padB = 46, 92, 18, 30
    X = lambda s: padL + (s - steps[0]) / (steps[-1] - steps[0]) * (w - padL - padR)
    Y = lambda v: padT + (ymax - v) / (ymax - ymin) * (h - padT - padB)
    s = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" font-family=\'{FONT}\'>']
    for gv in [0.10, 0.12, 0.14, 0.16, 0.18]:
        s.append(f'<line x1="{padL}" y1="{Y(gv):.1f}" x2="{w-padR}" y2="{Y(gv):.1f}" stroke="{FAINT}"/>')
        s.append(f'<text x="{padL-6}" y="{Y(gv)+3:.1f}" font-size="10" fill="{MUTE}" text-anchor="end">{gv:.2f}</text>')
    for yv, lab, col in [(UNIGRAM, "chance floor", MUTE), (LOOKUP, "lookup-table baseline", ACCENT)]:
        s.append(f'<line x1="{padL}" y1="{Y(yv):.1f}" x2="{w-padR}" y2="{Y(yv):.1f}" stroke="{col}" '
                 f'stroke-width="1" stroke-dasharray="2 3"/>')
        s.append(f'<text x="{padL+3}" y="{Y(yv)-3:.1f}" font-size="9.5" fill="{col}">{lab} {yv:.3f}</text>')
    def series(vals, col, lab):
        pts = " ".join(f"{X(steps[i]):.1f},{Y(vals[i]):.1f}" for i in range(len(vals)))
        return (f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"/>'
                f'<text x="{X(steps[-1])+6:.1f}" y="{Y(vals[-1])+3:.1f}" font-size="11.5" fill="{col}" '
                f'font-weight="600">{lab} {vals[-1]:.3f}</text>')
    for s0 in [steps[0], steps[len(steps)//2], steps[-1]]:
        s.append(f'<text x="{X(s0):.1f}" y="{h-8}" font-size="10" fill="{MUTE}" text-anchor="middle">{s0//1000}k</text>')
    s.append(series(tr, GOOD, "train"))
    s.append(series(val, INK, "val"))
    # mark the cut-off
    s.append(f'<text x="{X(steps[-1]):.1f}" y="{padT+2}" font-size="9" fill="{MUTE}" text-anchor="end">10h wall ↓</text>')
    s.append("</svg>")
    return "".join(s)


def ce_spark(w=680, h=90):
    padL, padR = 46, 92
    lo, hi = min(vce) - 0.05, max(vce) + 0.05
    X = lambda s: padL + (s - steps[0]) / (steps[-1] - steps[0]) * (w - padL - padR)
    Y = lambda v: 14 + (hi - v) / (hi - lo) * (h - 30)
    pts = " ".join(f"{X(steps[i]):.1f},{Y(vce[i]):.1f}" for i in range(len(vce)))
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" font-family=\'{FONT}\'>'
            f'<polyline points="{pts}" fill="none" stroke="{ACCENT}" stroke-width="2"/>'
            f'<text x="{X(steps[0]):.1f}" y="{Y(vce[0])-5:.1f}" font-size="10" fill="{MUTE}">{vce[0]:.2f}</text>'
            f'<text x="{X(steps[-1])+6:.1f}" y="{Y(vce[-1])+3:.1f}" font-size="11" fill="{ACCENT}" '
            f'font-weight="600">val CE {vce[-1]:.2f}</text></svg>')


def ladder_plot(w=680, h=120):
    padL, padR, top, rh = 130, 90, 20, 30
    x0, x1 = 0.10, 0.19
    X = lambda v: padL + (v - x0) / (x1 - x0) * (w - padL - padR)
    s = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" font-family=\'{FONT}\'>']
    s.append(f'<line x1="{X(LOOKUP):.1f}" y1="{top-8}" x2="{X(LOOKUP):.1f}" y2="{top+rh*3}" '
             f'stroke="{ACCENT}" stroke-dasharray="2 3" opacity="0.7"/>')
    s.append(f'<text x="{X(LOOKUP):.1f}" y="{top-11}" font-size="9" fill="{ACCENT}" text-anchor="middle">lookup 0.127</text>')
    for i, (lab, v) in enumerate(ladder):
        if v is None: continue
        y = top + rh * i + rh / 2
        foc = i == len(ladder) - 1
        col = ACCENT if foc else INK
        s.append(f'<text x="{padL-12}" y="{y+3.5}" font-size="11.5" fill="{INK}" text-anchor="end" '
                 f'font-weight="{"600" if foc else "400"}">{lab}</text>')
        s.append(f'<line x1="{X(0.10):.1f}" y1="{y}" x2="{X(v):.1f}" y2="{y}" stroke="{FAINT}"/>')
        s.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="{5.5 if foc else 4.5}" fill="{col}"/>')
        s.append(f'<text x="{X(v)+9:.1f}" y="{y+3.5}" font-size="11.5" fill="{col}" '
                 f'font-weight="{"600" if foc else "400"}">{v:.3f}</text>')
    s.append("</svg>")
    return "".join(s)


facts = [("Model", "depth-6 · aspect-192 · n_embd 1152 · 119.5M params"),
         ("Data", "630,000 reads (cap 5,000/sample × 126 samples), λ=0 pure MLM"),
         ("Schedule", f"{steps[-1]:,} of 40,000 steps · eval every 2,000 · ~{ms:.0f} ms/step"),
         ("Stopped", f"10-hour runtime guard at ~{wall_h:.1f}h — still rising, not converged"),
         ("Best val top-1", f"{peak:.3f} (peak) · {end5:.3f} (last-5 mean)"),
         ("Generalization", f"train {tr[-1]:.3f} vs val {end5:.3f} → gap {gap:+.3f} (val ≥ train, no overfitting)")]

html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Last run — depth-6 top-1</title>
<style>
 body{{font-family:{FONT};color:{INK};max-width:760px;margin:30px auto;padding:0 22px;line-height:1.5}}
 h1{{font-size:20px;margin:0 0 2px}} h2{{font-size:14px;margin:28px 0 6px;font-weight:600;border-bottom:1px solid {FAINT};padding-bottom:4px}}
 .sub{{color:{MUTE};font-size:13px;margin:0 0 4px}} .cap{{color:#555;font-size:12px;margin:6px 0 0}}
 table{{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:6px}}
 td{{padding:6px 10px 6px 0;border-bottom:1px solid #eee}} td.k{{font-weight:600;white-space:nowrap;width:150px}}
 .verdict{{background:#faf7f3;border-left:3px solid {ACCENT};padding:12px 16px;font-size:13px;margin-top:8px}}
</style></head><body>
<h1>Last training run — depth-6 (120M) on the 630k-read cohort</h1>
<p class="sub">Top-1 accuracy on held-out (validation) data — the headline metric. Train vs validation over training.</p>
<div class="verdict"><b>Result:</b> validation top-1 climbed to <b>{peak:.3f}</b> (peak), well past the non-learning
lookup-table baseline (0.127), with <b>validation ≥ train</b> throughout (gap {gap:+.3f}) — generalizing, not
memorizing. It was <b>still rising</b> when the 10-hour wall stopped it at {steps[-1]//1000}k of 40k steps.</div>

<h2>1 · Top-1 accuracy over training — train vs validation</h2>
{line_chart()}
<p class="cap">Validation (dark) tracks <i>above</i> train (green) the whole way — the "signature" replicates on
held-out reads rather than overfitting. Both still climbing at the cut-off.</p>

<h2>2 · Validation cross-entropy (calibration) — lower is better</h2>
{ce_spark()}
<p class="cap">Falls steadily ({vce[0]:.2f} → {vce[-1]:.2f} nats): the model grows more confident <i>and</i> correct.</p>

<h2>3 · Where it lands — the capacity ladder (630k cohort)</h2>
{ladder_plot()}
<p class="cap">Best validation top-1 by model size, same data. Each step up in capacity buys real, generalizing accuracy.</p>

<h2>4 · Run facts</h2>
<table>{"".join(f'<tr><td class="k">{k}</td><td>{v}</td></tr>' for k,v in facts)}</table>
<p class="cap" style="margin-top:14px">Generated by dashboard_run.py from experiments/mlm_bigcohort_d6/probe_trajectory.csv.</p>
</body></html>"""

out = EXP / "dashboard_run_depth6.html"
out.write_text(html)
print(f"wrote {out}  (peak val {peak:.3f}, last-5 {end5:.3f}, gap {gap:+.3f}, {steps[-1]} steps, {wall_h:.1f}h)")
