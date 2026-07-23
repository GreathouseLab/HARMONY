"""
Decode and visualize MLM predictions over the val set at the nucleotide level.

For every val read, applies the same 80/10/10 masking the trainer uses, runs the
FINAL MVT checkpoint, then writes an HTML page showing:

  - the full nucleotide read tokenized into BPE pieces (one cell per token)
  - two parallel tracks: TRUE nucleotides vs PRED nucleotides
  - masked positions colored by correctness (green=top-1 match, red=miss)
  - per-mask detail rows with top-5 predicted nucleotide pieces + probabilities
  - aggregate stats per branch at the top

Output: experiments/mlm_mvt/decoded_predictions.html  (self-contained, no JS deps)
"""
from __future__ import annotations

import html as html_mod
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train_mlm import GPT_MLM, build_model, mlm_mask, EXTENDED_VOCAB, MASK_ID
from prepare_genomic import Tokenizer
from probe_sample_coherence import parse_val_txt_grouped


# ============================================================ config
# CHECKPOINT and OUT_HTML can be overridden via env vars VIZ_CHECKPOINT / VIZ_OUT_HTML
# so the same visualizer can target either run (mlm_mvt, mlm_lam0, ...).
import os as _os
CHECKPOINT = _os.environ.get("VIZ_CHECKPOINT", "experiments/mlm_mvt/checkpoint.pt")
OUT_HTML = Path(_os.environ.get("VIZ_OUT_HTML", "experiments/mlm_mvt/decoded_predictions.html"))
N_READS_PER_SAMPLE = 50
SEED = 7
PAD = 4090
T_MAX = 64

from device_utils import get_device
DEVICE = get_device()


# ============================================================ helpers
def safe_nt(s: str) -> str:
    """Render whitespace / specials visibly."""
    if s == "":
        return "·"
    if s.isspace():
        return "·" * len(s)
    return s


def decode_token(tk: Tokenizer, tid: int) -> str:
    try:
        return tk.enc.decode([int(tid)])
    except Exception:
        return f"<id={int(tid)}>"


def is_special_id(tid: int) -> bool:
    return 4090 <= tid <= 4096


# ============================================================ main
def main():
    tk = Tokenizer.from_directory()
    _depth = int(_os.environ.get("VIZ_DEPTH", "2"))        # override for larger checkpoints
    _aspect = int(_os.environ.get("VIZ_ASPECT", "128"))
    model, config = build_model(DEVICE, depth=_depth, aspect_ratio=_aspect, seq_len=64)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[viz] checkpoint: step={ckpt.get('step')}  last_losses={ckpt.get('last_losses')}")

    samples = parse_val_txt_grouped("output/val.txt", n_per_sample=N_READS_PER_SAMPLE, seed=42)
    flat = []
    for s_idx in sorted(samples.keys()):
        for body in samples[s_idx]:
            flat.append((s_idx, body))
    print(f"[viz] {len(flat)} val reads ({len(samples)} samples × {N_READS_PER_SAMPLE})")

    torch.manual_seed(SEED)
    t0 = time.time()

    read_records = []
    agg = {b: {"n": 0, "top1": 0, "top5": 0} for b in ("MSK", "RAND", "KEEP")}

    for read_idx, (s_idx, body) in enumerate(flat, 1):
        text = f"<READ_START> {body} <READ_END>"
        ids = tk.encode(text)
        L = len(ids)
        padded = ids + [PAD] * (T_MAX - L) if L < T_MAX else ids[:T_MAX]
        tok = torch.tensor([padded], dtype=torch.long, device=DEVICE)
        attn_lens = torch.tensor([L], dtype=torch.long, device=DEVICE)

        masked, targets = mlm_mask(tok, vocab_size=EXTENDED_VOCAB, mask_prob=0.15)
        with torch.no_grad():
            logits, _ = model.forward_mlm(masked, attn_lens=attn_lens)
        probs = F.softmax(logits[0], dim=-1)

        # Per-token data for the visual
        tokens = []
        masks = []  # detail rows per masked pos
        for p in range(L):
            true_id = int(tok[0, p].item())
            in_id = int(masked[0, p].item())
            true_text = decode_token(tk, true_id)
            is_masked_pos = (int(targets[0, p].item()) != -1)
            entry = {
                "pos": p, "true_id": true_id, "true_text": true_text,
                "is_special": is_special_id(true_id),
                "is_masked": is_masked_pos,
            }
            if is_masked_pos:
                top5_p, top5_id_t = probs[p].topk(5)
                top5_ids = [int(x) for x in top5_id_t.tolist()]
                top5_probs = [float(x) for x in top5_p.tolist()]
                top5_texts = [decode_token(tk, t) for t in top5_ids]
                pred_id = top5_ids[0]
                pred_text = top5_texts[0]
                top1_correct = (pred_id == true_id)
                top5_correct = (true_id in top5_ids)
                if in_id == MASK_ID:        branch = "MSK"
                elif in_id == true_id:       branch = "KEEP"
                else:                        branch = "RAND"

                agg[branch]["n"] += 1
                agg[branch]["top1"] += int(top1_correct)
                agg[branch]["top5"] += int(top5_correct)

                entry.update({
                    "branch": branch,
                    "in_id": in_id,
                    "in_text": decode_token(tk, in_id),
                    "pred_id": pred_id, "pred_text": pred_text, "pred_prob": top5_probs[0],
                    "top1_correct": top1_correct, "top5_correct": top5_correct,
                    "top5_ids": top5_ids, "top5_texts": top5_texts, "top5_probs": top5_probs,
                })
                masks.append(entry)
            else:
                entry.update({
                    "branch": None,
                    "pred_text": true_text,   # passthrough
                    "top1_correct": True,
                    "pred_prob": None,
                })
            tokens.append(entry)

        read_records.append({
            "read_idx": read_idx, "sample_idx": s_idx, "n_tokens": L,
            "body": body, "tokens": tokens, "masks": masks,
        })

    print(f"[viz] inference done in {time.time()-t0:.1f}s")

    # Aggregate top-line
    total_n = sum(a["n"] for a in agg.values())
    total_top1 = sum(a["top1"] for a in agg.values())
    total_top5 = sum(a["top5"] for a in agg.values())

    # ====================================================== render HTML
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_HTML, "w") as f:
        f.write(_render_html(read_records, agg, total_n, total_top1, total_top5, ckpt))
    print(f"[viz] wrote {OUT_HTML} ({OUT_HTML.stat().st_size/1024:.1f} KB)")
    print(f"[viz] aggregate top-1: {100*total_top1/max(total_n,1):.1f}%  ({total_top1}/{total_n})")
    for b in ("MSK", "RAND", "KEEP"):
        a = agg[b]
        if a["n"]:
            print(f"        {b:5s}: top-1 {100*a['top1']/a['n']:.1f}%  top-5 {100*a['top5']/a['n']:.1f}%  (n={a['n']})")


def _render_html(reads, agg, total_n, total_top1, total_top5, ckpt) -> str:
    rows = []
    for rec in reads:
        rows.append(_render_read(rec))
    rows_html = "\n".join(rows)

    nt_count = sum(len(r["body"].replace(" ", "").replace("<PAIRED_END>", "")) for r in reads)
    samples_present = sorted(set(r["sample_idx"] for r in reads))
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>MLM decoded predictions — HARMONY val</title>
<style>
:root {{
  --bg: #0f1115; --fg: #e6e6e6; --muted: #8a8f99; --hl: #25282f;
  --keep: #2c3e50; --keep-fg: #b8c5d6;
  --special: #3a3a44; --special-fg: #d0d0d8;
  --correct: #16774a; --correct-fg: #d7f5e1;
  --wrong: #8a2a2a; --wrong-fg: #f5d7d7;
  --top5hit: #6b5a16; --top5hit-fg: #f5ecc6;
  --msk-edge: #ffd24a; --rand-edge: #c084fc; --keep-edge: #6b7280;
}}
body {{ background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 16px 24px; margin: 0; }}
h1, h2, h3 {{ font-weight: 500; margin: 0.4em 0; }}
h1 {{ font-size: 1.4em; }}
header.summary {{ background: #14171c; padding: 12px 16px; border-radius: 8px; margin-bottom: 18px; }}
.stats {{ display: flex; gap: 24px; flex-wrap: wrap; margin-top: 6px; font-size: 0.92em; }}
.stat {{ background: #1a1d23; padding: 6px 10px; border-radius: 5px; }}
.stat b {{ color: #ffd24a; }}
.legend {{ font-size: 0.85em; color: var(--muted); margin-top: 8px; }}
.legend .swatch {{ display: inline-block; width: 14px; height: 14px; vertical-align: middle; margin: 0 4px 0 12px; border-radius: 3px; border: 1px solid #444; }}
.read {{ background: #14171c; border-radius: 8px; padding: 12px 14px; margin-bottom: 16px; border-left: 3px solid #2a2d36; }}
.read .hdr {{ font-size: 0.92em; color: var(--muted); margin-bottom: 6px; }}
.read .hdr b {{ color: var(--fg); }}
.track-label {{ display: inline-block; width: 60px; color: var(--muted); font-size: 0.8em; font-family: ui-monospace, monospace; }}
.track {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 0.85em; white-space: nowrap; overflow-x: auto; padding: 4px 0; }}
.cell {{ display: inline-block; padding: 3px 4px; margin: 0 1px; border-radius: 3px; min-width: 8px; text-align: center; font-variant-numeric: tabular-nums; }}
.cell.unmasked {{ color: var(--fg); background: transparent; }}
.cell.special  {{ color: var(--special-fg); background: var(--special); font-size: 0.78em; }}
.cell.keep     {{ background: var(--keep); color: var(--keep-fg); border: 1px dotted var(--keep-edge); }}
.cell.correct  {{ background: var(--correct); color: var(--correct-fg); border: 1px solid var(--msk-edge); }}
.cell.correct.rand {{ border-style: dashed; border-color: var(--rand-edge); }}
.cell.correct.keep-branch {{ border: 1px dotted var(--keep-edge); }}
.cell.wrong    {{ background: var(--wrong); color: var(--wrong-fg); border: 1px solid var(--msk-edge); }}
.cell.wrong.rand {{ border-style: dashed; border-color: var(--rand-edge); }}
.cell.wrong.top5 {{ background: var(--top5hit); color: var(--top5hit-fg); }}
.mask-detail {{ font-family: ui-monospace, monospace; font-size: 0.82em; color: var(--muted); margin-top: 6px; padding-top: 6px; border-top: 1px dashed #2a2d36; }}
.mask-detail table {{ border-collapse: collapse; width: 100%; }}
.mask-detail td, .mask-detail th {{ padding: 2px 8px; text-align: left; vertical-align: top; }}
.mask-detail th {{ color: var(--muted); font-weight: normal; font-size: 0.85em; }}
.mask-detail tr.row-MSK  td:first-child {{ border-left: 3px solid var(--msk-edge); padding-left: 6px; }}
.mask-detail tr.row-RAND td:first-child {{ border-left: 3px solid var(--rand-edge); padding-left: 6px; }}
.mask-detail tr.row-KEEP td:first-child {{ border-left: 3px solid var(--keep-edge); padding-left: 6px; }}
.mask-detail .true  {{ color: var(--correct-fg); }}
.mask-detail .pred-ok  {{ color: var(--correct-fg); }}
.mask-detail .pred-bad {{ color: var(--wrong-fg); }}
.top5 {{ font-size: 0.85em; color: var(--muted); }}
.top5 .hit {{ color: var(--correct-fg); font-weight: bold; }}
nav.toc {{ position: sticky; top: 0; background: var(--bg); padding: 6px 0; border-bottom: 1px solid #2a2d36; margin-bottom: 12px; font-size: 0.88em; }}
nav.toc a {{ color: var(--muted); text-decoration: none; margin-right: 12px; }}
nav.toc a:hover {{ color: #ffd24a; }}
</style>
</head><body>
<header class="summary">
  <h1>MLM decoded predictions — HARMONY val set</h1>
  <div style="color: var(--muted); font-size:0.9em;">
    checkpoint: <b style="color:#fff">{html_mod.escape(str(CHECKPOINT))}</b> (step {ckpt.get("step")}, last_losses MLM={ckpt.get("last_losses",[None,None])[0]:.4f} contrastive={ckpt.get("last_losses",[None,None])[1]:.4f}).
    {len(reads)} val reads across {len(samples_present)} samples, {nt_count} nucleotides total. Mask rate 15% (BERT 80/10/10).
  </div>
  <div class="stats">
    <div class="stat">overall top-1: <b>{100*total_top1/max(total_n,1):.1f}%</b> ({total_top1}/{total_n})</div>
    <div class="stat">overall top-5: <b>{100*total_top5/max(total_n,1):.1f}%</b></div>
    <div class="stat">MSK top-1: <b>{100*agg["MSK"]["top1"]/max(agg["MSK"]["n"],1):.1f}%</b> (n={agg["MSK"]["n"]})</div>
    <div class="stat">RAND top-1: <b>{100*agg["RAND"]["top1"]/max(agg["RAND"]["n"],1):.1f}%</b> (n={agg["RAND"]["n"]})</div>
    <div class="stat">KEEP top-1: <b>{100*agg["KEEP"]["top1"]/max(agg["KEEP"]["n"],1):.1f}%</b> (n={agg["KEEP"]["n"]})</div>
  </div>
  <div class="legend">
    <span class="swatch" style="background:transparent;border:1px solid #444"></span> unmasked DNA token
    <span class="swatch" style="background:var(--special)"></span> special token (READ_START/PAIRED_END/etc.)
    <span class="swatch" style="background:var(--keep);border:1px dotted var(--keep-edge)"></span> KEEP branch (model second-guesses)
    <span class="swatch" style="background:var(--correct);border:1px solid var(--msk-edge)"></span> MSK correct (top-1)
    <span class="swatch" style="background:var(--top5hit);border:1px solid var(--msk-edge)"></span> MSK top-1 wrong but in top-5
    <span class="swatch" style="background:var(--wrong);border:1px solid var(--msk-edge)"></span> MSK wrong
    <span class="swatch" style="background:var(--correct);border:1px dashed var(--rand-edge)"></span> RAND branch (dashed border)
  </div>
</header>
<nav class="toc">
  {''.join(f'<a href="#s{s}">sample {s}</a>' for s in samples_present)}
</nav>
{rows_html}
</body></html>"""


def _render_read(rec) -> str:
    tokens = rec["tokens"]
    # TRUE track
    true_cells = []
    pred_cells = []
    for t in tokens:
        true_text = html_mod.escape(safe_nt(t["true_text"]))
        if t["is_special"]:
            cls = "cell special"
            true_cells.append(f'<span class="{cls}" title="pos {t["pos"]} special id={t["true_id"]}">{true_text}</span>')
            pred_cells.append(f'<span class="{cls}">{true_text}</span>')
        elif not t["is_masked"]:
            cls = "cell unmasked"
            true_cells.append(f'<span class="{cls}" title="pos {t["pos"]} id={t["true_id"]}">{true_text}</span>')
            pred_cells.append(f'<span class="{cls}">{true_text}</span>')
        else:
            branch = t["branch"]
            top1_ok = t["top1_correct"]
            top5_ok = t["top5_correct"]
            pred_text = html_mod.escape(safe_nt(t["pred_text"]))
            base_cls = "cell"
            extras = []
            if top1_ok:
                extras.append("correct")
            else:
                extras.append("wrong")
                if top5_ok:
                    extras.append("top5")
            if branch == "RAND":
                extras.append("rand")
            elif branch == "KEEP":
                extras.append("keep-branch")
            cls = base_cls + " " + " ".join(extras)
            top5_tt = " · ".join(
                f"{html_mod.escape(safe_nt(t['top5_texts'][i]))}({t['top5_probs'][i]:.3f})"
                for i in range(5)
            )
            tip = (f"pos {t['pos']} | branch {branch} | true={t['true_text']!r}(id {t['true_id']}) "
                   f"pred={t['pred_text']!r}(p={t['pred_prob']:.3f}) | top5: {top5_tt}")
            tip = html_mod.escape(tip)
            # TRUE row shows the gold token (same in any branch)
            true_cells.append(f'<span class="{cls}" title="{tip}">{true_text}</span>')
            pred_cells.append(f'<span class="{cls}" title="{tip}">{pred_text}</span>')

    # Mask-detail table
    detail_rows = []
    for m in rec["masks"]:
        true_text = html_mod.escape(safe_nt(m["true_text"]))
        pred_text = html_mod.escape(safe_nt(m["pred_text"]))
        ok = m["top1_correct"]
        pred_cls = "pred-ok" if ok else "pred-bad"
        top5_html = " ".join(
            f'<span class="{"hit" if m["top5_ids"][i]==m["true_id"] else ""}">{html_mod.escape(safe_nt(m["top5_texts"][i]))}({m["top5_probs"][i]:.3f})</span>'
            for i in range(5)
        )
        detail_rows.append(
            f'<tr class="row-{m["branch"]}">'
            f'<td>pos {m["pos"]:>2} <span style="color:#aaa">[{m["branch"]}]</span></td>'
            f'<td class="true">true: <b>{true_text}</b></td>'
            f'<td class="{pred_cls}">pred: <b>{pred_text}</b> (p={m["pred_prob"]:.3f})</td>'
            f'<td class="top5">top-5: {top5_html}</td>'
            f'</tr>'
        )

    sample_anchor = f's{rec["sample_idx"]}'
    body_short = rec["body"].replace(" ", "").replace("<PAIRED_END>", "·")
    body_short = (body_short[:160] + "…") if len(body_short) > 160 else body_short

    return f"""
<div class="read" id="{sample_anchor}_r{rec['read_idx']}">
  <div class="hdr">
    <b>read #{rec['read_idx']}</b> · sample <b id="{sample_anchor}">{rec['sample_idx']}</b> · {rec['n_tokens']} tokens · {len(rec['masks'])} masked positions
    <span style="color:#666; margin-left:12px; font-family:ui-monospace,monospace;">nt: {html_mod.escape(body_short)}</span>
  </div>
  <div class="track"><span class="track-label">TRUE&nbsp;:</span>{''.join(true_cells)}</div>
  <div class="track"><span class="track-label">PRED&nbsp;:</span>{''.join(pred_cells)}</div>
  <div class="mask-detail"><table>{''.join(detail_rows)}</table></div>
</div>
"""


if __name__ == "__main__":
    main()
