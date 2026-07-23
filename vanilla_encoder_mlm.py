"""
Step-3 / Option-1: VANILLA bidirectional encoder control.

Question: is the neural MLM's failure to match the count-model ceiling (val MSK-DNA top-1 ~0.135)
a property of OUR repurposed stack (rotary + 15.0 softcap + Muon + CLM-tuned LRs), or is 0.135
effectively the ceiling for any model that must generalize?

This trains a textbook BERT-style encoder — torch.nn.TransformerEncoder, LEARNED positional
embeddings, plain AdamW, NO softcap, NO Muon, NO rotary — on the SAME data (PairedDataLoader),
SAME masking (mlm_mask, 15%), pure MLM (no contrastive), and scores val with the SAME
MSK-DNA top-1/top-5/CE protocol as the sweep (fixed mask_seed=7, fixed 0.15, exclude {4090..4096}).
Sized to ~depth-2 scale (~5M) so this is an ARCHITECTURE test, not a capacity test.

Read:
  vanilla val top-1 >= ~0.135  -> the deficit was OUR stack; switch the encoder architecture.
  vanilla val top-1 ~= 0.11    -> 0.135 is near the real generalization ceiling; stop chasing it.

Writes experiments/mlm_vanilla/vanilla_result.json + probe_trajectory.csv. Bounded; NaN/runtime guards.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train_mlm import mlm_mask, EXTENDED_VOCAB, MASK_ID, SPECIAL_IDS_NEVER_MASK, PAD_ID
from prepare_genomic import Tokenizer
from paired_data_loader import PairedDataLoader
from probe_sample_coherence import parse_val_txt_grouped

OUT_DIR = Path("experiments/mlm_vanilla")
EXCL_SET = set(SPECIAL_IDS_NEVER_MASK)
SEQ_LEN = 64
# architecture (~depth-2 scale, standard)
D_MODEL, N_HEAD, N_LAYERS, FFN, DROPOUT = 256, 4, 4, 1024, 0.1
# training
MAX_STEPS, EVAL_EVERY, LOG_EVERY = 10000, 2000, 1000
LR, WEIGHT_DECAY, WARMUP = 3e-4, 0.01, 500
MAX_RUNTIME_H = 2.0
SEED = 42
VAL_MASK_SEED = 7
MAX_LOSS_ABORT = 100.0


class VanillaEncoderMLM(nn.Module):
    def __init__(self, vocab=EXTENDED_VOCAB, d=D_MODEL, nhead=N_HEAD,
                 nlayers=N_LAYERS, ffn=FFN, seq_len=SEQ_LEN, dropout=DROPOUT):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq_len, d)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=ffn,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.seq_len = seq_len
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, attn_lens):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.wte(idx) + self.pos(pos))
        key_pad = pos >= attn_lens.to(idx.device).unsqueeze(1)   # True = pad -> ignored
        h = self.encoder(x, src_key_padding_mask=key_pad)
        return self.head(self.norm(h))


def eval_val_mlm(model, tk, device, n_per_sample=50, parse_seed=42,
                 mask_seed=VAL_MASK_SEED, mask_prob=0.15, t_max=SEQ_LEN):
    """Identical scoring protocol to train_mlm.eval_val_mlm (MSK-only, DNA-only, fixed mask)."""
    samples = parse_val_txt_grouped("output/val.txt", n_per_sample=n_per_sample, seed=parse_seed)
    flat = [(s, b) for s in sorted(samples.keys()) for b in samples[s]]
    gen = torch.Generator().manual_seed(mask_seed)
    n_pos = n_top1 = n_top5 = 0
    logp = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _, body in flat:
            ids = tk.encode(f"<READ_START> {body} <READ_END>")
            L = len(ids)
            padded = ids + [PAD_ID] * (t_max - L) if L < t_max else ids[:t_max]
            L = min(L, t_max)
            tok_cpu = torch.tensor([padded], dtype=torch.long)
            masked_cpu, targets_cpu = mlm_mask(tok_cpu, vocab_size=EXTENDED_VOCAB,
                                               mask_prob=mask_prob, generator=gen)
            inp = masked_cpu.to(device)
            attn = torch.tensor([L], dtype=torch.long, device=device)
            logits = model(inp, attn)
            probs = F.softmax(logits[0].float(), dim=-1)
            for p in range(L):
                if int(targets_cpu[0, p].item()) == -1:
                    continue
                if int(masked_cpu[0, p].item()) != MASK_ID:
                    continue
                true_id = int(tok_cpu[0, p].item())
                if true_id in EXCL_SET:
                    continue
                top5 = probs[p].topk(5).indices.tolist()
                n_pos += 1
                if top5[0] == true_id:
                    n_top1 += 1
                if true_id in top5:
                    n_top5 += 1
                logp += float(torch.log(probs[p, true_id].clamp(min=1e-12)).item())
    if was_training:
        model.train()
    return {"n_positions": n_pos, "top1": n_top1 / n_pos,
            "top5": n_top5 / n_pos, "ce_nats": -logp / n_pos}


def main():
    from device_utils import get_device
    device = get_device()
    torch.manual_seed(SEED)
    tk = Tokenizer.from_directory()
    loader = PairedDataLoader("output/train.txt", tk, samples_per_batch=4, reads_per_sample=8,
                              n_reads_cap_per_sample=200, seq_len=SEQ_LEN, seed=SEED)
    model = VanillaEncoderMLM().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[vanilla] standard nn.TransformerEncoder: d={D_MODEL} heads={N_HEAD} layers={N_LAYERS} "
          f"ffn={FFN} dropout={DROPOUT} | params={n_params/1e6:.2f}M (depth-2 ref=4.82M)")
    print(f"[vanilla] AdamW lr={LR} wd={WEIGHT_DECAY} warmup={WARMUP} | no softcap/Muon/rotary")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))

    def lr_at(step):
        return LR * min(1.0, (step + 1) / WARMUP)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    traj_path = OUT_DIR / "probe_trajectory.csv"
    with open(traj_path, "w") as f:
        f.write("step,wall_clock_s,L_MLM,val_msk_top1,val_msk_top5,val_msk_ce\n")

    t0 = time.time()
    traj = []
    best = {"top1": -1, "step": -1}
    model.train()
    for step in range(MAX_STEPS):
        if time.time() - t0 > MAX_RUNTIME_H * 3600:
            print(f"[vanilla] MAX_RUNTIME at step {step}"); break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        token_ids, attn_lens, _ = next(loader)
        token_ids, attn_lens = token_ids.to(device), attn_lens.to(device)
        masked, targets = mlm_mask(token_ids, vocab_size=EXTENDED_VOCAB, mask_prob=0.15)
        logits = model(masked, attn_lens)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        if not torch.isfinite(loss).all() or float(loss.item()) > MAX_LOSS_ABORT:
            print(f"[vanilla] ABORT step {step}: bad loss {loss.item()}"); sys.exit(2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % LOG_EVERY == 0:
            print(f"  step {step:05d} | MLM={loss.item():.4f} | lr={lr_at(step):.2e} | "
                  f"dt={(time.time()-t0)/(step+1)*1000:.0f}ms/step")
        if step > 0 and step % EVAL_EVERY == 0:
            vm = eval_val_mlm(model, tk, device)
            wall = time.time() - t0
            with open(traj_path, "a") as f:
                f.write(f"{step},{wall:.1f},{loss.item():.6f},"
                        f"{vm['top1']:.6f},{vm['top5']:.6f},{vm['ce_nats']:.6f}\n")
            traj.append({"step": step, "wall_s": wall, "L_MLM": loss.item(), **vm})
            print(f"[vanilla] step {step} | val_msk top1={vm['top1']:.4f} top5={vm['top5']:.4f} "
                  f"CE={vm['ce_nats']:.4f} (n={vm['n_positions']})")
            if vm["top1"] > best["top1"]:
                best = {"top1": vm["top1"], "top5": vm["top5"], "ce": vm["ce_nats"], "step": step}
                torch.save({"model_state_dict": model.state_dict(), "step": step, "val": vm},
                           OUT_DIR / "best_val.pt")

    summary = {
        "architecture": {"type": "nn.TransformerEncoder (vanilla BERT-style)",
                         "d_model": D_MODEL, "n_head": N_HEAD, "n_layers": N_LAYERS,
                         "ffn": FFN, "dropout": DROPOUT, "pos": "learned", "softcap": None,
                         "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
                         "warmup": WARMUP, "params_M": n_params / 1e6},
        "training": {"max_steps": MAX_STEPS, "mask_prob": 0.15, "lam": 0, "seed": SEED},
        "best_val": best,
        "final_val": traj[-1] if traj else None,
        "trajectory": traj,
        "references": {"unigram": 0.0923, "depth2_lam0_ourstack": 0.1090,
                       "depth4_big_ourstack": 0.1119, "sweep_best_ourstack": 0.1120,
                       "count_model_overall": 0.1272, "count_model_clean_ceiling": 0.1352},
    }
    gap = best["top1"] - 0.1352
    if best["top1"] >= 0.1352 - 0.005:
        verdict = ("Vanilla encoder REACHES the count-model ceiling -> the deficit was OUR repurposed "
                   "stack (rotary/softcap/Muon/CLM-LRs). Switch the MLM encoder architecture.")
        label = "our-stack-was-the-problem"
    elif best["top1"] >= 0.1120 + 0.01:
        verdict = ("Vanilla encoder beats our stack but not the count ceiling -> partial: standard "
                   "architecture helps, but local n-gram stats are hard to fully learn+generalize.")
        label = "partial-improvement"
    else:
        verdict = ("Vanilla encoder lands at ~0.11 too -> 0.135 is near the real generalization ceiling "
                   "for a learned model on this within-read task; the count model just memorizes local "
                   "stats. Stop chasing within-read; the lever is added information / Layer 2.")
        label = "ceiling-is-real"
    summary["verdict"] = {"label": label, "best_minus_count_ceiling": gap, "interpretation": verdict}

    (OUT_DIR / "vanilla_result.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[vanilla] BEST val top1={best['top1']:.4f} @ step {best['step']} "
          f"(vs our-stack 0.1120, count-model 0.1352)")
    print(f"[vanilla] verdict [{label}]: {verdict}")
    print(f"[vanilla] wrote {OUT_DIR/'vanilla_result.json'}")


if __name__ == "__main__":
    main()
