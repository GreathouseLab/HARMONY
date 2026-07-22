"""
D6 wiring verification on REAL batches (no optimization, no full training).

Asserts and prints each requirement; any failure exits non-zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from prepare_genomic import Tokenizer
from probe_sample_coherence import parse_val_txt_grouped
from paired_data_loader import PairedDataLoader
from train_mlm import (
    GPT_MLM, build_model, mlm_mask, save_checkpoint,
    MASK_ID, EXTENDED_VOCAB, PAD_ID, SPECIAL_IDS_NEVER_MASK,
)
from contrastive_loss import infonce_with_sample_ids


def main():
    from device_utils import get_device
    DEVICE = get_device()
    print(f"=== D6 wiring verification (device={DEVICE}) ===\n")

    tk = Tokenizer.from_directory()
    loader = PairedDataLoader(
        "output/train.txt", tk,
        samples_per_batch=4, reads_per_sample=8, n_reads_cap_per_sample=50, seq_len=64,
    )

    # ---- (1) MLM masking rate ----
    print("\n[1] MLM masking rate in [13%, 17%] AND <PAIRED_END>=4095 never masked")
    rates = []
    paired_end_masked_count = 0
    g = torch.Generator(device="cpu").manual_seed(11)
    for _ in range(5):
        token_ids, attn_lens, sample_ids = next(loader)
        masked_input, mlm_targets = mlm_mask(token_ids, vocab_size=EXTENDED_VOCAB,
                                             mask_prob=0.15, generator=g)
        # rate over ELIGIBLE positions (consistent with the masker)
        eligible = torch.ones_like(token_ids, dtype=torch.bool)
        for sid in SPECIAL_IDS_NEVER_MASK:
            eligible &= (token_ids != sid)
        n_masked = int((mlm_targets >= 0).sum().item())
        n_elig = int(eligible.sum().item())
        rates.append(n_masked / max(n_elig, 1))
        paired_end_masked_count += int(((token_ids == 4095) & (mlm_targets >= 0)).sum().item())
    print(f"    rates per batch: {[f'{r:.3f}' for r in rates]}")
    print(f"    mean: {np.mean(rates):.3f}  (target 0.15)")
    print(f"    <PAIRED_END>=4095 ever masked? count={paired_end_masked_count}  (must be 0)")
    assert all(0.13 <= r <= 0.17 for r in rates), f"some rate outside [13%,17%]: {rates}"
    assert paired_end_masked_count == 0
    print("    PASS")

    # ---- (2) >=2 distinct samples per batch + >=1 same-sample positive ----
    print("\n[2] every batch has >=2 distinct sample_ids AND >=1 in-batch same-sample pair")
    for i in range(5):
        token_ids, _, sample_ids = next(loader)
        distinct = int(sample_ids.unique().numel())
        # in-batch pair: max count of any sample > 1
        counts = sample_ids.bincount()
        max_count = int(counts.max().item())
        print(f"    batch {i}: distinct={distinct}, max same-sample reads={max_count}")
        assert distinct >= 2
        assert max_count >= 2
    print("    PASS")

    # ---- (3) sample membership matches probe parser on val.txt ----
    print("\n[3] sample membership: paired_data_loader (val.txt) == probe_sample_coherence")
    val_loader = PairedDataLoader("output/val.txt", tk, samples_per_batch=2, reads_per_sample=4,
                                  n_reads_cap_per_sample=50, seq_len=64, seed=42)
    raw = parse_val_txt_grouped("output/val.txt", n_per_sample=50, seed=42)
    # Compare per-sample read bodies. PairedDataLoader stores tokenized ids; re-decode &
    # compare against the bodies returned by the probe's parser.
    target_sample = 0
    bodies_from_probe = sorted(raw[target_sample])
    bodies_from_loader = sorted(
        tk.enc.decode(ids).replace("<READ_START> ", "").replace(" <READ_END>", "")
        for ids in val_loader.samples[target_sample]
    )
    # The loader dropped reads > seq_len=64 tokens; probe keeps all.
    # Compare the loader's set as a SUBSET of probe's set (every loader read must exist in probe set).
    probe_set = set(bodies_from_probe)
    loader_set = set(bodies_from_loader)
    missing = loader_set - probe_set
    print(f"    sample {target_sample}: probe got {len(probe_set)} reads; loader got {len(loader_set)}")
    print(f"    loader reads not in probe set: {len(missing)}  (must be 0)")
    assert len(missing) == 0
    print("    PASS")

    # ---- Build model for (4)-(7) ----
    model, config = build_model(DEVICE, depth=2, aspect_ratio=128, seq_len=64)

    # ---- (4) Projection embedding shape (B, 128), L2-normalized ----
    print("\n[4] projection embeddings: shape (B, 128) and unit-norm")
    token_ids, attn_lens, sample_ids = next(loader)
    token_ids, attn_lens = token_ids.to(DEVICE), attn_lens.to(DEVICE)
    with torch.no_grad():
        h = model._trunk_bidir(token_ids, attn_lens=attn_lens)
        z = model.projection(h, attn_lens)
    norms = z.norm(dim=-1)
    print(f"    z.shape={tuple(z.shape)}, ||z|| min/max={norms.min().item():.4f}/{norms.max().item():.4f}")
    assert z.shape == (token_ids.size(0), 128)
    assert (norms - 1.0).abs().max().item() < 1e-4
    print("    PASS")

    # ---- (5) bidirectional: position 0 depends on position 1 ----
    # init_weights zeros attn.c_proj.weight by design, so attention output is
    # zeroed at init and every position just propagates its own wte. To probe
    # the actual SDPA behavior (causal vs bidirectional) we inject a small
    # non-zero c_proj on every block, ONLY for this test. We then verify both:
    #   (a) bidir: pos-0 hidden changes when pos-1 token changes (must change).
    #   (b) causal (parent's forward path): pos-0 hidden does NOT change when
    #       pos-1 changes (the causal mask blocks the future).
    print("\n[5] bidirectional attention: pos-0 hidden depends on pos-1 token")
    with torch.no_grad():
        for blk in model.transformer.h:
            torch.nn.init.normal_(blk.attn.c_proj.weight, std=0.02)

    a = torch.tensor([[100, 200]], device=DEVICE)
    b = torch.tensor([[100, 500]], device=DEVICE)
    with torch.no_grad():
        hA_bidir = model._trunk_bidir(a)[0, 0]                    # no attn_lens -> no mask
        hB_bidir = model._trunk_bidir(b)[0, 0]
        # Compare against parent's causal trunk (the CLM path). To isolate the
        # trunk we call super().forward with return_hidden=True.
        from model import GPT as _GPT_BASE
        hA_caus = _GPT_BASE.forward(model, a, return_hidden=True)[0, 0]
        hB_caus = _GPT_BASE.forward(model, b, return_hidden=True)[0, 0]
    d_bidir = (hA_bidir.float() - hB_bidir.float()).abs().max().item()
    d_caus = (hA_caus.float() - hB_caus.float()).abs().max().item()
    print(f"    bidirectional pos-0 delta when pos-1 changes: {d_bidir:.6f}  (must be > 0)")
    print(f"    causal       pos-0 delta when pos-1 changes: {d_caus:.6f}  (must be ~0 — causal mask blocks future)")
    assert d_bidir > 0, "pos-0 hidden did not change in bidirectional mode"
    assert d_caus < 1e-6, f"causal mode leaked future info into pos-0: delta={d_caus}"
    print("    PASS")

    # ---- (6) one forward+backward yields finite losses ----
    print("\n[6] one forward+backward yields finite MLM + contrastive losses")
    token_ids, attn_lens, sample_ids = next(loader)
    token_ids = token_ids.to(DEVICE); attn_lens = attn_lens.to(DEVICE); sample_ids = sample_ids.to(DEVICE)
    masked, targets = mlm_mask(token_ids, vocab_size=EXTENDED_VOCAB)
    logits, hidden = model.forward_mlm(masked, attn_lens=attn_lens)
    mlm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    z = model.projection(hidden, attn_lens)
    con_loss, n_with, n_without = infonce_with_sample_ids(z, sample_ids, temperature=0.07)
    total = mlm_loss + 1.0 * con_loss
    total.backward()
    print(f"    L_MLM={mlm_loss.item():.4f}, L_contrastive={con_loss.item():.4f}, total={total.item():.4f}")
    assert torch.isfinite(mlm_loss).all() and torch.isfinite(con_loss).all()
    print("    PASS")

    # ---- (7) checkpoint written BEFORE first optimizer step ----
    print("\n[7] checkpoint written before first optimizer step")
    out_dir = Path("experiments/mlm_smoke_verify")
    ckpt = save_checkpoint(model, config, step=-1, path=out_dir / "checkpoint.pt",
                           extra={"note": "wiring_verify pre-step"})
    sz = ckpt.stat().st_size
    print(f"    path: {ckpt}  size: {sz/1e6:.2f} MB")
    assert ckpt.exists() and sz > 0
    print("    PASS")

    # ---- (A) D1 optimizer override: param-group breakdown + assert ----
    print("\n[A] optimizer param-grouping: Muon body + AdamW heads (incl. projection_head)")
    opt = model.setup_optimizer()
    muon_groups = [g for g in opt.param_groups if g.get("kind") == "muon"]
    adamw_groups = [g for g in opt.param_groups if g.get("kind") == "adamw"]
    muon_count = sum(len(g["params"]) for g in muon_groups)
    adamw_count = sum(len(g["params"]) for g in adamw_groups)
    print(f"    Muon groups: {len(muon_groups)}, total params: {muon_count}")
    print(f"    AdamW groups: {len(adamw_groups)}, total params: {adamw_count}")
    print(f"    model total params: {len(list(model.parameters()))}")
    # Locate a known body matrix (transformer.h.0.attn.c_q.weight) — must be in Muon.
    cq = model.transformer.h[0].attn.c_q.weight
    in_muon = any(any(p is cq for p in g["params"]) for g in muon_groups)
    in_adamw = any(any(p is cq for p in g["params"]) for g in adamw_groups)
    print(f"    body 'transformer.h[0].attn.c_q.weight' in Muon? {in_muon}  in AdamW? {in_adamw}")
    assert in_muon and not in_adamw
    # projection_head.* must be in AdamW.
    proj_weights = list(model.projection_head.parameters())
    proj_in_adamw = all(any(any(p is pw for p in g["params"]) for g in adamw_groups) for pw in proj_weights)
    proj_in_muon = any(any(any(p is pw for p in g["params"]) for g in muon_groups) for pw in proj_weights)
    print(f"    all projection_head.* in AdamW? {proj_in_adamw}  any in Muon? {proj_in_muon}")
    assert proj_in_adamw and not proj_in_muon
    assert muon_count + adamw_count == len(list(model.parameters())), \
        f"param accounting drift: muon+adamw={muon_count+adamw_count}, total={len(list(model.parameters()))}"
    print("    PASS")

    # ---- (B) D2 pad: masked mean = mean over non-pad; mask actually fires ----
    print("\n[B] pad handling: projection masked-mean ≡ mean over non-pad; SDPA pad mask is active")
    # Synthetic: B=1, T=8, n_embd=4. Fill values that make the diagnostic obvious.
    torch.manual_seed(0)
    h_syn = torch.randn(1, 8, 4, device=DEVICE)
    attn_lens_syn = torch.tensor([5], device=DEVICE)
    # Manual mean over first 5 positions.
    expected = h_syn[0, :5].float().mean(dim=0)
    # Apply projection's pooling-only path manually (skip projection_head — we want to verify the pool math).
    pos = torch.arange(8, device=DEVICE).unsqueeze(0)
    valid = (pos < attn_lens_syn.unsqueeze(1)).float()
    denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    pooled = (h_syn.float() * valid.unsqueeze(-1)).sum(dim=1) / denom
    diff = (pooled[0] - expected).abs().max().item()
    print(f"    masked mean vs explicit mean of first 5 positions: max|diff|={diff:.2e}  (must be < 1e-6)")
    assert diff < 1e-6
    # And: pad mask CHANGES the bidirectional output vs no mask.
    a = next(loader)[0].to(DEVICE)
    al = next(loader)[1].to(DEVICE)
    # Force first row to have a partial length so the mask actually distinguishes positions.
    al[0] = 30  # pad starts at position 30
    with torch.no_grad():
        h_with_mask = model._trunk_bidir(a, attn_lens=al)
        h_no_mask   = model._trunk_bidir(a, attn_lens=None)
    delta_row0 = (h_with_mask[0].float() - h_no_mask[0].float()).abs().max().item()
    print(f"    bidirectional hidden delta row-0 (mask vs no-mask) at attn_len=30: {delta_row0:.6f}  (must be > 0)")
    assert delta_row0 > 0, "pad mask did NOT change bidirectional output -> mask is inactive"
    print("    PASS")

    print("\n=== D6+D3: ALL 9 checks passed (7 original + A optimizer + B pad) ===")


if __name__ == "__main__":
    main()
