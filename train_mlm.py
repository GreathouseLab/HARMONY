"""
MLM + InfoNCE contrastive training arm for HARMONY.

Subclasses model.GPT to add:
  - bidirectional forward path (SDPA without is_causal/attn_mask)
  - mlm_head reuse of lm_head with softcap (matches CLM logit pipeline)
  - projection_head MLP: n_embd -> 256 -> 128, applied to mean-pooled per-read hidden

Does NOT modify model.py. The parent's CLM forward(idx, targets) path is
bit-identical (D4 verification).

Fail-closed defaults baked in from the 2026-05-15 incident lesson:
  - MAX_RUNTIME_HOURS wall guard
  - checkpoint written BEFORE first optimizer step
  - abort on NaN/inf or loss>100
  - one-shot OOM-halving retry
  - assert batch has >=2 distinct samples each step
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, GPTConfig, MuonAdamW
from model import norm, has_ve, apply_rotary_emb
from prepare_genomic import Tokenizer
from paired_data_loader import PairedDataLoader
from contrastive_loss import infonce_with_sample_ids
from device_utils import (get_device, empty_cache, describe,
                           init_distributed, is_main, dist_active,
                           broadcast_parameters, allreduce_gradients, barrier, cleanup_distributed)


PROJECT_DIR = Path(__file__).parent
MASK_ID = 4096
EXTENDED_VOCAB = 4097
PAD_ID = 4090                          # <|bos|>, see paired_data_loader
SPECIAL_IDS_NEVER_MASK = {4090, 4091, 4092, 4093, 4094, 4095, MASK_ID}

MAX_RUNTIME_HOURS_DEFAULT = 12.0
MAX_LOSS_BEFORE_ABORT = 100.0


# ============================================================
# Subclass: bidirectional GPT with MLM + projection heads
# ============================================================

class GPT_MLM(GPT):
    """Bidirectional MLM-and-contrastive head atop the canonical GPT.

    forward(idx, targets)            -> super().forward (CAUSAL, bit-identical to CLM baseline)
    forward(idx, return_hidden=True) -> BIDIRECTIONAL hidden state, used by probe and projection
    forward_mlm(idx)                 -> (mlm_logits, hidden_bidir) for MLM + contrastive paths
    """
    def __init__(self, config: GPTConfig, projection_dim: int = 128):
        super().__init__(config)
        # Reuse lm_head for MLM logits (softcap applied later, mirroring parent forward).
        # Projection MLP on per-read mean-pooled hidden.
        self.projection_head = nn.Sequential(
            nn.Linear(config.n_embd, 256, bias=True),
            nn.GELU(),
            nn.Linear(256, projection_dim, bias=True),
        )
        # MLM logit softcap (governor on prediction confidence). 15.0 is the CLM-inherited
        # default; set to 0/None to disable. Configurable so Step-2 sweeps can test it.
        self.mlm_softcap = 15.0

    # ---- Bidirectional attention loop with key-padding mask (parent's modules) ----

    def _build_key_padding_mask(self, attn_lens, T, device):
        """Returns SDPA attn_mask of shape (B, 1, 1, T): True at valid key positions,
        False at pad keys. Broadcasts across heads and queries."""
        pos = torch.arange(T, device=device).unsqueeze(0)         # (1, T)
        valid = pos < attn_lens.to(device).unsqueeze(1)            # (B, T) bool
        return valid.unsqueeze(1).unsqueeze(1)                    # (B, 1, 1, T)

    def _attn_bidir(self, attn, x, ve, cos_sin, attn_mask=None):
        B, T, C = x.size()
        q = attn.c_q(x).view(B, T, attn.n_head, attn.head_dim)
        k = attn.c_k(x).view(B, T, attn.n_kv_head, attn.head_dim)
        v = attn.c_v(x).view(B, T, attn.n_kv_head, attn.head_dim)
        if ve is not None:
            ve = ve.view(B, T, attn.n_kv_head, attn.head_dim)
            gate = 2 * torch.sigmoid(attn.ve_gate(x[..., :attn.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        k = k.repeat_interleave(attn.n_head // attn.n_kv_head, dim=2)
        v = v.repeat_interleave(attn.n_head // attn.n_kv_head, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # Bidirectional SDPA with optional key-padding mask. True = attend.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return attn.c_proj(y)

    def _block_bidir(self, block, x, ve, cos_sin, attn_mask=None):
        x = x + self._attn_bidir(block.attn, norm(x), ve, cos_sin, attn_mask=attn_mask)
        x = x + block.mlp(norm(x))
        return x

    def _trunk_bidir(self, idx, attn_lens=None):
        """Same trunk as parent.GPT.forward but bidirectional + optional pad mask."""
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        attn_mask = self._build_key_padding_mask(attn_lens, T, idx.device) if attn_lens is not None else None

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = self._block_bidir(block, x, ve, cos_sin, attn_mask=attn_mask)
        x = norm(x)
        return x

    # ---- Overrides ----

    def forward(self, idx, targets=None, reduction='mean', return_hidden=False, attn_lens=None):
        """If return_hidden=True -> bidirectional hidden (for probe + projection).
           Else -> delegate to parent (CAUSAL, bit-identical CLM)."""
        if return_hidden:
            return self._trunk_bidir(idx, attn_lens=attn_lens)
        return super().forward(idx, targets=targets, reduction=reduction, return_hidden=False)

    def forward_mlm(self, idx, attn_lens=None):
        """Returns (mlm_logits[B, T, V], hidden_bidir[B, T, n_embd])."""
        h = self._trunk_bidir(idx, attn_lens=attn_lens)
        logits = self.lm_head(h).float()
        cap = getattr(self, "mlm_softcap", 15.0)
        if cap and cap > 0:
            logits = cap * torch.tanh(logits / cap)             # softcap (disabled if cap<=0)
        return logits, h

    def projection(self, hidden_bidir, attn_lens):
        """Masked mean-pool over non-pad positions, then projection MLP + L2-norm.
        Returns z (B, projection_dim)."""
        B, T, D = hidden_bidir.size()
        device = hidden_bidir.device
        pos = torch.arange(T, device=device).unsqueeze(0)        # (1, T)
        valid = (pos < attn_lens.to(device).unsqueeze(1)).float()  # (B, T)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)     # (B, 1)
        pooled = (hidden_bidir.float() * valid.unsqueeze(-1)).sum(dim=1) / denom  # (B, D)
        z = self.projection_head(pooled)
        z = F.normalize(z, p=2, dim=-1)
        return z

    # ---- D1: optimizer override (Muon body + AdamW heads incl. projection_head) ----

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.6, matrix_lr=0.04,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5,
                        projection_lr=0.004):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        projection_params = list(self.projection_head.parameters())

        # Strict accounting — mirrors model.py's assert (NOT relaxed).
        total_accounted = (len(matrix_params) + len(embedding_params) + len(lm_head_params) +
                           len(value_embeds_params) + len(resid_params) + len(x0_params) +
                           len(projection_params))
        n_self = len(list(self.parameters()))
        assert n_self == total_accounted, (
            f"GPT_MLM optimizer param-accounting drift: have {n_self}, accounted {total_accounted}"
        )

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"[GPT_MLM.setup_optimizer] Muon body + AdamW heads (incl. projection_head). "
              f"AdamW LR scale = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params,        lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params,      lr=embedding_lr   * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params,   lr=embedding_lr   * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params,          lr=scalar_lr * 0.01,                  betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params,             lr=scalar_lr,                         betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=projection_params,     lr=projection_lr * dmodel_lr_scale,  betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer


# ============================================================
# MLM masking (BERT 80/10/10)
# ============================================================

def mlm_mask(token_ids: torch.Tensor, vocab_size: int,
             mask_prob: float = 0.15,
             never_mask_ids: set[int] = SPECIAL_IDS_NEVER_MASK,
             generator: torch.Generator | None = None):
    """BERT-style 80/10/10 MLM masking.

    Args:
        token_ids: (B, T) input ids.
        vocab_size: max id for random replacement (exclusive of MASK).
    Returns:
        (masked_input, mlm_targets) — mlm_targets has the gold token at masked positions
        and -1 at unmasked positions (so cross_entropy with ignore_index=-1 only counts masked).
    """
    B, T = token_ids.shape
    device = token_ids.device

    # Eligibility: any id not in never_mask_ids is eligible to be masked.
    eligible = torch.ones((B, T), dtype=torch.bool, device=device)
    for sid in never_mask_ids:
        eligible &= (token_ids != sid)

    if generator is None:
        u = torch.rand((B, T), device=device)
    else:
        u = torch.rand((B, T), device=device, generator=generator)
    mask_decision = (u < mask_prob) & eligible                    # (B, T) bool

    # Of those, 80% -> [MASK], 10% -> random eligible id, 10% -> unchanged.
    if generator is None:
        u2 = torch.rand((B, T), device=device)
    else:
        u2 = torch.rand((B, T), device=device, generator=generator)
    do_mask = mask_decision & (u2 < 0.8)
    do_rand = mask_decision & (u2 >= 0.8) & (u2 < 0.9)

    masked = token_ids.clone()
    masked[do_mask] = MASK_ID
    if do_rand.any():
        n = int(do_rand.sum().item())
        # Random replacement from non-special id range [0, 4090).
        if generator is None:
            rand_ids = torch.randint(0, 4090, (n,), device=device)
        else:
            rand_ids = torch.randint(0, 4090, (n,), device=device, generator=generator)
        masked[do_rand] = rand_ids

    targets = torch.full_like(token_ids, fill_value=-1)
    targets[mask_decision] = token_ids[mask_decision]
    return masked, targets


# ============================================================
# Build model + optimizer
# ============================================================

def build_model(device, depth=2, aspect_ratio=128, seq_len=64):
    base_dim = depth * aspect_ratio
    head_dim = 128
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_head = max(1, model_dim // head_dim)
    config = GPTConfig(
        n_layer=depth, n_head=n_head, n_kv_head=n_head, n_embd=model_dim,
        sequence_len=seq_len,
        window_pattern="L",   # not used for bidirectional path; CLM path stays causal
        vocab_size=EXTENDED_VOCAB,
    )
    model = GPT_MLM(config).to(device)
    model.init_weights()
    return model, config


def eval_val_mlm(model, tk, val_path, device, n_per_sample=50, parse_seed=42,
                 mask_seed=7, mask_prob=0.15, t_max=64):
    """Periodic VAL MLM eval: MSK-only, DNA-only top-1/top-5/CE on val reads.

    This is the metric we actually care about (within-read masked DNA prediction) and
    was previously NOT measured during training — only Probe-1 coherence AUC was.

    Determinism / isolation guarantees:
      - masking uses a fixed CPU torch.Generator seeded at `mask_seed`, so the masked
        positions are IDENTICAL at every eval step -> the val trajectory is clean, not noisy.
      - because it is a CPU generator (not the global/MPS RNG), it does NOT perturb training;
        a run with this eval is bit-identical in its train trajectory to one without it.
    Same protocol/exclusions as floor_diag_big.py (excludes special ids {4090..4096}, MSK
    branch only). Not bit-identical masking to that standalone script (different RNG stream),
    but the same scoring; internal step-to-step consistency is what the trajectory needs.
    """
    from probe_sample_coherence import parse_val_txt_grouped
    samples = parse_val_txt_grouped(val_path, n_per_sample=n_per_sample, seed=parse_seed)
    flat = [(s, b) for s in sorted(samples.keys()) for b in samples[s]]
    gen = torch.Generator().manual_seed(mask_seed)  # CPU, deterministic, isolated from global RNG
    n_pos = n_top1 = n_top5 = 0
    logp_sum = 0.0
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
            masked = masked_cpu.to(device)
            attn_lens = torch.tensor([L], dtype=torch.long, device=device)
            logits, _ = model.forward_mlm(masked, attn_lens=attn_lens)
            probs = F.softmax(logits[0].float(), dim=-1)
            for p in range(L):
                if int(targets_cpu[0, p].item()) == -1:
                    continue
                if int(masked_cpu[0, p].item()) != MASK_ID:        # MSK branch only
                    continue
                true_id = int(tok_cpu[0, p].item())
                if true_id in SPECIAL_IDS_NEVER_MASK:               # DNA only (parity w/ floor_diag)
                    continue
                top5 = probs[p].topk(5).indices.tolist()
                n_pos += 1
                if top5[0] == true_id:
                    n_top1 += 1
                if true_id in top5:
                    n_top5 += 1
                logp_sum += float(torch.log(probs[p, true_id].clamp(min=1e-12)).item())
    if was_training:
        model.train()
    if n_pos == 0:
        return {"n_positions": 0, "top1": float("nan"), "top5": float("nan"), "ce_nats": float("nan")}
    return {"n_positions": n_pos, "top1": n_top1 / n_pos,
            "top5": n_top5 / n_pos, "ce_nats": -logp_sum / n_pos}


def eval_train_mlm(model, token_id_lists, device, mask_seed=7, mask_prob=0.15, t_max=64):
    """MSK-DNA top-1/CE on a FIXED set of already-tokenized TRAINING reads.

    This is the memorization sanity check (Nick's recommendation): if the model cannot drive
    TRAIN top-1 well above the ~0.11 val plateau given enough passes, it cannot fit the data
    and any 'ceiling' conclusion is confounded by a fitting/optimization problem, not the data.
    Same scoring/exclusions as eval_val_mlm; deterministic mask; RNG-isolated (CPU generator)."""
    gen = torch.Generator().manual_seed(mask_seed)
    n_pos = n_top1 = 0
    logp_sum = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for ids in token_id_lists:
            L = len(ids)
            padded = ids + [PAD_ID] * (t_max - L) if L < t_max else ids[:t_max]
            L = min(L, t_max)
            tok_cpu = torch.tensor([padded], dtype=torch.long)
            masked_cpu, targets_cpu = mlm_mask(tok_cpu, vocab_size=EXTENDED_VOCAB,
                                               mask_prob=mask_prob, generator=gen)
            logits, _ = model.forward_mlm(masked_cpu.to(device),
                                          attn_lens=torch.tensor([L], dtype=torch.long, device=device))
            probs = F.softmax(logits[0].float(), dim=-1)
            for p in range(L):
                if int(targets_cpu[0, p].item()) == -1:
                    continue
                if int(masked_cpu[0, p].item()) != MASK_ID:
                    continue
                true_id = int(tok_cpu[0, p].item())
                if true_id in SPECIAL_IDS_NEVER_MASK:
                    continue
                n_pos += 1
                if int(probs[p].argmax().item()) == true_id:
                    n_top1 += 1
                logp_sum += float(torch.log(probs[p, true_id].clamp(min=1e-12)).item())
    if was_training:
        model.train()
    if n_pos == 0:
        return {"n_positions": 0, "top1": float("nan"), "ce_nats": float("nan")}
    return {"n_positions": n_pos, "top1": n_top1 / n_pos, "ce_nats": -logp_sum / n_pos}


def save_checkpoint(model, config, step, path: Path, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": config,                # GPTConfig dataclass; torch can pickle it
        "step": step,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)
    return path


# ============================================================
# Train loop
# ============================================================

def train(args):
    # Distributed data-parallel init (no-op / single-device when not launched under mpiexec/torchrun).
    dinfo = init_distributed()
    device = dinfo["device"]
    rank, world_size = dinfo["rank"], dinfo["world_size"]
    main = is_main()

    def _p(*a, **k):            # print only on rank 0 (avoids N-way console spam)
        if main:
            print(*a, **k)

    _p(f"[train_mlm] {describe(device)}"
       + (f" | DDP world_size={world_size} (backend-averaged grads)" if dist_active() else ""))
    tk = Tokenizer.from_directory()
    _p(f"[train_mlm] tokenizer vocab_size={tk.get_vocab_size()}; extending to {EXTENDED_VOCAB} (added [MASK]={MASK_ID})")

    loader = PairedDataLoader(
        txt_path=args.train_txt,
        tokenizer=tk,
        samples_per_batch=args.samples_per_batch,
        reads_per_sample=args.reads_per_sample,
        n_reads_cap_per_sample=args.reads_cap,
        seq_len=args.seq_len,
        seed=args.seed + rank,   # each rank draws a DIFFERENT data shard (the point of data parallelism)
    )

    device_batch_size = args.samples_per_batch * args.reads_per_sample
    # Fixed TRAIN-eval set for the memorization sanity check (built from already-tokenized
    # loader reads; deterministic; ~700 reads). Lets us watch TRAIN top-1 vs VAL top-1.
    train_eval_lists = [ids for s in sorted(loader.samples)[:14] for ids in loader.samples[s][:50]]
    _p(f"[train_mlm] train-eval memorization set: {len(train_eval_lists)} reads")
    model, config = build_model(device, depth=args.depth, aspect_ratio=args.aspect_ratio, seq_len=args.seq_len)
    model.mlm_softcap = args.mlm_softcap
    broadcast_parameters(model)   # DDP: all ranks start from rank-0 weights (no-op if single-device)
    n_params = sum(p.numel() for p in model.parameters())
    _p(f"[train_mlm] model: depth={args.depth} n_embd={config.n_embd} params={n_params/1e6:.2f}M")
    _p(f"[train_mlm] mlm_softcap={args.mlm_softcap} (<=0 = off) | train mask_prob={args.mask_prob}")

    # MuonAdamW with extended grouping (projection_head joins AdamW alongside lm_head/embeddings).
    optimizer = model.setup_optimizer(
        unembedding_lr=0.004, embedding_lr=0.6, matrix_lr=0.04,
        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5,
        projection_lr=0.004,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_mlm_log.jsonl"
    ckpt_path = out_dir / "checkpoint.pt"
    latest_path = out_dir / "latest.pt"
    best_path = out_dir / "best.pt"
    traj_path = out_dir / "probe_trajectory.csv"

    # Pre-step checkpoint (incident lesson). Rank 0 only — all ranks hold identical weights.
    if main:
        save_checkpoint(model, config, step=-1, path=ckpt_path, extra={"note": "pre-step-0 baseline"})
        _p(f"[train_mlm] pre-step checkpoint: {ckpt_path} ({ckpt_path.stat().st_size/1e6:.2f} MB)")

    eval_every = getattr(args, "eval_every", 0)
    val_txt_path = Path(getattr(args, "val_txt", "output/val.txt"))
    best_auc = float("-inf")
    best_step = -1
    best_val_ce = float("inf")          # lower is better; selects best_val.pt
    best_val_step = -1
    best_val_path = out_dir / "best_val.pt"
    val_n_per_sample = getattr(args, "val_n_per_sample", 50)
    val_mask_seed = getattr(args, "val_mask_seed", 7)
    if eval_every > 0 and main:
        with open(traj_path, "w") as f:
            # val_msk_* columns appended at the END so index-based readers of the
            # original 5 columns (step,wall,L_MLM,L_contrastive,probe1_auc) still work.
            f.write("step,wall_clock_s,L_MLM,L_contrastive,probe1_auc,"
                    "val_msk_top1,val_msk_top5,val_msk_ce,train_msk_top1,train_msk_ce\n")
        _p(f"[train_mlm] periodic Probe 1 + VAL-MLM every {eval_every} steps -> {traj_path}")

    t0 = time.time()
    max_seconds = args.max_runtime_hours * 3600
    init_losses = None
    last_losses = None

    # Only rank 0 writes the per-step jsonl; other ranks discard theirs (avoids a write race).
    with open(log_path if main else os.devnull, "w") as log_f:
        for step in range(args.max_steps):
            if time.time() - t0 > max_seconds:
                _p(f"[train_mlm] MAX_RUNTIME reached at step {step}; exiting cleanly.")
                break

            tries = 0
            while True:
                try:
                    token_ids, attn_lens, sample_ids = next(loader)
                    distinct = sample_ids.unique().numel()
                    if distinct < 2:
                        print("ABORT: batch has <2 distinct samples.")
                        sys.exit(2)

                    token_ids = token_ids.to(device)
                    attn_lens = attn_lens.to(device)
                    sample_ids = sample_ids.to(device)

                    masked_input, mlm_targets = mlm_mask(token_ids, vocab_size=EXTENDED_VOCAB,
                                                         mask_prob=args.mask_prob)
                    logits, hidden = model.forward_mlm(masked_input, attn_lens=attn_lens)
                    mlm_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        mlm_targets.view(-1),
                        ignore_index=-1,
                    )

                    z = model.projection(hidden, attn_lens)
                    con_loss, n_with_pos, n_without = infonce_with_sample_ids(z, sample_ids, temperature=0.07)

                    total = mlm_loss + args.lam * con_loss

                    if not torch.isfinite(mlm_loss).all() or not torch.isfinite(con_loss).all():
                        print(f"ABORT step {step}: non-finite loss "
                              f"(MLM={mlm_loss.item()}, contrastive={con_loss.item()})")
                        sys.exit(2)
                    if float(total.item()) > MAX_LOSS_BEFORE_ABORT:
                        print(f"ABORT step {step}: total loss {float(total.item()):.4f} > {MAX_LOSS_BEFORE_ABORT}")
                        sys.exit(2)

                    optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    allreduce_gradients(model)   # DDP: average grads across ranks (no-op if single)
                    optimizer.step()
                    break
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg or "memory" in msg:
                        if tries >= 1:
                            print(f"ABORT step {step}: OOM after halving once: {e}")
                            sys.exit(2)
                        tries += 1
                        new_K = max(2, loader.K)
                        new_M = max(1, loader.M // 2)
                        print(f"[train_mlm] OOM at step {step}; halving M {loader.M}->{new_M} and retrying.")
                        loader.M = new_M
                        empty_cache(device)
                        continue
                    raise

            entry = {
                "step": step,
                "mlm_loss": float(mlm_loss.item()),
                "contrastive_loss": float(con_loss.item()),
                "total_loss": float(total.item()),
                "n_distinct_samples": int(distinct),
                "n_anchors_with_pos": n_with_pos,
                "n_anchors_no_pos": n_without,
                "wall_s": time.time() - t0,
            }
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            if init_losses is None:
                init_losses = (float(mlm_loss.item()), float(con_loss.item()))
            last_losses = (float(mlm_loss.item()), float(con_loss.item()))

            if step % args.log_every == 0:
                _p(f"  step {step:04d} | MLM={mlm_loss.item():.4f} con={con_loss.item():.4f} "
                   f"total={total.item():.4f} | distinct_samples={distinct} | "
                   f"dt={(time.time()-t0)/(step+1)*1000:.0f}ms/step (cum)")

            # Periodic Probe 1 + VAL-MLM eval — rank 0 only (all ranks hold identical weights);
            # other ranks wait at the barrier so the collective all-reduce stays in lockstep.
            if eval_every > 0 and step > 0 and step % eval_every == 0:
              if main:
                from probe_sample_coherence import run_probe1
                model.eval()
                t_eval = time.time()
                eval_result = run_probe1(
                    checkpoint_path=ckpt_path,
                    val_path=val_txt_path,
                    output_path=out_dir / f"probe1_step{step:05d}.json",
                    n_reads_per_sample=50,
                    device=device,
                    model=model,
                    tokenizer=tk,
                    ckpt_meta={"n_embd": config.n_embd, "n_layer": config.n_layer,
                               "vocab_size": config.vocab_size, "seed": args.seed,
                               "hyperparams": {}, "val_bpb": None, "git_sha": None},
                )
                auc = float(eval_result.get("auc", float("nan")))
                # VAL-MLM eval (the metric we actually optimize for; model already in eval mode)
                vm = eval_val_mlm(model, tk, val_txt_path, device,
                                  n_per_sample=val_n_per_sample, mask_seed=val_mask_seed)
                tm = eval_train_mlm(model, train_eval_lists, device, mask_seed=val_mask_seed)
                wall = time.time() - t0
                with open(traj_path, "a") as f:
                    f.write(f"{step},{wall:.1f},{mlm_loss.item():.6f},{con_loss.item():.6f},{auc:.6f},"
                            f"{vm['top1']:.6f},{vm['top5']:.6f},{vm['ce_nats']:.6f},"
                            f"{tm['top1']:.6f},{tm['ce_nats']:.6f}\n")
                print(f"[mvt] step {step} | wall {wall:.0f}s | probe1_auc={auc:.4f} | "
                      f"val_msk top1={vm['top1']:.4f} CE={vm['ce_nats']:.4f} | "
                      f"TRAIN_msk top1={tm['top1']:.4f} CE={tm['ce_nats']:.4f} "
                      f"| eval_dt={time.time()-t_eval:.1f}s")
                # latest + best (probe-AUC) checkpoints — unchanged
                save_checkpoint(model, config, step=step, path=latest_path,
                                extra={"probe1_auc": auc, "wall_clock_s": wall,
                                       "val_msk_top1": vm["top1"], "val_msk_ce": vm["ce_nats"]})
                if auc > best_auc:
                    best_auc = auc
                    best_step = step
                    save_checkpoint(model, config, step=step, path=best_path,
                                    extra={"probe1_auc": auc, "wall_clock_s": wall,
                                           "is_best": True})
                    print(f"[mvt] NEW BEST AUC {auc:.4f} at step {step} -> {best_path}")
                # best_val checkpoint — selected on VAL-MLM CE (lower is better)
                if vm["ce_nats"] == vm["ce_nats"] and vm["ce_nats"] < best_val_ce:  # NaN-safe
                    best_val_ce = vm["ce_nats"]
                    best_val_step = step
                    save_checkpoint(model, config, step=step, path=best_val_path,
                                    extra={"probe1_auc": auc, "wall_clock_s": wall,
                                           "val_msk_top1": vm["top1"], "val_msk_ce": vm["ce_nats"],
                                           "is_best_val": True})
                    print(f"[mvt] NEW BEST VAL-MLM CE {vm['ce_nats']:.4f} (top1={vm['top1']:.4f}) "
                          f"at step {step} -> {best_val_path}")
                model.train()
              barrier()   # DDP: resync all ranks after rank-0 eval (keeps the all-reduce in lockstep)

    # Final checkpoint (rank 0 only; weights are identical across ranks after each all-reduce).
    if main:
        save_checkpoint(model, config, step=step, path=ckpt_path,
                        extra={"init_losses": init_losses, "last_losses": last_losses})
        _p(f"[train_mlm] final checkpoint: {ckpt_path} ({ckpt_path.stat().st_size/1e6:.2f} MB)")
    _p(f"[train_mlm] step 0 losses: {init_losses}")
    _p(f"[train_mlm] last losses:   {last_losses}")
    cleanup_distributed()
    return init_losses, last_losses, ckpt_path


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-txt", default="output/train.txt")
    ap.add_argument("--out-dir", default="experiments/mlm_smoke")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--max-runtime-hours", type=float, default=MAX_RUNTIME_HOURS_DEFAULT)
    ap.add_argument("--samples-per-batch", type=int, default=4)
    ap.add_argument("--reads-per-sample", type=int, default=8)
    ap.add_argument("--reads-cap", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--aspect-ratio", type=int, default=128)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--mlm-softcap", type=float, default=15.0,
                    help="Logit softcap for MLM head (CLM-inherited 15.0). <=0 disables it.")
    ap.add_argument("--mask-prob", type=float, default=0.15,
                    help="MLM mask probability used during TRAINING (val eval stays fixed at 0.15).")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=0,
                    help="Run Probe 1 every N steps (0 = never). MVT uses 2000.")
    ap.add_argument("--val-txt", type=str, default="output/val.txt")
    ap.add_argument("--val-n-per-sample", type=int, default=50,
                    help="Reads per val sample for the periodic VAL-MLM eval.")
    ap.add_argument("--val-mask-seed", type=int, default=7,
                    help="Fixed seed for deterministic val masking (kept constant across steps).")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    args = cli()
    train(args)
