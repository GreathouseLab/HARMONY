"""
InfoNCE / NT-Xent contrastive loss with in-batch negatives, sample-id positives.

Used by the MLM + contrastive arm of HARMONY. Pure function, no global state.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def infonce_with_sample_ids(z: torch.Tensor, sample_ids: torch.Tensor,
                            temperature: float = 0.07):
    """InfoNCE loss with in-batch negatives, positives drawn by sample_id equality.

    Args:
        z: L2-normalized read embeddings, shape (B, D).
        sample_ids: LongTensor of shape (B,) — sample provenance per row.
        temperature: scalar temperature (default 0.07).

    Returns:
        (loss, n_anchors_with_positive, n_anchors_no_positive):
            loss: scalar — mean InfoNCE over anchors that have >=1 positive.
                  Tensor zero (with grad-enabled compute graph) if no anchor has positives.
            n_anchors_with_positive: int.
            n_anchors_no_positive: int — caller can assert this is 0 / small.
    """
    B = z.size(0)
    assert sample_ids.dim() == 1 and sample_ids.size(0) == B
    assert abs(z.norm(dim=-1).mean().item() - 1.0) < 1e-2, "z must be L2-normalized"

    # Similarity matrix (B, B), exclude self-pairs.
    sim = z @ z.t() / temperature                                # (B, B)
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))              # self -> -inf so it's never picked

    pos_mask = (sample_ids.view(-1, 1) == sample_ids.view(1, -1))
    pos_mask = pos_mask & ~self_mask                              # exclude self
    has_pos = pos_mask.any(dim=1)                                 # (B,) bool

    n_with = int(has_pos.sum().item())
    n_without = B - n_with
    if n_with == 0:
        return z.sum() * 0.0, 0, n_without                        # keeps autograd graph

    # log-softmax over all non-self negatives+positives, then sum log-probs of positives.
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)    # (B, B); self entries are -inf
    # Mean log-prob over positives per anchor. Use torch.where to avoid -inf*0 = NaN.
    pos_count = pos_mask.sum(dim=1).clamp(min=1).float()          # avoid div0 (rows with 0 will be masked out)
    masked_log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    mean_log_prob_pos = masked_log_prob.sum(dim=1) / pos_count

    # Average only over anchors with at least one positive.
    loss = -mean_log_prob_pos[has_pos].mean()
    return loss, n_with, n_without


if __name__ == "__main__":
    torch.manual_seed(0)
    # Toy: 2 clusters of 4 vectors each, separated; vs shuffled labels.
    cluster_a = F.normalize(torch.randn(4, 8) + torch.tensor([3.0] + [0.0] * 7), dim=-1)
    cluster_b = F.normalize(torch.randn(4, 8) + torch.tensor([0.0] * 7 + [3.0]), dim=-1)
    z = torch.cat([cluster_a, cluster_b], dim=0)
    ids_good = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    ids_shuf = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    loss_good, nw, nwo = infonce_with_sample_ids(z, ids_good)
    loss_shuf, _, _ = infonce_with_sample_ids(z, ids_shuf)
    print(f"separated  loss = {loss_good.item():.4f}  (n_with_pos={nw}, n_without={nwo})")
    print(f"shuffled   loss = {loss_shuf.item():.4f}")
    assert loss_good.item() < loss_shuf.item(), "Separated case should have lower InfoNCE"
    print("self-test: PASS")
