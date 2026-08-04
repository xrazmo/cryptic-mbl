"""
model.py — Task 6 (spec §5)

Two pieces:

1. `PocketEncoder` — the shared-weight backbone. This is a placeholder
   SE(3)-invariant message-passing network (distance/angle-based, using
   torch_geometric's standard layers) that reproduces the *interface*
   EZSpecificity's backbone should have: pocket graph (with `pos`, `x`,
   `edge_index`) in, fixed-length embedding out. Swap `PocketEncoder`'s
   internals for EZSpecificity's actual SE(3)-equivariant cross-attention
   layers (Cui et al., Nature 2025) when that codebase is available —
   everything downstream (Siamese wrapper, triplet loss, training loop)
   only depends on the forward(data) -> Tensor[embed_dim] contract, not on
   what's inside. This placeholder is NOT a substitute for the real
   backbone — it's here so the rest of the pipeline is testable/runnable
   before that integration lands.

2. `SiameseTripletModel` — wraps the shared encoder for triplet-loss
   training, plus `triplet_loss`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data


class _DistanceAwareConv(MessagePassing):
    """
    Minimal SE(3)-invariant message-passing layer: messages are a function
    of (neighbor node features, pairwise distance), never of raw absolute
    coordinates directly, so the layer is translation/rotation invariant
    by construction. Angular/orientation information (which a true
    equivariant layer like EZSpecificity's would preserve through the
    network rather than collapsing to scalar distances at every layer) is
    the main capability this placeholder is missing — that's the specific
    gap the EZSpecificity swap-in is meant to close, most importantly for
    resolving Zn coordination geometry and carbonyl orientation.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim), nn.SiLU()
        )

    def forward(self, x, pos, edge_index):
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j):
        dist = (pos_i - pos_j).norm(dim=-1, keepdim=True)
        return self.msg_mlp(torch.cat([x_i, x_j, dist], dim=-1))

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class PocketEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            _DistanceAwareConv(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.input_proj(data.x)
        for conv, norm in zip(self.layers, self.norms):
            residual = x
            x = conv(x, data.pos, data.edge_index)
            x = norm(x + residual)
            x = self.dropout(x)
        pooled = global_mean_pool(x, data.batch if hasattr(data, "batch") else
                                   torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        embedding = self.readout(pooled)
        return F.normalize(embedding, p=2, dim=-1)  # unit-norm, so Euclidean == scaled cosine


class SiameseTripletModel(nn.Module):
    """Thin wrapper: shared-weight encoder applied to anchor/positive/negative."""

    def __init__(self, encoder: PocketEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, anchor: Data, positive: Data, negative: Data):
        return self.encoder(anchor), self.encoder(positive), self.encoder(negative)

    def embed(self, data: Data) -> torch.Tensor:
        return self.encoder(data)


def triplet_loss(
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    negative_emb: torch.Tensor,
    margin: float = 0.3,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    L = max(0, margin + d(a,p) - d(a,n)), per-sample; sample_weight applies
    the confidence-tier loss weighting from spec §8 (pass
    utils.TIER_LOSS_WEIGHTS[tier] per triplet, e.g. min tier across the
    three instances, upstream in the training loop).
    """
    d_ap = F.pairwise_distance(anchor_emb, positive_emb, p=2)
    d_an = F.pairwise_distance(anchor_emb, negative_emb, p=2)
    per_sample = F.relu(margin + d_ap - d_an)
    if sample_weight is not None:
        per_sample = per_sample * sample_weight
        return per_sample.sum() / sample_weight.sum().clamp_min(1e-8)
    return per_sample.mean()
