"""
model.py — Task 6 (spec §5)

Three encoder pieces plus the Siamese wrapper:

1. `PocketEncoder` — the "flat" shared-weight backbone: one GNN over the
   full concatenated node-feature vector (amino-acid identity + chemistry/
   geometry + ESM2, ~97% of raw input dims). This is a placeholder
   SE(3)-invariant message-passing network (distance/sequence-adjacency
   based, using torch_geometric's standard layers) that reproduces the
   *interface* EZSpecificity's backbone should have: pocket graph (with
   `pos`, `x`, `edge_index`, `edge_attr`) in, fixed-length embedding out.
   Swap `PocketEncoder`'s internals for EZSpecificity's actual
   SE(3)-equivariant cross-attention layers (Cui et al., Nature 2025) when
   that codebase is available — everything downstream (Siamese wrapper,
   triplet loss, training loop) only depends on the forward(data) ->
   Tensor[embed_dim] contract, not on what's inside. This placeholder is
   NOT a substitute for the real backbone — it's here so the rest of the
   pipeline is testable/runnable before that integration lands.

2. `StructuralEncoder` + `ESM2ProjectionEncoder` + `BranchedPocketEncoder`
   — the alternative "branched" architecture: separate structural and
   ESM2 branches, each pooled and L2-normalized independently before a
   small fusion MLP, plus ESM2 modality dropout and an optional auxiliary
   structure-only loss (see BranchedPocketEncoder's docstring for why —
   the flat model's ESM2-dominated concatenation was found to actively
   hurt small-positive-count panels relative to an ESM2-only model,
   not just to dilute a working ESM2 signal). Selected via train.py's
   --architecture flag; kept alongside PocketEncoder rather than
   replacing it, so the two remain A/B-comparable on the same graphs.

3. `SiameseTripletModel` — wraps either encoder for triplet-loss
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
    of (neighbor node features, pairwise distance, sequence-adjacency
    flag), never of raw absolute coordinates directly, so the layer is
    translation/rotation invariant by construction. Angular/orientation
    information (which a true equivariant layer like EZSpecificity's would
    preserve through the network rather than collapsing to scalar
    distances at every layer) is the main capability this placeholder is
    missing — that's the specific gap the EZSpecificity swap-in is meant
    to close, most importantly for resolving Zn coordination geometry and
    carbonyl orientation.

    The sequence-adjacency flag (edge_attr[:, 1], see graph_construction.py)
    lets the message MLP learn to treat a loop's own backbone neighbor
    differently from an otherwise-unrelated spatial contact at a similar
    distance, rather than the two being indistinguishable.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 2, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim), nn.SiLU()
        )

    def forward(self, x, pos, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        dist = (pos_i - pos_j).norm(dim=-1, keepdim=True)
        is_seq_adjacent = edge_attr[:, 1:2]
        return self.msg_mlp(torch.cat([x_i, x_j, dist, is_seq_adjacent], dim=-1))

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
            x = conv(x, data.pos, data.edge_index, data.edge_attr)
            x = norm(x + residual)
            x = self.dropout(x)
        pooled = global_mean_pool(x, data.batch if hasattr(data, "batch") else
                                   torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        embedding = self.readout(pooled)
        return F.normalize(embedding, p=2, dim=-1)  # unit-norm, so Euclidean == scaled cosine


class StructuralEncoder(nn.Module):
    """
    The "residue pocket + loops" branch: the same distance/sequence-adjacency
    aware GNN as PocketEncoder, but restricted to the non-ESM2 columns
    (amino-acid identity + chemistry/geometry, see
    graph_construction.AA_IDENTITY_SLICE/STRUCTURAL_SLICE) so its embedding
    cannot leak sequence-homology signal through the ESM2 block. Pooled and
    L2-normalized independently of the ESM2 branch (see BranchedPocketEncoder)
    so neither branch's raw feature scale can dominate the other before
    fusion.
    """

    def __init__(
        self, in_dim: int, hidden_dim: int = 128, embed_dim: int = 64,
        n_layers: int = 4, dropout: float = 0.1,
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

    def forward(self, x_structural: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x_structural)
        for conv, norm in zip(self.layers, self.norms):
            residual = x
            x = conv(x, pos, edge_index, edge_attr)
            x = norm(x + residual)
            x = self.dropout(x)
        pooled = global_mean_pool(x, batch)
        return F.normalize(self.readout(pooled), p=2, dim=-1)


class ESM2ProjectionEncoder(nn.Module):
    """
    The "ESM2 residues" branch: a small per-residue MLP bottleneck (not a
    GNN -- ESM2 embeddings already encode long-range sequence context, the
    bottleneck's job is only dimensionality reduction before fusion) from
    esm_dim (1280) down to embed_dim, mean-pooled over the pocket's
    residues, L2-normalized independently of the structural branch.
    """

    def __init__(self, esm_dim: int = 1280, proj_dim: int = 128, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(esm_dim, proj_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(proj_dim, embed_dim),
        )

    def forward(self, x_esm: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        per_residue = self.proj(x_esm)
        pooled = global_mean_pool(per_residue, batch)
        return F.normalize(pooled, p=2, dim=-1)


class BranchedPocketEncoder(nn.Module):
    """
    Stage 3 design: separate structural and ESM2 branches (each pooled and
    L2-normalized on its own) fused by a small MLP, instead of the flat
    PocketEncoder's single concatenated 1320-dim input -- concatenating a
    17-dim chemistry block with a 1280-dim ESM2 block let the first linear
    layer treat ESM2 as ~97% of the input by raw dimensionality; here
    neither branch can dominate the other purely by scale, since each is
    unit-normalized before fusion.

    esm_dropout_prob: during training only, independently zeros each
    graph's *projected* ESM2 embedding with this probability before fusion
    (not the raw ESM2 input -- cheaper, and the effect on the fusion layer
    is identical), forcing the fusion layer to not collapse onto
    ESM2-only shortcuts. Per the ablation-matrix finding that ESM2-only
    training beat the flat combined model on the smallest-positive-count
    panel (B1_B2_transfer), while flat combined matched ESM2-only on the
    largest (B3_transfer) -- i.e. structure alone is not the fix, but an
    over-reliant fusion is part of the problem.

    An auxiliary structure-only loss (see train.py's structural_aux_loss_weight)
    is trained against z_structure directly, using the same shared-weight
    structural encoder -- exposed via forward(..., return_components=True).
    """

    def __init__(
        self,
        structural_dim: int,
        esm_dim: int = 1280,
        structural_hidden_dim: int = 128,
        structure_embed_dim: int = 64,
        esm_proj_dim: int = 128,
        esm_embed_dim: int = 64,
        fusion_hidden_dim: int = 128,
        embed_dim: int = 128,
        esm_dropout_prob: float = 0.4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.structural_dim = structural_dim
        self.structural_encoder = StructuralEncoder(
            structural_dim, structural_hidden_dim, structure_embed_dim, n_layers, dropout,
        )
        self.esm_encoder = ESM2ProjectionEncoder(esm_dim, esm_proj_dim, esm_embed_dim, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(structure_embed_dim + esm_embed_dim, fusion_hidden_dim), nn.SiLU(),
            nn.Linear(fusion_hidden_dim, embed_dim),
        )
        self.esm_dropout_prob = esm_dropout_prob

    def forward(self, data: Data, return_components: bool = False):
        batch = data.batch if hasattr(data, "batch") else torch.zeros(
            data.x.size(0), dtype=torch.long, device=data.x.device
        )
        x_structural = data.x[:, :self.structural_dim]
        x_esm = data.x[:, self.structural_dim:]

        z_structure = self.structural_encoder(x_structural, data.pos, data.edge_index, data.edge_attr, batch)
        z_sequence = self.esm_encoder(x_esm, batch)

        fusion_input = z_sequence
        if self.training and self.esm_dropout_prob > 0:
            keep = (torch.rand(z_sequence.size(0), device=z_sequence.device) >= self.esm_dropout_prob)
            fusion_input = z_sequence * keep.float().unsqueeze(-1)

        fused = self.fusion(torch.cat([z_structure, fusion_input], dim=-1))
        fused = F.normalize(fused, p=2, dim=-1)
        if return_components:
            return fused, z_structure
        return fused


class SiameseTripletModel(nn.Module):
    """Thin wrapper: shared-weight encoder applied to anchor/positive/negative."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, anchor: Data, positive: Data, negative: Data, return_components: bool = False):
        if return_components:
            fa, sa = self.encoder(anchor, return_components=True)
            fp, sp = self.encoder(positive, return_components=True)
            fn, sn = self.encoder(negative, return_components=True)
            return (fa, fp, fn), (sa, sp, sn)
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
