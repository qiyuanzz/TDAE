from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
except Exception:  # pragma: no cover - optional dependency
    GATConv = None


class FeaturePropagator(nn.Module):
    """Binary oracle propagator: fill skipped patches by weighted neighbor interpolation."""

    def __init__(self, d_light: int, d_full: int, neighbor_size: int = 3, use_residual: bool = False, temperature: float = 0.1) -> None:
        super().__init__()
        if neighbor_size % 2 == 0 or neighbor_size < 3:
            raise ValueError('neighbor_size must be an odd integer >= 3.')
        self.neighbor_size = neighbor_size
        self.use_residual = use_residual
        self.temperature = temperature
        if use_residual:
            self.residual_mlp = nn.Sequential(nn.Linear(d_light + d_full, d_full), nn.ReLU(inplace=True), nn.Linear(d_full, d_full))

    @staticmethod
    def _grid_lookup(grid_indices: torch.Tensor) -> dict[tuple[int, int], int]:
        return {(int(rc[0]), int(rc[1])): idx for idx, rc in enumerate(grid_indices.detach().cpu().tolist())}

    def _neighbor_indices(self, idx: int, grid_indices: torch.Tensor, encoded_hard: torch.Tensor, lookup: dict[tuple[int, int], int]) -> list[int]:
        row, col = [int(v) for v in grid_indices[idx].detach().cpu().tolist()]
        start_radius = self.neighbor_size // 2
        for radius in range(start_radius, start_radius + 3):
            neighbors: list[int] = []
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    j = lookup.get((row + dr, col + dc))
                    if j is not None and bool(encoded_hard[j]):
                        neighbors.append(j)
            if neighbors:
                return neighbors
        return []

    def _global_nearest(self, idx: int, light_feats: torch.Tensor, encoded_hard: torch.Tensor) -> int:
        encoded = torch.where(encoded_hard)[0]
        if encoded.numel() == 0:
            return idx
        sims = F.cosine_similarity(light_feats[idx : idx + 1], light_feats[encoded], dim=-1)
        return int(encoded[sims.argmax()].item())

    def forward(self, full_feats: torch.Tensor, light_feats: torch.Tensor, grid_indices: torch.Tensor, gate_mask: torch.Tensor) -> torch.Tensor:
        device = full_feats.device
        full_feats = full_feats.float()
        light_feats = light_feats.float()
        grid_indices = grid_indices.long().to(device)
        soft_gate = gate_mask.float().to(device).clamp(0.0, 1.0)
        encoded_hard = soft_gate >= 0.5
        if not bool(encoded_hard.any()):
            encoded_hard[soft_gate.argmax()] = True
        lookup = self._grid_lookup(grid_indices)
        interpolated = full_feats.clone()
        skipped = torch.where(~encoded_hard)[0]
        for idx_tensor in skipped:
            idx = int(idx_tensor.item())
            neighbors = self._neighbor_indices(idx, grid_indices, encoded_hard, lookup)
            if not neighbors:
                interp = full_feats[self._global_nearest(idx, light_feats, encoded_hard)]
            else:
                neighbor_idx = torch.tensor(neighbors, device=device, dtype=torch.long)
                sims = F.cosine_similarity(light_feats[idx : idx + 1], light_feats[neighbor_idx], dim=-1)
                weights = F.softmax(sims / self.temperature, dim=0)
                interp = (weights.unsqueeze(1) * full_feats[neighbor_idx]).sum(dim=0)
            if self.use_residual:
                interp = interp + self.residual_mlp(torch.cat([light_feats[idx], interp], dim=0))
            interpolated[idx] = interp
        return soft_gate.unsqueeze(1) * full_feats + (1.0 - soft_gate.unsqueeze(1)) * interpolated


class KNNGraphAttentionLayer(nn.Module):
    """Small dependency-free GAT layer over precomputed kNN indices."""

    def __init__(self, d_hidden: int, n_heads: int = 4, num_levels: int = 4) -> None:
        super().__init__()
        if d_hidden % n_heads != 0:
            raise ValueError('d_hidden must be divisible by n_heads.')
        self.n_heads = n_heads
        self.head_dim = d_hidden // n_heads
        self.q = nn.Linear(d_hidden, d_hidden)
        self.k = nn.Linear(d_hidden, d_hidden)
        self.v = nn.Linear(d_hidden, d_hidden)
        self.out = nn.Linear(d_hidden, d_hidden)
        self.level_bias = nn.Embedding(num_levels, n_heads)

    def forward(self, h: torch.Tensor, neighbor_idx: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        if neighbor_idx.numel() == 0:
            return h
        n, k = neighbor_idx.shape
        q = self.q(h).view(n, self.n_heads, self.head_dim)
        key = self.k(h)[neighbor_idx].view(n, k, self.n_heads, self.head_dim)
        val = self.v(h)[neighbor_idx].view(n, k, self.n_heads, self.head_dim)
        scores = (q.unsqueeze(1) * key).sum(dim=-1) / math.sqrt(self.head_dim)
        level_diff = (levels.unsqueeze(1) - levels[neighbor_idx]).abs().clamp(max=3)
        scores = scores + self.level_bias(level_diff).to(scores.dtype)
        attn = F.softmax(scores, dim=1)
        msg = (attn.unsqueeze(-1) * val).sum(dim=1).reshape(n, self.n_heads * self.head_dim)
        return self.out(msg)


class GraphFeaturePropagator(nn.Module):
    """Level-aware graph propagation for L0/L1/L2/L3 patch features."""

    def __init__(
        self,
        d_light: int,
        d_medium: int,
        d_full: int,
        d_hidden: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        k_neighbors: int = 8,
        num_levels: int = 4,
        use_pyg: bool | None = None,
    ) -> None:
        super().__init__()
        self.k_neighbors = int(k_neighbors)
        self.num_levels = int(num_levels)
        if use_pyg is True and GATConv is None:
            raise RuntimeError('use_pyg=True requires torch-geometric to be installed.')
        self.use_pyg = (GATConv is not None) if use_pyg is None else bool(use_pyg)
        self.level_projections = nn.ModuleList([
            nn.Linear(d_light, d_hidden),
            nn.Linear(d_light, d_hidden),
            nn.Linear(d_medium, d_hidden),
            nn.Linear(d_full, d_hidden),
        ])
        if self.use_pyg:
            self.gat_layers = nn.ModuleList([
                GATConv(d_hidden, d_hidden // n_heads, heads=n_heads, concat=True, dropout=0.1, add_self_loops=True)
                for _ in range(n_layers)
            ])
        else:
            self.gat_layers = nn.ModuleList([KNNGraphAttentionLayer(d_hidden, n_heads=n_heads, num_levels=num_levels) for _ in range(n_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_layers)])
        self.output_proj = nn.Linear(d_hidden, d_full)

    def build_knn_graph(self, grid_indices: torch.Tensor) -> torch.Tensor:
        n = grid_indices.shape[0]
        if n <= 1 or self.k_neighbors <= 0:
            return torch.empty(n, 0, device=grid_indices.device, dtype=torch.long)
        k = min(self.k_neighbors + 1, n)
        dists = torch.cdist(grid_indices.float(), grid_indices.float())
        idx = dists.topk(k, largest=False).indices
        return idx[:, 1:]

    def build_edge_index(self, grid_indices: torch.Tensor) -> torch.Tensor:
        neighbor_idx = self.build_knn_graph(grid_indices)
        if neighbor_idx.numel() == 0:
            return torch.empty(2, 0, device=grid_indices.device, dtype=torch.long)
        src = torch.arange(neighbor_idx.shape[0], device=grid_indices.device, dtype=torch.long)
        src = src.unsqueeze(1).expand_as(neighbor_idx).reshape(-1)
        dst = neighbor_idx.reshape(-1)
        return torch.stack([src, dst], dim=0)

    def forward(self, multi_level_feats: dict[int, torch.Tensor], gate_soft: torch.Tensor, grid_indices: torch.Tensor) -> torch.Tensor:
        n = grid_indices.shape[0]
        h = torch.zeros(n, self.level_projections[0].out_features, device=grid_indices.device, dtype=multi_level_feats[1].dtype)
        for level in range(self.num_levels):
            h = h + gate_soft[:, level : level + 1] * self.level_projections[level](multi_level_feats[level].float())
        levels = gate_soft.argmax(dim=-1)
        if self.use_pyg:
            edge_index = self.build_edge_index(grid_indices.long())
            for layer, norm in zip(self.gat_layers, self.layer_norms):
                h = norm(F.elu(layer(h, edge_index)) + h)
        else:
            neighbor_idx = self.build_knn_graph(grid_indices.long())
            for layer, norm in zip(self.gat_layers, self.layer_norms):
                h = norm(F.elu(h + layer(h, neighbor_idx, levels)))
        return self.output_proj(h)
