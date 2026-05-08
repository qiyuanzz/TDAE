from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEncoder(nn.Module):
    """Encode integer (row, col) patch grid coordinates."""

    def __init__(self, d_out: int = 64) -> None:
        super().__init__()
        if d_out < 4:
            raise ValueError('d_out must be at least 4.')
        self.d_out = int(d_out)
        self.linear = nn.Linear(self.d_out, self.d_out)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.float()
        n = coords.shape[0]
        pe = torch.zeros(n, self.d_out, device=coords.device, dtype=coords.dtype)
        quarter = max(self.d_out // 4, 1)
        div = torch.exp(torch.arange(quarter, device=coords.device, dtype=coords.dtype) * (-math.log(10000.0) / max(quarter, 1)))
        pe[:, 0:quarter] = torch.sin(coords[:, 0:1] * div)
        pe[:, quarter:2 * quarter] = torch.cos(coords[:, 0:1] * div)
        pe[:, 2 * quarter:3 * quarter] = torch.sin(coords[:, 1:2] * div)
        pe[:, 3 * quarter:4 * quarter] = torch.cos(coords[:, 1:2] * div)
        return self.linear(pe)


class GatingNetwork(nn.Module):
    """Multi-granularity gate over L0 skip, L1 light, L2 medium, L3 full."""

    NUM_LEVELS = 4
    DEFAULT_FLOPS_COST = (0.125, 0.125, 0.5, 1.0)

    def __init__(
        self,
        d_light: int,
        d_pos: int = 64,
        hidden: int = 256,
        flops_cost: tuple[float, float, float, float] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos_encoder = SinusoidalPosEncoder(d_out=d_pos)
        self.mlp = nn.Sequential(
            nn.Linear(d_light + d_pos, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.NUM_LEVELS),
        )
        cost = flops_cost or self.DEFAULT_FLOPS_COST
        self.register_buffer('cost_vec', torch.tensor(cost, dtype=torch.float32), persistent=False)

    def forward(
        self,
        light_feats: torch.Tensor,
        grid_indices: torch.Tensor,
        tau: float = 1.0,
        hard: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = self.pos_encoder(grid_indices.float())
        logits = self.mlp(torch.cat([light_feats.float(), pos], dim=-1))
        if hard is None:
            hard = not self.training
        if hard:
            gate_hard = logits.argmax(dim=-1)
            gate_soft = F.one_hot(gate_hard, self.NUM_LEVELS).to(logits.dtype)
        else:
            gate_soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            gate_hard = logits.argmax(dim=-1)
        avg_cost = (gate_soft * self.cost_vec.to(logits.device)).sum(dim=-1).mean()
        return gate_soft, gate_hard, avg_cost
