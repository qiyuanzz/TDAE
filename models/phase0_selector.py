from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Phase0Selection:
    gate_soft: torch.Tensor
    gate_hard: torch.Tensor
    avg_cost: torch.Tensor
    level_dist: torch.Tensor
    difficulty: torch.Tensor
    risk: torch.Tensor


class BudgetedPhase0Selector(nn.Module):
    """Deterministic light-feature selector for budgeted L0/L1/L2/L3 allocation."""

    def __init__(
        self,
        c_target: float = 0.30,
        flops_cost: tuple[float, float, float, float] | list[float] | None = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        k_recon: int = 8,
        temperature: float = 0.2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        costs = torch.tensor(flops_cost if flops_cost is not None else (0.125, 0.125, 0.5, 1.0), dtype=torch.float32)
        if costs.numel() != 4:
            raise ValueError(f'flops_cost must contain 4 values, got {costs.tolist()}')
        if not bool(torch.all(costs > 0)):
            raise ValueError(f'flops_cost values must be positive, got {costs.tolist()}')
        self.register_buffer('flops_cost', costs, persistent=False)
        self.c_target = float(c_target)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.k_recon = int(k_recon)
        self.temperature = float(temperature)
        self.eps = float(eps)

    def _normalize01(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values
        v_min = values.min()
        v_max = values.max()
        denom = v_max - v_min
        if float(denom.detach().cpu()) <= self.eps:
            return torch.zeros_like(values)
        return (values - v_min) / denom.clamp_min(self.eps)

    def _spatial_knn(self, grid_indices: torch.Tensor) -> torch.Tensor:
        n = int(grid_indices.shape[0])
        if n <= 1 or self.k_recon <= 0:
            return torch.empty(n, 0, device=grid_indices.device, dtype=torch.long)
        k = min(self.k_recon + 1, n)
        dists = torch.cdist(grid_indices.float(), grid_indices.float())
        return dists.topk(k, largest=False).indices[:, 1:]

    def _difficulty_terms(self, light_feats: torch.Tensor, grid_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = int(light_feats.shape[0])
        if n == 0:
            empty = light_feats.new_zeros((0,))
            return empty, empty, empty

        light_norm = F.normalize(light_feats.float(), dim=1, eps=self.eps)
        centroid = F.normalize(light_norm.mean(dim=0, keepdim=True), dim=1, eps=self.eps)
        d_feat = self._normalize01(1.0 - F.cosine_similarity(light_norm, centroid, dim=1))

        if n <= 1:
            zero = light_feats.new_zeros((n,))
            return d_feat, zero, zero

        spatial_dists = torch.cdist(grid_indices.float(), grid_indices.float())
        nearest_dist = spatial_dists.topk(2, largest=False).values[:, 1]
        d_spatial = self._normalize01(nearest_dist)

        neighbor_idx = self._spatial_knn(grid_indices)
        if neighbor_idx.numel() == 0:
            d_recon = light_feats.new_zeros((n,))
        else:
            neighbor_feats = light_norm[neighbor_idx]
            sims = (light_norm.unsqueeze(1) * neighbor_feats).sum(dim=-1)
            weights = F.softmax(sims / max(self.temperature, self.eps), dim=1)
            recon = (weights.unsqueeze(-1) * neighbor_feats).sum(dim=1)
            d_recon = self._normalize01(1.0 - F.cosine_similarity(light_norm, recon, dim=1))
        return d_feat, d_spatial, d_recon

    def compute_risk(self, light_feats: torch.Tensor, grid_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d_feat, d_spatial, d_recon = self._difficulty_terms(light_feats, grid_indices)
        denom = max(self.alpha + self.beta + self.gamma, self.eps)
        difficulty = self._normalize01((self.alpha * d_feat + self.beta * d_spatial + self.gamma * d_recon) / denom)

        # L0 is good when neighbors can reconstruct the patch; L1 is good when the light
        # feature itself is trustworthy. L2/L3 progressively reduce proxy risk.
        r0 = 0.70 * d_recon + 0.20 * d_spatial + 0.10 * d_feat
        r1 = 0.60 * d_feat + 0.25 * d_recon + 0.15 * difficulty
        r2 = 0.35 * difficulty + 0.10 * d_recon
        r3 = torch.zeros_like(difficulty)
        return torch.stack([r0, r1, r2, r3], dim=1), difficulty

    def _initial_low_cost_levels(self, risk: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        min_cost = costs.min()
        cheapest = torch.where(torch.isclose(costs, min_cost))[0]
        local = risk[:, cheapest].argmin(dim=1)
        return cheapest[local]

    def _fill_budget(self, risk: torch.Tensor, initial_levels: torch.Tensor, costs: torch.Tensor, c_target: float) -> torch.Tensor:
        n = int(initial_levels.numel())
        if n == 0:
            return initial_levels
        target = min(max(float(c_target), float(costs.min())), float(costs.max()))
        budget_total = target * n
        current = initial_levels.clone()
        current_cost = costs[current].sum()
        if float(current_cost.detach().cpu()) >= budget_total - self.eps:
            return current

        candidates: list[tuple[float, float, float, int, int]] = []
        for patch_idx in range(n):
            src_level = int(current[patch_idx].item())
            src_cost = float(costs[src_level].item())
            src_risk = float(risk[patch_idx, src_level].detach().cpu())
            for dst_level in range(4):
                dst_cost = float(costs[dst_level].item())
                if dst_cost <= src_cost + self.eps:
                    continue
                benefit = src_risk - float(risk[patch_idx, dst_level].detach().cpu())
                if benefit <= self.eps:
                    continue
                delta = dst_cost - src_cost
                efficiency = benefit / delta
                candidates.append((-efficiency, -benefit, delta, patch_idx, dst_level))
        candidates.sort()

        used: set[int] = set()
        for _, _, delta, patch_idx, dst_level in candidates:
            if patch_idx in used:
                continue
            if float(current_cost.detach().cpu()) + delta <= budget_total + self.eps:
                current[patch_idx] = int(dst_level)
                current_cost = current_cost + costs[dst_level] - costs[initial_levels[patch_idx]]
                used.add(patch_idx)
        return current

    def forward(self, light_feats: torch.Tensor, grid_indices: torch.Tensor, c_target: float | None = None) -> Phase0Selection:
        device = light_feats.device
        costs = self.flops_cost.to(device=device, dtype=light_feats.dtype)
        risk, difficulty = self.compute_risk(light_feats.float(), grid_indices.long().to(device))
        initial = self._initial_low_cost_levels(risk, costs)
        gate_hard = self._fill_budget(risk, initial, costs, self.c_target if c_target is None else float(c_target))
        gate_soft = F.one_hot(gate_hard, 4).to(dtype=light_feats.dtype, device=device)
        avg_cost = (gate_soft * costs.view(1, 4)).sum(dim=1).mean()
        return Phase0Selection(
            gate_soft=gate_soft,
            gate_hard=gate_hard,
            avg_cost=avg_cost,
            level_dist=gate_soft.float().mean(dim=0),
            difficulty=difficulty,
            risk=risk,
        )
