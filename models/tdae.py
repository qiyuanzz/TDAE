from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_propagator import GraphFeaturePropagator
from .gating_network import GatingNetwork
from .mil_aggregators import build_aggregator
from .phase0_selector import BudgetedPhase0Selector


class TDAE(nn.Module):
    """Adaptive 4-level WSI encoding with graph propagation."""

    def __init__(
        self,
        d_light: int,
        d_full: int,
        n_classes: int,
        d_medium: int | None = None,
        aggregator_type: str = 'abmil',
        gating_hidden: int = 256,
        pos_encoding_dim: int = 64,
        d_gat_hidden: int = 256,
        gat_heads: int = 4,
        gat_layers: int = 2,
        k_neighbors: int = 8,
        flops_cost: tuple[float, float, float, float] | list[float] | None = None,
        use_pyg_gat: bool | None = None,
        selection_method: str = 'trainable_gating',
        c_target: float = 0.30,
        phase0_alpha: float = 1.0,
        phase0_beta: float = 1.0,
        phase0_gamma: float = 1.0,
        phase0_k_recon: int = 8,
        **_: object,
    ) -> None:
        super().__init__()
        d_medium = d_full if d_medium is None else d_medium
        self.selection_method = str(selection_method)
        self.c_target = float(c_target)
        costs = tuple(flops_cost) if flops_cost is not None else (0.125, 0.125, 0.5, 1.0)
        self.register_buffer('flops_cost', torch.tensor(costs, dtype=torch.float32), persistent=False)
        self.gating: GatingNetwork | None = None
        self.phase0_selector: BudgetedPhase0Selector | None = None
        if self.selection_method == 'trainable_gating':
            self.gating = GatingNetwork(d_light=d_light, d_pos=pos_encoding_dim, hidden=gating_hidden, flops_cost=costs)
        elif self.selection_method == 'tdae_auto':
            self.phase0_selector = BudgetedPhase0Selector(
                c_target=self.c_target,
                flops_cost=costs,
                alpha=phase0_alpha,
                beta=phase0_beta,
                gamma=phase0_gamma,
                k_recon=phase0_k_recon,
            )
        elif self.selection_method == 'full_upper_bound':
            pass
        else:
            raise ValueError(f'Unsupported selection_method: {selection_method}')
        self.propagator = GraphFeaturePropagator(
            d_light=d_light,
            d_medium=d_medium,
            d_full=d_full,
            d_hidden=d_gat_hidden,
            n_heads=gat_heads,
            n_layers=gat_layers,
            k_neighbors=k_neighbors,
            use_pyg=use_pyg_gat,
        )
        self.aggregator = build_aggregator(aggregator_type, d_in=d_full, n_classes=n_classes)

    def _parse_inputs(self, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(args) == 4:
            light_feats, medium_feats, full_feats, grid_indices = args
            return light_feats, medium_feats, full_feats, grid_indices
        if len(args) == 3:
            full_feats, light_feats, grid_indices = args
            return light_feats, full_feats, full_feats, grid_indices
        raise TypeError('TDAE.forward expects (light, medium, full, coords) or legacy (full, light, coords).')

    def forward(
        self,
        *args: torch.Tensor,
        tau: float = 1.0,
        hard_gate: bool | None = None,
        force_encode_all: bool = False,
        c_target: float | None = None,
    ) -> dict[str, torch.Tensor]:
        light_feats, medium_feats, full_feats, grid_indices = self._parse_inputs(args)
        device = full_feats.device
        light_feats = light_feats.to(device).float()
        medium_feats = medium_feats.to(device).float()
        full_feats = full_feats.to(device).float()
        grid_indices = grid_indices.to(device).long()

        if force_encode_all or self.selection_method == 'full_upper_bound':
            gate_hard = torch.full((light_feats.shape[0],), 3, device=device, dtype=torch.long)
            gate_soft = F.one_hot(gate_hard, 4).to(full_feats.dtype)
            avg_cost = self.flops_cost.to(device=device, dtype=full_feats.dtype)[3]
        elif self.selection_method == 'tdae_auto':
            if self.phase0_selector is None:
                raise RuntimeError('tdae_auto requires phase0_selector.')
            selection = self.phase0_selector(light_feats, grid_indices, c_target=self.c_target if c_target is None else c_target)
            gate_soft = selection.gate_soft.to(device=device, dtype=full_feats.dtype)
            gate_hard = selection.gate_hard.to(device=device)
            avg_cost = selection.avg_cost.to(device=device, dtype=full_feats.dtype)
        else:
            if self.gating is None:
                raise RuntimeError('trainable_gating requires gating network.')
            gate_soft, gate_hard, avg_cost = self.gating(light_feats, grid_indices, tau=tau, hard=hard_gate)

        multi_level_feats = {0: light_feats, 1: light_feats, 2: medium_feats, 3: full_feats}
        propagated = self.propagator(multi_level_feats, gate_soft, grid_indices)
        logits, attn = self.aggregator(propagated)
        level_dist = gate_soft.mean(dim=0)
        high_compute = gate_soft[:, 2:].sum(dim=-1)
        return {
            'logits': logits,
            'gate_soft': gate_soft,
            'gate_hard': gate_hard,
            'avg_cost': avg_cost,
            'level_dist': level_dist,
            'gate_probs': high_compute,
            'gate_mask': (gate_hard >= 2).to(full_feats.dtype),
            'attn_weights': attn,
            'propagated_feats': propagated,
            'encode_rate': high_compute.mean(),
            'grid_indices': grid_indices,
        }
