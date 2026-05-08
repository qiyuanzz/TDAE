from __future__ import annotations

from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def nll_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    alpha: float | None = 0.0,
    eps: float = 1e-7,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Discrete-time survival negative log likelihood.

    ``logits`` are hazard logits with shape ``[batch, n_bins]``. ``y`` is the
    time-bin index, and ``c`` is the censoring indicator where 1 means censored.
    """
    if logits.ndim != 2:
        raise ValueError(f'nll_loss expects logits with shape [batch, n_bins], got {tuple(logits.shape)}')
    y = y.to(logits.device).long().view(-1, 1)
    c = c.to(logits.device).long().view(-1, 1)
    if y.shape[0] != logits.shape[0] or c.shape[0] != logits.shape[0]:
        raise ValueError('logits, y, and c must have the same batch size.')
    if bool((y < 0).any()) or bool((y >= logits.shape[1]).any()):
        raise ValueError(f'time-bin labels must be in [0, {logits.shape[1] - 1}].')

    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    survival_padded = torch.cat([torch.ones_like(c, dtype=hazards.dtype), survival], dim=1)

    s_prev = torch.gather(survival_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(survival_padded, dim=1, index=y + 1).clamp(min=eps)

    uncensored_loss = -(1 - c).float() * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = -c.float() * torch.log(s_this)
    neg_l = censored_loss + uncensored_loss
    loss = neg_l if alpha is None else (1 - float(alpha)) * neg_l + float(alpha) * uncensored_loss

    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    raise ValueError(f'Bad input for reduction: {reduction}')


class NLLSurvLoss(nn.Module):
    """Discrete-time survival NLL for hazard logits."""

    def __init__(self, alpha: float | None = 0.0, eps: float = 1e-7, reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, times: torch.Tensor, censorships: torch.Tensor, **kwargs) -> torch.Tensor:
        return nll_loss(logits=logits, y=times, c=censorships, alpha=self.alpha, eps=self.eps, reduction=self.reduction)


def build_survival_time_bin_cutpoints(labels: list[dict[str, torch.Tensor]], n_bins: int) -> torch.Tensor:
    """Fit MCAT/SurvPath-style discrete survival bins with ``pd.qcut``.

    The labels passed here should represent the full fold cohort, meaning train
    and validation cases before the KFold split is applied to the current fold.
    Following MCAT/SurvPath, quantile boundaries are fit on uncensored/event
    cases and then used for all cases.
    """
    if n_bins < 2:
        raise ValueError(f'n_bins must be >= 2 for nll_surv, got {n_bins}')
    if not labels:
        raise ValueError('Cannot build survival time bins from an empty label list.')
    times = torch.stack([label['time'].detach().float().cpu().view(()) for label in labels])
    events = torch.stack([label['event'].detach().float().cpu().view(()) for label in labels])
    source_times = times[events > 0]
    if source_times.numel() < n_bins:
        source_times = times
    if source_times.numel() < n_bins:
        raise ValueError(f'Need at least {n_bins} survival times to build {n_bins} qcut bins, got {source_times.numel()}.')
    try:
        _, bin_edges = pd.qcut(source_times.numpy(), q=n_bins, labels=False, retbins=True)
        cutpoints = torch.tensor(bin_edges[1:-1], dtype=torch.float32)
    except ValueError:
        quantiles = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float32)[1:-1]
        cutpoints = torch.quantile(source_times.float(), quantiles).float()
    if cutpoints.numel() != n_bins - 1:
        raise ValueError(f'Expected {n_bins - 1} cutpoints for {n_bins} bins, got {cutpoints.numel()}.')
    return cutpoints.float()


def discretize_survival_times(times: torch.Tensor, cutpoints: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Map continuous survival times to fixed discrete time-bin indices."""
    if n_bins < 2:
        raise ValueError(f'n_bins must be >= 2 for nll_surv, got {n_bins}')
    if cutpoints.numel() != n_bins - 1:
        raise ValueError(f'Expected {n_bins - 1} cutpoints for {n_bins} bins, got {cutpoints.numel()}')
    flat_times = times.float().view(-1)
    cutpoints = cutpoints.to(device=flat_times.device, dtype=flat_times.dtype)
    return torch.bucketize(flat_times, cutpoints, right=False).clamp(0, n_bins - 1).long()


def survival_logits_to_risk(logits: torch.Tensor) -> torch.Tensor:
    """Convert discrete hazard logits to a scalar risk for C-index evaluation."""
    logits = logits.float()
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    return -survival.sum(dim=1)


def partial_ll_loss(lrisks: torch.Tensor, survival_times: torch.Tensor, event_indicators: torch.Tensor) -> torch.Tensor:
    """Cox partial likelihood loss.

    ``event_indicators`` uses 1 for observed events and 0 for censored cases.
    """
    event_indicators = event_indicators.to(lrisks.device).float().reshape(-1)
    survival_times = survival_times.to(lrisks.device).float().reshape(-1)
    lrisks = lrisks.reshape(-1)
    num_uncensored = torch.sum(event_indicators, 0)
    if num_uncensored.item() == 0:
        return torch.sum(lrisks) * 0
    if lrisks.shape[0] != survival_times.shape[0] or lrisks.shape[0] != event_indicators.shape[0]:
        raise ValueError('lrisks, survival_times, and event_indicators must have the same batch size.')

    sindex = torch.argsort(-survival_times)
    survival_times = survival_times[sindex]
    event_indicators = event_indicators[sindex]
    lrisks = lrisks[sindex]

    log_risk_stable = torch.logcumsumexp(lrisks, 0)
    likelihood = lrisks - log_risk_stable
    uncensored_likelihood = likelihood * event_indicators
    log_likelihood = -torch.sum(uncensored_likelihood)
    return log_likelihood / num_uncensored


class CoxLoss(nn.Module):
    """Cox partial likelihood loss with censorship input.

    ``censorships`` uses 1 for censored cases and 0 for observed events.
    """

    def forward(self, logits: torch.Tensor, times: torch.Tensor, censorships: torch.Tensor, **kwargs) -> torch.Tensor:
        return partial_ll_loss(lrisks=logits, survival_times=times, event_indicators=(1 - censorships).float())


def cox_partial_likelihood(logits: torch.Tensor, target: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compatibility wrapper for code paths that store event indicators directly."""
    return partial_ll_loss(logits, target['time'], target['event'])


class TDAELoss(nn.Module):
    def __init__(self, lambda_budget: float = 1.0, mu_diversity: float = 0.1, task: str = 'classification') -> None:
        super().__init__()
        self.lambda_budget = lambda_budget
        self.mu_diversity = mu_diversity
        self.task = task

    def forward(self, outputs: dict[str, Any], targets: torch.Tensor | dict[str, torch.Tensor], c_target: float = 0.3, r_target: float | None = None) -> dict[str, torch.Tensor]:
        if r_target is not None:
            c_target = r_target
        if self.task == 'classification':
            logits = outputs['logits'].unsqueeze(0)
            target_tensor = targets if isinstance(targets, torch.Tensor) else targets['label']
            l_mil = F.cross_entropy(logits, target_tensor.long().view(1).to(logits.device))
        elif self.task == 'survival':
            if not isinstance(targets, dict):
                raise ValueError('survival task requires targets with event/time tensors.')
            l_mil = cox_partial_likelihood(outputs['logits'], targets)
        else:
            raise ValueError(f'Unsupported task: {self.task}')

        avg_cost = outputs['avg_cost'].float() if 'avg_cost' in outputs else outputs.get('gate_probs', torch.tensor(0.0, device=outputs['logits'].device)).float().mean()
        l_budget = F.relu(avg_cost - float(c_target))
        l_diversity = self._diversity_loss(outputs.get('gate_soft'), outputs.get('grid_indices'))
        total = l_mil + self.lambda_budget * l_budget + self.mu_diversity * l_diversity
        result = {
            'total': total,
            'l_mil': l_mil.detach(),
            'l_budget': l_budget.detach(),
            'l_diversity': l_diversity.detach(),
            'avg_cost': avg_cost.detach(),
            'encode_rate': outputs.get('gate_probs', avg_cost).float().mean().detach(),
        }
        if 'level_dist' in outputs:
            for idx, value in enumerate(outputs['level_dist'].detach()):
                result[f'level_{idx}'] = value
        return result

    @staticmethod
    def _diversity_loss(gate_soft: torch.Tensor | None, grid_indices: torch.Tensor | None) -> torch.Tensor:
        if gate_soft is None:
            device = grid_indices.device if grid_indices is not None else 'cpu'
            return torch.tensor(0.0, device=device)
        device = gate_soft.device
        if grid_indices is None:
            return torch.tensor(0.0, device=device)
        high_score = gate_soft[:, 2:].sum(dim=-1)
        selected = grid_indices.to(device).float()[high_score > 0.5]
        if selected.shape[0] < 2:
            return torch.tensor(0.0, device=device)
        dists = torch.cdist(selected, selected)
        dists.fill_diagonal_(float('inf'))
        return -dists.min(dim=1).values.mean()
