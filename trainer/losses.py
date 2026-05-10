from __future__ import annotations

import torch
import torch.nn as nn

from .datasets.feature_dataset import build_time_bin_cutpoints, discretize_survival_times


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
    """Compatibility wrapper for survival time-bin cutpoints.

    The caller controls the label scope. In normal training, FeatureDataset
    passes the whole fold before train/val split views are created.
    """
    return build_time_bin_cutpoints(labels, n_bins)


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
