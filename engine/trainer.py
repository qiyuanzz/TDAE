from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .evaluator import evaluate_model
from .losses import (
    NLLSurvLoss,
    TDAELoss,
    build_survival_time_bin_cutpoints,
    cox_partial_likelihood,
    discretize_survival_times,
)


def slide_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError('TDAE currently expects DataLoader batch_size=1.')
    return batch[0]


def case_index_groups(dataset) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = OrderedDict()
    for idx, case_id in enumerate(dataset.slides['case_submitter_id'].astype(str).tolist()):
        groups.setdefault(case_id, []).append(idx)
    return groups


@dataclass
class TrainerConfig:
    method: str = 'trainable_gating'
    warmup_epochs: int = 5
    joint_epochs: int = 20
    lr_gating: float = 5e-4
    lr_aggregator: float = 2e-4
    weight_decay: float = 1e-5
    lambda_budget: float = 1.0
    mu_diversity: float = 0.1
    tau_start: float = 1.0
    tau_end: float = 0.1
    c_start: float = 0.6
    c_end: float = 0.3
    c_target: float = 0.3
    eval_every: int = 1
    task: str = 'classification'
    survival_batch_size: int = 32
    bag_loss: str = 'nll_surv'
    alpha_surv: float = 0.0
    n_classes: int = 4
    device: str = 'cuda'


class TDAETrainer:
    def __init__(self, model: torch.nn.Module, train_dataset, val_dataset, config: TrainerConfig) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')
        self.model.to(self.device)
        self.trainable_gating = config.method == 'trainable_gating' and getattr(model, 'gating', None) is not None
        param_groups = []
        if self.trainable_gating:
            param_groups.append({'params': model.gating.parameters(), 'lr': config.lr_gating})
        param_groups.extend(
            [
                {'params': model.propagator.parameters(), 'lr': config.lr_gating},
                {'params': model.aggregator.parameters(), 'lr': config.lr_aggregator},
            ]
        )
        self.optimizer = torch.optim.Adam(param_groups, weight_decay=config.weight_decay)
        lambda_budget = config.lambda_budget if self.trainable_gating else 0.0
        mu_diversity = config.mu_diversity if self.trainable_gating else 0.0
        self.loss_fn = TDAELoss(lambda_budget, mu_diversity, config.task)
        self.survival_loss = NLLSurvLoss(alpha=config.alpha_surv) if config.task == 'survival' and config.bag_loss == 'nll_surv' else None
        self.time_bin_cutpoints = (
            self._build_time_bin_cutpoints()
            if self.survival_loss is not None
            else None
        )

    def _build_time_bin_cutpoints(self) -> torch.Tensor:
        """MCAT/SurvPath-style cutpoints: train fold + uncensored (event==1) only.

        Quantile cutpoints are fit on the current fold's training cases with
        observed events, then applied identically to train and val cases at
        loss time. Validation labels never participate in the cutpoint fit.
        """
        labels_by_case: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        dataset = self.train_dataset
        if hasattr(dataset, 'slides') and {'event', 'survival_days'}.issubset(dataset.slides.columns):
            rows = dataset.slides.drop_duplicates('case_submitter_id')
            for _, row in rows.iterrows():
                case_id = str(row['case_submitter_id'])
                if case_id not in labels_by_case:
                    labels_by_case[case_id] = {
                        'event': torch.tensor(float(row['event']), dtype=torch.float32),
                        'time': torch.tensor(float(row['survival_days']), dtype=torch.float32),
                    }
        else:
            groups = case_index_groups(dataset)
            for case_id, indices in groups.items():
                if indices and case_id not in labels_by_case:
                    labels_by_case[case_id] = dataset[indices[0]]['label']
        labels = list(labels_by_case.values())
        return build_survival_time_bin_cutpoints(labels, self.config.n_classes)

    def _uses_budget_losses(self) -> bool:
        return self.trainable_gating and (self.config.lambda_budget > 0 or self.config.mu_diversity > 0)

    def _freeze_gating(self, freeze: bool) -> None:
        if getattr(self.model, 'gating', None) is None:
            return
        for param in self.model.gating.parameters():
            param.requires_grad = not freeze

    def _schedule(self, epoch: int) -> tuple[float, float]:
        if not self.trainable_gating:
            return 1.0, float(self.config.c_target)
        if epoch < self.config.warmup_epochs:
            return 1.0, 1.0
        progress = (epoch - self.config.warmup_epochs) / max(self.config.joint_epochs - 1, 1)
        tau = self.config.tau_start - progress * (self.config.tau_start - self.config.tau_end)
        c_target = self.config.c_start - progress * (self.config.c_start - self.config.c_end)
        return float(tau), float(c_target)

    def _forward_sample(self, sample: dict[str, Any], tau: float, warmup: bool, c_target: float) -> dict[str, torch.Tensor]:
        return self.model(
            sample['light_feats'].to(self.device),
            sample['medium_feats'].to(self.device),
            sample['full_feats'].to(self.device),
            sample['grid_indices'].to(self.device),
            tau=tau,
            hard_gate=warmup,
            force_encode_all=warmup and self.trainable_gating,
            c_target=c_target,
        )

    def _classification_step(self, sample: dict[str, Any], tau: float, c_target: float, warmup: bool) -> dict[str, float]:
        label = sample['label'].to(self.device)
        outputs = self._forward_sample(sample, tau=tau, warmup=warmup, c_target=c_target)
        losses = self.loss_fn(outputs, label, c_target=c_target)
        self.optimizer.zero_grad(set_to_none=True)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {key: float(value.detach().cpu()) for key, value in losses.items()}

    def _survival_batch_loss(self, outputs: list[dict[str, torch.Tensor]], labels: list[dict[str, torch.Tensor]], c_target: float) -> dict[str, torch.Tensor]:
        times = torch.stack([label['time'].to(self.device).view(()) for label in labels])
        events = torch.stack([label['event'].to(self.device).view(()) for label in labels])
        if self.config.bag_loss == 'nll_surv':
            if self.survival_loss is None:
                raise RuntimeError('nll_surv requires survival_loss.')
            logits = torch.stack([out['logits'].flatten() for out in outputs], dim=0)
            if self.time_bin_cutpoints is None:
                raise RuntimeError('nll_surv requires train-fold time_bin_cutpoints.')
            time_bins = discretize_survival_times(times, self.time_bin_cutpoints.to(self.device), self.config.n_classes)
            if all('censorship' in label for label in labels):
                censorships = torch.stack([label['censorship'].to(self.device).view(()) for label in labels]).long()
            else:
                censorships = (1.0 - events.float()).long()
            l_mil = self.survival_loss(logits, time_bins, censorships)
        else:
            risks = torch.cat([out['logits'].flatten()[:1] for out in outputs], dim=0)
            l_mil = cox_partial_likelihood(risks, {'event': events, 'time': times})
        avg_cost = torch.stack([out['avg_cost'].float() for out in outputs]).mean()
        if self._uses_budget_losses():
            l_budget = F.relu(avg_cost - float(c_target))
            l_diversity = torch.stack([
                out.get('l_diversity', self.loss_fn._diversity_loss(out.get('gate_soft'), out.get('grid_indices')))
                for out in outputs
            ]).mean()
            total = l_mil + self.config.lambda_budget * l_budget + self.config.mu_diversity * l_diversity
        else:
            l_budget = avg_cost.new_zeros(())
            l_diversity = avg_cost.new_zeros(())
            total = l_mil
        level_dist = torch.stack([out['level_dist'] for out in outputs]).mean(dim=0).detach()
        result = {
            'total': total,
            'l_mil': l_mil.detach(),
            'l_budget': l_budget.detach(),
            'l_diversity': l_diversity.detach(),
            'avg_cost': avg_cost.detach(),
            'encode_rate': torch.stack([out['gate_probs'].float().mean() for out in outputs]).mean().detach(),
        }
        for idx, value in enumerate(level_dist):
            result[f'level_{idx}'] = value
        return result

    # Offset added to per-slide grid coordinates when concatenating multiple
    # slides of the same case into a single bag. Must be larger than any
    # plausible intra-slide grid extent so the kNN graph never connects
    # patches from different slides. WSIs at 224px @ 20x have grid extents
    # well below 1000, so 1e5 is safe.
    _CASE_SLIDE_GRID_OFFSET: int = 100_000

    def _forward_case(self, dataset, indices: list[int], tau: float, warmup: bool, c_target: float) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """MCAT/SurvPath-style case bag.

        Concatenate all slides of the same case into one bag and run a single
        forward pass. Per-slide grid coordinates are shifted by a large
        per-slide offset so the propagator's kNN graph cannot wire patches
        across slides.
        """
        if not indices:
            raise ValueError('Cannot train survival case with no slides.')
        light_chunks: list[torch.Tensor] = []
        medium_chunks: list[torch.Tensor] = []
        full_chunks: list[torch.Tensor] = []
        coord_chunks: list[torch.Tensor] = []
        label: dict[str, torch.Tensor] | None = None
        for slide_pos, idx in enumerate(indices):
            sample = dataset[idx]
            if label is None:
                label = sample['label']
            light_chunks.append(sample['light_feats'].to(self.device))
            medium_chunks.append(sample['medium_feats'].to(self.device))
            full_chunks.append(sample['full_feats'].to(self.device))
            coords = sample['grid_indices'].to(self.device).long().clone()
            coords = coords + slide_pos * self._CASE_SLIDE_GRID_OFFSET
            coord_chunks.append(coords)
        if label is None:
            raise ValueError('Cannot train survival case with no slides.')
        case_sample = {
            'light_feats': torch.cat(light_chunks, dim=0),
            'medium_feats': torch.cat(medium_chunks, dim=0),
            'full_feats': torch.cat(full_chunks, dim=0),
            'grid_indices': torch.cat(coord_chunks, dim=0),
        }
        output = self._forward_sample(case_sample, tau=tau, warmup=warmup, c_target=c_target)
        l_diversity = (
            self.loss_fn._diversity_loss(output.get('gate_soft'), output.get('grid_indices'))
            if self._uses_budget_losses()
            else output['avg_cost'].new_zeros(())
        )
        return {
            'logits': output['logits'].flatten(),
            'avg_cost': output['avg_cost'].float(),
            'level_dist': output['level_dist'],
            'gate_probs': output['gate_probs'].float().mean().view(1),
            'l_diversity': l_diversity,
        }, label

    def _survival_step(self, batch: list[dict[str, Any]], tau: float, c_target: float, warmup: bool) -> dict[str, float]:
        outputs = [self._forward_sample(sample, tau=tau, warmup=warmup, c_target=c_target) for sample in batch]
        labels = [sample['label'] for sample in batch]
        losses = self._survival_batch_loss(outputs, labels, c_target=c_target)
        self.optimizer.zero_grad(set_to_none=True)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {key: float(value.detach().cpu()) for key, value in losses.items()}

    def train(self, output_dir: str | Path) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_loader = None
        if self.config.task != 'survival':
            train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, collate_fn=slide_collate)
        history: list[dict[str, float]] = []
        total_epochs = (self.config.warmup_epochs + self.config.joint_epochs) if self.trainable_gating else self.config.joint_epochs
        survival_case_groups = case_index_groups(self.train_dataset) if self.config.task == 'survival' else {}
        survival_case_ids = list(survival_case_groups)
        for epoch in range(total_epochs):
            tau, c_target = self._schedule(epoch)
            warmup = self.trainable_gating and epoch < self.config.warmup_epochs
            self._freeze_gating(warmup)
            self.model.train()
            last_losses: dict[str, float] = {}
            if self.config.task == 'survival':
                pending_outputs: list[dict[str, torch.Tensor]] = []
                pending_labels: list[dict[str, torch.Tensor]] = []
                order = torch.randperm(len(survival_case_ids)).tolist()
                update_batch_size = max(1, int(self.config.survival_batch_size)) if self.config.bag_loss == 'nll_surv' else max(2, int(self.config.survival_batch_size))
                for case_pos in order:
                    case_id = survival_case_ids[case_pos]
                    output, label = self._forward_case(self.train_dataset, survival_case_groups[case_id], tau=tau, warmup=warmup, c_target=c_target)
                    pending_outputs.append(output)
                    pending_labels.append(label)
                    if len(pending_outputs) >= update_batch_size:
                        losses = self._survival_batch_loss(pending_outputs, pending_labels, c_target=c_target)
                        self.optimizer.zero_grad(set_to_none=True)
                        losses['total'].backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        last_losses = {key: float(value.detach().cpu()) for key, value in losses.items()}
                        pending_outputs = []
                        pending_labels = []
                if pending_outputs and (self.config.bag_loss == 'nll_surv' or len(pending_outputs) > 1 or not last_losses):
                    losses = self._survival_batch_loss(pending_outputs, pending_labels, c_target=c_target)
                    self.optimizer.zero_grad(set_to_none=True)
                    losses['total'].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    last_losses = {key: float(value.detach().cpu()) for key, value in losses.items()}
            else:
                assert train_loader is not None
                for sample in train_loader:
                    last_losses = self._classification_step(sample, tau=tau, c_target=c_target, warmup=warmup)
            row = {'epoch': float(epoch), 'tau': tau, 'c_target': c_target, **last_losses}
            if (epoch + 1) % self.config.eval_every == 0 and len(self.val_dataset) > 0:
                row.update(evaluate_model(self.model, self.val_dataset, self.config.task, self.device))
            history.append(row)
        checkpoint = output_dir / 'tdae_last.pt'
        torch.save({'model': self.model.state_dict(), 'history': history}, checkpoint)
        return {'checkpoint': str(checkpoint), 'history': history}
