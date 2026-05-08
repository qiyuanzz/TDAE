from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .losses import survival_logits_to_risk
from utils.metrics import classification_metrics, survival_c_index


def slide_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError('batch_size must be 1 for variable-length WSI features.')
    return batch[0]


def _model_forward(model: torch.nn.Module, sample: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {'tau': 0.1, 'hard_gate': True}
    if hasattr(model, 'c_target'):
        kwargs['c_target'] = getattr(model, 'c_target')
    return model(
        sample['light_feats'].to(device),
        sample['medium_feats'].to(device),
        sample['full_feats'].to(device),
        sample['grid_indices'].to(device),
        **kwargs,
    )


@torch.no_grad()
def evaluate_classification(model: torch.nn.Module, dataset, device: torch.device | str = 'cpu') -> dict[str, float]:
    model.eval()
    device = torch.device(device)
    probs: list[torch.Tensor] = []
    labels: list[int] = []
    costs: list[float] = []
    for sample in DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=slide_collate):
        outputs = _model_forward(model, sample, device)
        probs.append(F.softmax(outputs['logits'], dim=0).detach().cpu())
        labels.append(int(sample['label']))
        costs.append(float(outputs['avg_cost'].detach().cpu()))
    if not probs:
        return {}
    metrics = classification_metrics(torch.stack(probs), torch.tensor(labels))
    metrics['avg_cost'] = float(sum(costs) / len(costs))
    return metrics


# Offset for per-slide grid coordinates inside a multi-slide case bag.
# Must exceed the maximum intra-slide grid extent so the kNN graph never
# connects patches across slides. See trainer._CASE_SLIDE_GRID_OFFSET.
_CASE_SLIDE_GRID_OFFSET = 100_000


def _case_index_groups(dataset) -> "OrderedDict[str, list[int]]":
    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    if hasattr(dataset, 'slides') and 'case_submitter_id' in dataset.slides.columns:
        for idx, case_id in enumerate(dataset.slides['case_submitter_id'].astype(str).tolist()):
            groups.setdefault(case_id, []).append(idx)
    else:
        for idx in range(len(dataset)):
            groups.setdefault(f'slide_{idx}', []).append(idx)
    return groups


def _build_case_bag(dataset, indices: list[int], device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Concatenate all slides of a case into a single bag with offset coords."""
    light_chunks: list[torch.Tensor] = []
    medium_chunks: list[torch.Tensor] = []
    full_chunks: list[torch.Tensor] = []
    coord_chunks: list[torch.Tensor] = []
    label: dict[str, torch.Tensor] | None = None
    for slide_pos, idx in enumerate(indices):
        sample = dataset[idx]
        if label is None:
            label = sample['label']
        light_chunks.append(sample['light_feats'].to(device))
        medium_chunks.append(sample['medium_feats'].to(device))
        full_chunks.append(sample['full_feats'].to(device))
        coords = sample['grid_indices'].to(device).long().clone()
        coords = coords + slide_pos * _CASE_SLIDE_GRID_OFFSET
        coord_chunks.append(coords)
    if label is None:
        raise ValueError('Empty slide list for case.')
    bag = {
        'light_feats': torch.cat(light_chunks, dim=0),
        'medium_feats': torch.cat(medium_chunks, dim=0),
        'full_feats': torch.cat(full_chunks, dim=0),
        'grid_indices': torch.cat(coord_chunks, dim=0),
    }
    return bag, label


@torch.no_grad()
def evaluate_survival(model: torch.nn.Module, dataset, device: torch.device | str = 'cpu') -> dict[str, float]:
    """Per-case bag evaluation matching MCAT/SurvPath.

    All slides of a case are concatenated into one bag (with per-slide grid
    offsets) and a single forward pass produces hazard logits used for the
    case-level risk score. This avoids the per-slide logit/risk averaging
    inconsistency between training (bag concat) and evaluation.
    """
    model.eval()
    device = torch.device(device)
    case_groups = _case_index_groups(dataset)
    if not case_groups:
        return {}
    risks: list[torch.Tensor] = []
    times: list[torch.Tensor] = []
    events: list[torch.Tensor] = []
    costs: list[float] = []
    n_slides = 0
    for indices in case_groups.values():
        bag, label = _build_case_bag(dataset, indices, device)
        kwargs: dict[str, Any] = {'tau': 0.1, 'hard_gate': True}
        if hasattr(model, 'c_target'):
            kwargs['c_target'] = getattr(model, 'c_target')
        outputs = model(
            bag['light_feats'], bag['medium_feats'], bag['full_feats'], bag['grid_indices'],
            **kwargs,
        )
        logits = outputs['logits'].flatten().detach().cpu()
        risk = survival_logits_to_risk(logits.unsqueeze(0))[0] if logits.numel() > 1 else logits[0]
        risks.append(risk)
        times.append(label['time'].flatten().detach().cpu()[0])
        events.append(label['event'].flatten().detach().cpu()[0])
        costs.append(float(outputs['avg_cost'].detach().cpu()))
        n_slides += len(indices)
    case_risks = torch.stack(risks)
    case_times = torch.stack(times)
    case_events = torch.stack(events)
    return {
        'c_index': survival_c_index(case_risks, case_times, case_events),
        'avg_cost': float(sum(costs) / len(costs)) if costs else float('nan'),
        'n_cases': float(len(case_groups)),
        'n_events': float(case_events.sum().item()),
        'n_slides': float(n_slides),
    }


def evaluate_model(model: torch.nn.Module, dataset, task: str, device: torch.device | str = 'cpu') -> dict[str, float]:
    if task == 'classification':
        return evaluate_classification(model, dataset, device)
    if task == 'survival':
        return evaluate_survival(model, dataset, device)
    raise ValueError(f'Unsupported task: {task}')


def measure_wall_time(fn, repeats: int = 3) -> dict[str, float]:
    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return {'mean_seconds': float(sum(times) / len(times)), 'min_seconds': float(min(times))}
