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


@torch.no_grad()
def evaluate_survival(model: torch.nn.Module, dataset, device: torch.device | str = 'cpu') -> dict[str, float]:
    model.eval()
    device = torch.device(device)
    cases: OrderedDict[str, dict[str, Any]] = OrderedDict()
    n_slides = 0
    for sample in DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=slide_collate):
        outputs = _model_forward(model, sample, device)
        case_id = str(sample.get('case_id', f'slide_{n_slides}'))
        entry = cases.setdefault(
            case_id,
            {
                'risks': [],
                'time': sample['label']['time'].flatten().detach().cpu()[0],
                'event': sample['label']['event'].flatten().detach().cpu()[0],
                'costs': [],
            },
        )
        logits = outputs['logits'].flatten().detach().cpu()
        if logits.numel() > 1:
            risk = survival_logits_to_risk(logits.unsqueeze(0))[0]
        else:
            risk = logits[0]
        entry['risks'].append(risk)
        entry['costs'].append(float(outputs['avg_cost'].detach().cpu()))
        n_slides += 1
    if not cases:
        return {}
    case_risks = torch.stack([torch.stack(entry['risks']).mean() for entry in cases.values()])
    case_times = torch.stack([entry['time'] for entry in cases.values()])
    case_events = torch.stack([entry['event'] for entry in cases.values()])
    costs = [cost for entry in cases.values() for cost in entry['costs']]
    return {
        'c_index': survival_c_index(case_risks, case_times, case_events),
        'avg_cost': float(sum(costs) / len(costs)) if costs else float('nan'),
        'n_cases': float(len(cases)),
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
