from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    pred_np = probs_np.argmax(axis=1)
    metrics = {
        'accuracy': float(accuracy_score(labels_np, pred_np)),
        'f1_macro': float(f1_score(labels_np, pred_np, average='macro', zero_division=0)),
    }
    try:
        if probs_np.shape[1] == 2:
            metrics['auc'] = float(roc_auc_score(labels_np, probs_np[:, 1]))
        else:
            metrics['auc'] = float(roc_auc_score(labels_np, probs_np, multi_class='ovr'))
    except ValueError:
        metrics['auc'] = float('nan')
    return metrics


def survival_c_index(risks: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> float:
    risk = risks.detach().cpu().flatten().numpy()
    time = times.detach().cpu().flatten().numpy()
    event = events.detach().cpu().flatten().numpy().astype(bool)
    concordant = 0.0
    permissible = 0.0
    n = len(risk)
    for i in range(n):
        for j in range(n):
            if time[i] < time[j] and event[i]:
                permissible += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return float(concordant / permissible) if permissible > 0 else float('nan')


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float('nan')
