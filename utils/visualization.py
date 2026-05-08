from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap, BoundaryNorm


LEVEL_COLORS = ['#d9d9d9', '#8ecae6', '#219ebc', '#023047']


def gating_map_array(gate_values: torch.Tensor, grid_indices: torch.Tensor) -> np.ndarray:
    grid_indices = grid_indices.detach().cpu().long()
    gate_values = gate_values.detach().cpu()
    if gate_values.ndim == 2:
        gate_values = gate_values.argmax(dim=-1)
    gate_values = gate_values.float()
    shape = grid_indices.max(dim=0).values + 1
    canvas = np.full((int(shape[0]), int(shape[1])), np.nan, dtype=np.float32)
    for value, (row, col) in zip(gate_values.tolist(), grid_indices.tolist()):
        canvas[int(row), int(col)] = float(value)
    return canvas


def save_gating_map(gate_values: torch.Tensor, grid_indices: torch.Tensor, save_path: str | Path, num_levels: int = 4) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = gating_map_array(gate_values, grid_indices)
    plt.figure(figsize=(8, 8))
    if num_levels == 4:
        cmap = ListedColormap(LEVEL_COLORS)
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        plt.imshow(canvas, cmap=cmap, norm=norm, interpolation='nearest')
    else:
        plt.imshow(canvas, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path
