from __future__ import annotations

import torch
import torch.nn as nn


class Attn_Net_Gated(nn.Module):
    """Gated attention from CLAM (Lu et al., Nat. BME 2021)."""

    def __init__(self, L: int = 1024, D: int = 256, dropout: float = 0.0, n_classes: int = 1) -> None:
        super().__init__()
        layers_a: list[nn.Module] = [nn.Linear(L, D), nn.Tanh()]
        layers_b: list[nn.Module] = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout > 0:
            layers_a.append(nn.Dropout(dropout))
            layers_b.append(nn.Dropout(dropout))
        self.attention_a = nn.Sequential(*layers_a)
        self.attention_b = nn.Sequential(*layers_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.attention_a(x)
        b = self.attention_b(x)
        attention = self.attention_c(a * b)
        return attention, x
